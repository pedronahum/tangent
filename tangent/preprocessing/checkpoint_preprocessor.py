"""Transform AST to checkpoint-ready form BEFORE reverse_ad.py sees it.

This module implements the key insight: instead of modifying how reverse_ad.py works,
we transform the loop into a form that naturally generates the right code when
processed by reverse_ad.py.
"""

import gast
import copy
from typing import Dict, Optional, List, Set
from tangent.analysis.checkpoint_analyzer import CheckpointingPlan, LoopInfo


class CheckpointPreprocessor(gast.NodeTransformer):
    """
    Transform loops into checkpoint-ready form.
    This runs BEFORE reverse_ad.py.

    KEY INSIGHT: Instead of modifying how reverse_ad.py works,
    we transform the loop into a form that naturally generates
    the right code when processed by reverse_ad.py.
    """

    def __init__(self, plan: CheckpointingPlan):
        self.plan = plan
        self.current_loop_id: Optional[str] = None
        self.loop_id_map: Dict[int, str] = {}
        self.function_counter = 0
        self.replacement_statements: Dict[int, List[gast.AST]] = {}
        self.extracted_functions: List[gast.FunctionDef] = []  # Module-level functions

    def visit_Module(self, node: gast.Module) -> gast.Module:
        """Visit module and add extracted functions at module level."""
        # First visit all children to extract loop bodies
        self.generic_visit(node)

        # Add extracted functions at the beginning of the module
        if self.extracted_functions:
            node.body = self.extracted_functions + node.body

        return node

    def visit_FunctionDef(self, node: gast.FunctionDef) -> gast.FunctionDef:
        """Visit function and handle loop replacements that need multiple statements."""
        # First, visit all children normally to mark loops for replacement
        self.generic_visit(node)

        # Now rebuild the body, expanding any loops that were marked for replacement
        new_body = []
        for stmt in node.body:
            if id(stmt) in self.replacement_statements:
                # Replace this single statement with multiple statements
                new_body.extend(self.replacement_statements[id(stmt)])
            else:
                new_body.append(stmt)

        node.body = new_body
        return node

    def visit_For(self, node: gast.For) -> gast.AST:
        """Transform checkpointable loops."""
        loop_id = f"loop_{id(node)}"
        self.loop_id_map[id(node)] = loop_id

        if loop_id not in self.plan.loops:
            # Not analyzed, leave as-is
            return self.generic_visit(node)

        loop_info = self.plan.loops[loop_id]

        if not loop_info.can_checkpoint:
            # Can't checkpoint, leave as-is
            return self.generic_visit(node)

        # Transform to checkpointable form
        # Store the replacement statements for later
        statements = self._create_checkpointed_loop_statements(node, loop_info)
        self.replacement_statements[id(node)] = statements

        # Return the node itself - it will be replaced in visit_FunctionDef
        return node

    def _create_checkpointed_loop_statements(self, node: gast.For,
                                            loop_info: LoopInfo) -> List[gast.AST]:
        """
        Transform a regular loop into checkpoint-ready form with body extraction.

        Original:
            for i in range(n):
                state = state + 0.1

        Transformed (Phase 4b with body extraction):
            # Extract loop body as function
            def _loop_body_XXX(state, i):
                state = state + 0.1
                return state

            # Checkpoint setup
            _checkpoint_positions = [...]
            _checkpoint_positions_set = set(_checkpoint_positions)
            _checkpoints = {}
            _iteration_index = 0

            # Loop with checkpointing
            for i in range(n):
                if _iteration_index in _checkpoint_positions_set:
                    _checkpoints[_iteration_index] = {
                        'state': copy.deepcopy(state),
                        'iteration': _iteration_index
                    }

                # Call extracted body function
                state = _loop_body_XXX(state, i)

                _iteration_index += 1

        Returns:
            List of AST statements that replace the original loop
        """

        statements = []

        # 1. Extract loop body as a function (NEW for Phase 4b!)
        # Add to module-level functions instead of local statements
        body_function = self._extract_loop_body_as_function(node, loop_info)
        if body_function:
            self.extracted_functions.append(body_function)

        # 2. Checkpoint setup code
        checkpoint_setup = self._create_checkpoint_setup(node, loop_info)
        statements.extend(checkpoint_setup)

        # 3. Create modified loop with checkpointing
        modified_loop = self._create_checkpoint_aware_loop(node, loop_info, body_function)
        statements.append(modified_loop)

        return statements

    def _extract_loop_body_as_function(self, node: gast.For,
                                      loop_info: LoopInfo) -> Optional[gast.FunctionDef]:
        """
        Extract loop body as a standalone function for recomputation.

        Transform:
            for i in range(n):
                state = state + 0.1
                accumulator = accumulator * decay

        Into:
            def _loop_body_XXXX(state, accumulator, i):
                state = state + 0.1
                accumulator = accumulator * decay
                return state, accumulator

        Args:
            node: The For loop node
            loop_info: Loop information from analysis

        Returns:
            FunctionDef node or None if extraction not possible
        """
        # Generate unique function name
        func_name = f"_loop_body_{id(node)}"

        # Get loop variable name
        if isinstance(node.target, gast.Name):
            loop_var = node.target.id
        else:
            # Complex target (tuple unpacking etc) - skip for now
            return None

        # Get modified variables from analysis
        modified_vars = sorted(loop_info.modified_variables)

        if not modified_vars:
            # No variables modified - no need for function
            return None

        # Create function parameters: modified variables + loop variable
        # Note: In gast, function arguments are Name nodes, not ast.arg nodes
        params = []
        for var in modified_vars:
            params.append(gast.Name(id=var, ctx=gast.Param(), annotation=None))
        params.append(gast.Name(id=loop_var, ctx=gast.Param(), annotation=None))

        # Clone the loop body
        new_body = [copy.deepcopy(stmt) for stmt in node.body]

        # Add return statement with modified variables
        if len(modified_vars) == 1:
            # Single return value
            return_value = gast.Name(id=modified_vars[0], ctx=gast.Load(), annotation=None)
        else:
            # Multiple return values as tuple
            return_value = gast.Tuple(
                elts=[gast.Name(id=var, ctx=gast.Load(), annotation=None)
                     for var in modified_vars],
                ctx=gast.Load()
            )

        return_stmt = gast.Return(value=return_value)
        new_body.append(return_stmt)

        # Create function definition
        func_def = gast.FunctionDef(
            name=func_name,
            args=gast.arguments(
                args=params,
                posonlyargs=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=new_body,
            decorator_list=[],
            returns=None,
            type_comment=None
        )

        # Copy location information from the original loop node
        gast.copy_location(func_def, node)

        return func_def

    def _create_checkpoint_setup(self, node: gast.For,
                                loop_info: LoopInfo) -> List[gast.AST]:
        """
        Create checkpoint setup code.

        Generates:
            _num_checkpoints = compute_optimal_checkpoints(n)
            _checkpoint_positions = compute_checkpoint_positions(n, _num_checkpoints)
            _checkpoint_positions_set = set(_checkpoint_positions)
            _checkpoints = {}
            _iteration_index = 0
        """
        statements = []

        # For static iteration counts, we can compute positions ahead of time
        if loop_info.num_iterations is not None:
            # Static: precompute positions
            num_checkpoints = max(1, int(loop_info.num_iterations ** 0.5))
            positions = loop_info.checkpoint_positions or []

            # _checkpoint_positions = [...]
            positions_list = gast.List(
                elts=[gast.Constant(value=p, kind=None) for p in positions],
                ctx=gast.Load()
            )
            assign_positions = gast.Assign(
                targets=[gast.Name(id='_checkpoint_positions', ctx=gast.Store(),
                                 annotation=None)],
                value=positions_list,
                type_comment=None
            )
            statements.append(assign_positions)

        else:
            # Dynamic: compute at runtime
            # _num_checkpoints = compute_optimal_checkpoints(len(iter))
            # This is more complex - for now we'll use a placeholder
            pass

        # _checkpoint_positions_dict = {pos: True for pos in _checkpoint_positions}
        # Since dict comprehensions might not be supported, build dict from list of positions
        # Create: _checkpoint_positions_dict = {}
        # Then in loop: if _checkpoint_positions_dict.get(_iteration_index, False):
        #
        # Actually, simpler: Create dict directly: {13: True, 27: True, ...}
        if loop_info.num_iterations is not None:
            positions = loop_info.checkpoint_positions or []
            # Create dict: {13: True, 27: True, ...}
            checkpoint_dict_items = []
            for pos in positions:
                checkpoint_dict_items.append((
                    gast.Constant(value=pos, kind=None),
                    gast.Constant(value=True, kind=None)
                ))

            assign_dict = gast.Assign(
                targets=[gast.Name(id='_checkpoint_positions_dict', ctx=gast.Store(),
                                 annotation=None)],
                value=gast.Dict(
                    keys=[k for k, v in checkpoint_dict_items],
                    values=[v for k, v in checkpoint_dict_items]
                ),
                type_comment=None
            )
            statements.append(assign_dict)

        # _checkpoints = {}
        assign_checkpoints = gast.Assign(
            targets=[gast.Name(id='_checkpoints', ctx=gast.Store(), annotation=None)],
            value=gast.Dict(keys=[], values=[]),
            type_comment=None
        )
        statements.append(assign_checkpoints)

        # _iteration_index = 0
        assign_index = gast.Assign(
            targets=[gast.Name(id='_iteration_index', ctx=gast.Store(),
                             annotation=None)],
            value=gast.Constant(value=0, kind=None),
            type_comment=None
        )
        statements.append(assign_index)

        return statements

    def _create_checkpoint_aware_loop(self, node: gast.For,
                                     loop_info: LoopInfo,
                                     body_function: Optional[gast.FunctionDef] = None) -> gast.For:
        """
        Create modified loop that includes checkpoint storage.

        If body_function is provided (Phase 4b):
            if _iteration_index in _checkpoint_positions_set:
                _checkpoints[_iteration_index] = {'state': copy.deepcopy(state)}

            state = _loop_body_XXX(state, i)  # Call extracted function

            _iteration_index += 1

        Otherwise (Phase 4a):
            if _iteration_index in _checkpoint_positions_set:
                _checkpoints[_iteration_index] = {'state': copy.deepcopy(state)}

            state = state + 0.1  # Inline original body

            _iteration_index += 1
        """
        new_body = []

        # Add checkpoint storage at beginning of loop
        checkpoint_save = self._create_checkpoint_save_code(loop_info)
        new_body.append(checkpoint_save)

        if body_function:
            # Phase 4b: Call extracted body function
            body_call_stmt = self._create_body_function_call(
                body_function, loop_info, node.target
            )
            if body_call_stmt:
                new_body.append(body_call_stmt)
            else:
                # Fallback to inline body
                new_body.extend(copy.deepcopy(node.body))
        else:
            # Phase 4a: Use original inline body
            new_body.extend(copy.deepcopy(node.body))

        # Add iteration index increment at end
        increment = gast.AugAssign(
            target=gast.Name(id='_iteration_index', ctx=gast.Store(), annotation=None),
            op=gast.Add(),
            value=gast.Constant(value=1, kind=None)
        )
        new_body.append(increment)

        # Create new loop with modified body
        new_loop = gast.For(
            target=node.target,
            iter=copy.deepcopy(node.iter),
            body=new_body,
            orelse=node.orelse,  # Preserve any else clause
            type_comment=None
        )

        # Copy location from original loop
        gast.copy_location(new_loop, node)

        return new_loop

    def _create_body_function_call(self, body_function: gast.FunctionDef,
                                   loop_info: LoopInfo,
                                   loop_target: gast.AST) -> Optional[gast.AST]:
        """
        Create statement that calls the extracted body function.

        For single modified variable:
            state = _loop_body_XXX(state, i)

        For multiple modified variables:
            state, accumulator = _loop_body_XXX(state, accumulator, i)

        Args:
            body_function: The extracted function definition
            loop_info: Loop information
            loop_target: The loop target variable (e.g., 'i')

        Returns:
            Assignment statement or None if can't create
        """
        func_name = body_function.name
        modified_vars = sorted(loop_info.modified_variables)

        if not modified_vars:
            return None

        # Get loop variable name
        if isinstance(loop_target, gast.Name):
            loop_var = loop_target.id
        else:
            return None

        # Create function call: _loop_body_XXX(state, accumulator, i)
        call_args = []

        # Add modified variables as arguments
        for var in modified_vars:
            call_args.append(gast.Name(id=var, ctx=gast.Load(), annotation=None))

        # Add loop variable as last argument
        call_args.append(gast.Name(id=loop_var, ctx=gast.Load(), annotation=None))

        func_call = gast.Call(
            func=gast.Name(id=func_name, ctx=gast.Load(), annotation=None),
            args=call_args,
            keywords=[]
        )

        # Create assignment: var = func(...) or (var1, var2) = func(...)
        if len(modified_vars) == 1:
            # Single assignment
            targets = [gast.Name(id=modified_vars[0], ctx=gast.Store(), annotation=None)]
        else:
            # Tuple assignment
            targets = [gast.Tuple(
                elts=[gast.Name(id=var, ctx=gast.Store(), annotation=None)
                     for var in modified_vars],
                ctx=gast.Store()
            )]

        assign_stmt = gast.Assign(
            targets=targets,
            value=func_call,
            type_comment=None
        )

        return assign_stmt

    def _create_checkpoint_save_code(self, loop_info: LoopInfo) -> gast.If:
        """
        Create code to save checkpoint if at checkpoint position.

        Generates:
            if _iteration_index in _checkpoint_positions_set:
                _checkpoints[_iteration_index] = {
                    'state': copy.deepcopy(state),
                    ...
                }
        """
        # Build dictionary of variables to save
        dict_items = []
        for var_name in sorted(loop_info.modified_variables):
            # Create key-value pair: 'var_name': copy.deepcopy(var_name)
            key = gast.Constant(value=var_name, kind=None)
            value = gast.Call(
                func=gast.Attribute(
                    value=gast.Name(id='copy', ctx=gast.Load(), annotation=None),
                    attr='deepcopy',
                    ctx=gast.Load()
                ),
                args=[gast.Name(id=var_name, ctx=gast.Load(), annotation=None)],
                keywords=[]
            )
            dict_items.append((key, value))

        # Create dictionary
        checkpoint_dict = gast.Dict(
            keys=[k for k, v in dict_items],
            values=[v for k, v in dict_items]
        )

        # Create assignment: _checkpoints[_iteration_index] = {...}
        # Note: gast compatibility - newer versions don't use Index wrapper
        if hasattr(gast, 'Index'):
            # Older gast (< 0.4.0)
            slice_node = gast.Index(value=gast.Name(id='_iteration_index',
                                                   ctx=gast.Load(), annotation=None))
        else:
            # Newer gast (>= 0.4.0)
            slice_node = gast.Name(id='_iteration_index',
                                  ctx=gast.Load(), annotation=None)

        save_stmt = gast.Assign(
            targets=[gast.Subscript(
                value=gast.Name(id='_checkpoints', ctx=gast.Load(), annotation=None),
                slice=slice_node,
                ctx=gast.Store()
            )],
            value=checkpoint_dict,
            type_comment=None
        )

        # Create if statement using dict.get() instead of 'in' operator
        # if _checkpoint_positions_dict.get(_iteration_index, False):
        if_stmt = gast.If(
            test=gast.Call(
                func=gast.Attribute(
                    value=gast.Name(id='_checkpoint_positions_dict', ctx=gast.Load(), annotation=None),
                    attr='get',
                    ctx=gast.Load()
                ),
                args=[
                    gast.Name(id='_iteration_index', ctx=gast.Load(), annotation=None),
                    gast.Constant(value=False, kind=None)
                ],
                keywords=[]
            ),
            body=[save_stmt],
            orelse=[]
        )

        return if_stmt
