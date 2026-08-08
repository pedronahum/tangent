"""
Dead Code Elimination for Tangent gradients.

This module implements backward slicing and activity analysis to eliminate
unnecessary computations from gradient functions.
"""

import ast
import gast
from typing import Set, Dict, List, Tuple


class VariableCollector(ast.NodeVisitor):
    """Collect all variables used in an expression."""

    def __init__(self):
        self.variables = set()

    def visit_Name(self, node):
        self.variables.add(node.id)
        self.generic_visit(node)

    @staticmethod
    def collect(node):
        """Helper to collect variables from a node."""
        collector = VariableCollector()
        collector.visit(node)
        return collector.variables


class DefUseAnalyzer:
    """
    Analyze definitions and uses in a function.

    Builds:
    - def_map: line_num -> set of variables defined at that line
    - use_map: line_num -> set of variables used at that line
    - def_sites: var_name -> list of line numbers where it's defined
    """

    def __init__(self, func_ast):
        self.func_ast = func_ast
        self.def_map = {}  # line_num -> {vars defined}
        self.use_map = {}  # line_num -> {vars used}
        self.def_sites = {}  # var_name -> [line_nums]
        self._analyze()

    def _analyze(self):
        """Analyze the AST to build def-use information."""
        for i, stmt in enumerate(self.func_ast.body):
            self._analyze_statement(stmt, i)

    def _analyze_statement(self, stmt, line_num):
        """Analyze a single statement.

        Note: statements nested inside control flow are analyzed under the
        SAME line number as their enclosing top-level statement, so the
        def/use maps must be merged (setdefault + update) rather than
        overwritten, or the outer statement's defs/uses get clobbered.
        """
        if isinstance(stmt, (ast.Assign, gast.Assign)):
            # Variables defined
            defined = set()
            for target in stmt.targets:
                if isinstance(target, (ast.Name, gast.Name)):
                    defined.add(target.id)
                elif isinstance(target, (ast.Tuple, gast.Tuple)):
                    # Handle tuple unpacking: a, b = ...
                    for elt in target.elts:
                        if isinstance(elt, (ast.Name, gast.Name)):
                            defined.add(elt.id)

            # Variables used
            used = VariableCollector.collect(stmt.value)

            self.def_map.setdefault(line_num, set()).update(defined)
            self.use_map.setdefault(line_num, set()).update(used)

            for var in defined:
                self.def_sites.setdefault(var, []).append(line_num)

        elif isinstance(stmt, (ast.AugAssign, gast.AugAssign)):
            # Augmented assignment: x += 1
            if isinstance(stmt.target, (ast.Name, gast.Name)):
                var = stmt.target.id
                self.def_map.setdefault(line_num, set()).add(var)
                self.use_map.setdefault(line_num, set()).update(
                    {var} | VariableCollector.collect(stmt.value))
                self.def_sites.setdefault(var, []).append(line_num)

        elif isinstance(stmt, (ast.Return, gast.Return)):
            # Return uses variables
            if stmt.value:
                used = VariableCollector.collect(stmt.value)
                self.use_map.setdefault(line_num, set()).update(used)

        elif isinstance(stmt, (ast.If, gast.If)):
            # If statement condition uses variables
            cond_vars = VariableCollector.collect(stmt.test)
            self.use_map.setdefault(line_num, set()).update(cond_vars)

            # Recursively analyze body and orelse
            for body_stmt in stmt.body:
                self._analyze_statement(body_stmt, line_num)
            for else_stmt in stmt.orelse:
                self._analyze_statement(else_stmt, line_num)

        elif isinstance(stmt, (ast.For, gast.For)):
            # For loop iterator uses variables
            iter_vars = VariableCollector.collect(stmt.iter)
            self.use_map.setdefault(line_num, set()).update(iter_vars)

            # Loop variable is defined
            if isinstance(stmt.target, (ast.Name, gast.Name)):
                loop_var = stmt.target.id
                self.def_map.setdefault(line_num, set()).add(loop_var)
                self.def_sites.setdefault(loop_var, []).append(line_num)

            # Recursively analyze loop body
            for body_stmt in stmt.body:
                self._analyze_statement(body_stmt, line_num)

        elif isinstance(stmt, (ast.While, gast.While)):
            # While loop condition uses variables
            cond_vars = VariableCollector.collect(stmt.test)
            self.use_map.setdefault(line_num, set()).update(cond_vars)

            # Recursively analyze loop body
            for body_stmt in stmt.body:
                self._analyze_statement(body_stmt, line_num)


class BackwardSlicer:
    """
    Compute backward slice for gradient computations.

    Given a set of target variables (requested gradients), computes
    the minimal set of statements needed to compute them.
    """

    def __init__(self, func_ast, def_use_analyzer):
        self.func_ast = func_ast
        self.def_use = def_use_analyzer
        self.relevant_stmts = set()

    def slice(self, target_vars: Set[str]) -> Set[int]:
        """
        Compute backward slice from target variables.

        Args:
            target_vars: Set of variable names we need (requested gradients)

        Returns:
            Set of statement indices that are relevant
        """
        worklist = list(target_vars)
        visited_vars = set()

        while worklist:
            var = worklist.pop()

            if var in visited_vars:
                continue
            visited_vars.add(var)

            # Find all definitions of this variable
            for def_line in self.def_use.def_sites.get(var, []):
                self.relevant_stmts.add(def_line)

                # Add all variables used in this definition to worklist
                used_vars = self.def_use.use_map.get(def_line, set())
                for used_var in used_vars:
                    if used_var not in visited_vars:
                        worklist.append(used_var)

        return self.relevant_stmts


class ActivityAnalyzer:
    """
    Perform forward and backward activity analysis.

    Forward: Which variables depend on active inputs?
    Backward: Which variables affect active outputs?

    This provides more precise analysis than backward slicing alone.
    """

    def __init__(self, func_ast, active_inputs: Set[str], active_outputs: Set[str]):
        self.func_ast = func_ast
        self.active_inputs = active_inputs
        self.active_outputs = active_outputs
        self.def_use = DefUseAnalyzer(func_ast)

    def forward_analysis(self) -> Set[str]:
        """
        Propagate activity forward from active inputs.
        A variable is active if it transitively depends on active inputs.
        """
        active_vars = set(self.active_inputs)
        changed = True

        while changed:
            changed = False
            for line_num in range(len(self.func_ast.body)):
                defined = self.def_use.def_map.get(line_num, set())
                used = self.def_use.use_map.get(line_num, set())

                # If any used variable is active, defined variables become active
                if any(var in active_vars for var in used):
                    for var in defined:
                        if var not in active_vars:
                            active_vars.add(var)
                            changed = True

        return active_vars

    def backward_analysis(self) -> Set[str]:
        """
        Propagate activity backward from active outputs.
        A variable is active if active outputs transitively depend on it.
        """
        active_vars = set(self.active_outputs)
        changed = True

        while changed:
            changed = False
            # Traverse in reverse order
            for line_num in reversed(range(len(self.func_ast.body))):
                defined = self.def_use.def_map.get(line_num, set())
                used = self.def_use.use_map.get(line_num, set())

                # If any defined variable is active, used variables become active
                if any(var in active_vars for var in defined):
                    for var in used:
                        if var not in active_vars:
                            active_vars.add(var)
                            changed = True

        return active_vars

    def compute_active_variables(self) -> Set[str]:
        """
        Compute which variables are truly active.

        A variable is active if:
        1. It depends on active inputs (forward active), AND
        2. Active outputs depend on it (backward active)

        This intersection gives us the minimal set of active variables.
        """
        forward_active = self.forward_analysis()
        backward_active = self.backward_analysis()

        # Variables that are both forward and backward active
        active = forward_active & backward_active

        return active


# Tape operations whose calls must never be eliminated: the push/pop sequence
# has to stay balanced and in order, regardless of whether the popped value
# ends up used by the requested gradients.
_TAPE_OP_NAMES = frozenset(('push', 'pop', 'push_stack', 'pop_stack'))


def _touches_tape(stmt):
    """Return True if the statement performs any tape push/pop operation."""
    for node in gast.walk(stmt):
        if not isinstance(node, gast.Call):
            continue
        func = node.func
        if isinstance(func, gast.Attribute):
            name = func.attr
        elif isinstance(func, gast.Name):
            name = func.id
        else:
            name = None
        if name in _TAPE_OP_NAMES:
            return True
    return False


class GradientDCE:
    """
    Main DCE optimizer for gradient functions.

    Takes a gradient function AST and list of requested gradients,
    returns optimized AST with dead code eliminated.
    """

    def __init__(self, grad_func_ast, requested_grads: List[str], use_activity_analysis=True, verbose=0):
        self.grad_func_ast = grad_func_ast
        self.requested_grads = set(requested_grads)
        self.use_activity_analysis = use_activity_analysis
        self.verbose = verbose

    def optimize(self):
        """Apply DCE optimization with optional activity analysis."""
        # Step 1: Analyze def-use chains
        analyzer = DefUseAnalyzer(self.grad_func_ast)

        # Step 2: Identify gradient variables we need
        # Convention: gradient of x is named bx or d_x
        # Tangent uses both conventions, so check for both
        gradient_vars = set()
        for var in self.requested_grads:
            gradient_vars.add(f'b{var}')  # Tangent's main convention
            gradient_vars.add(f'd_{var}')  # Alternative convention
            gradient_vars.add(f'_b{var}')  # Another variant

        # Also include the return variable (gradient is returned)
        # Check what variables are used in the return statement
        for i, stmt in enumerate(self.grad_func_ast.body):
            if isinstance(stmt, (ast.Return, gast.Return)):
                if stmt.value:
                    return_vars = VariableCollector.collect(stmt.value)
                    gradient_vars.update(return_vars)

        # Step 3: Optional activity analysis for more precision
        active_vars = None
        if self.use_activity_analysis:
            # Extract function parameters (active inputs)
            active_inputs = self._extract_function_params()
            # Active outputs are the gradient variables we need
            active_outputs = gradient_vars

            # Run activity analysis
            activity = ActivityAnalyzer(self.grad_func_ast, active_inputs, active_outputs)
            active_vars = activity.compute_active_variables()

            # Filter gradient vars to only active ones
            gradient_vars = gradient_vars & active_vars

        # Statements that are kept unconditionally (tape pushes/pops and other
        # side-effecting statements, see _touches_tape and _is_essential) do
        # not participate in the slice otherwise, so the variables they
        # reference - most importantly the stack itself, which is not
        # "active" dataflow-wise - must be added to the slice seed. Otherwise
        # their definitions get eliminated and the generated code fails with
        # NameError/stack underflow at runtime.
        for stmt in self.grad_func_ast.body:
            if _touches_tape(stmt) or self._is_essential(stmt):
                gradient_vars.update(VariableCollector.collect(stmt))

        # Step 4: Backward slice from these gradients
        slicer = BackwardSlicer(self.grad_func_ast, analyzer)
        relevant_stmts = slicer.slice(gradient_vars)

        # Step 4: Count statements before optimization
        original_count = len(self.grad_func_ast.body)

        # Step 5: Remove irrelevant statements
        # Always keep: function def, return statements, docstrings, and any
        # statement that performs tape push/pop operations.
        optimized_body = []
        for i, stmt in enumerate(self.grad_func_ast.body):
            # Always keep essential statements and tape operations
            if self._is_essential(stmt) or _touches_tape(stmt):
                optimized_body.append(stmt)
            # Keep relevant statements
            elif i in relevant_stmts:
                # For control flow, recursively prune the body
                stmt = self._prune_control_flow(stmt, relevant_stmts)
                optimized_body.append(stmt)

        self.grad_func_ast.body = optimized_body

        # Step 6: Report statistics
        eliminated = original_count - len(optimized_body)
        if eliminated > 0 and self.verbose >= 1:
            print(f"DCE: Eliminated {eliminated} statements ({original_count} → {len(optimized_body)})")

        return self.grad_func_ast

    def _prune_control_flow(self, stmt, relevant_stmts):
        """
        Recursively prune dead code from control flow statements.

        For if/for/while statements, we keep the control flow structure
        but may remove dead code from the body.
        """
        # For now, keep control flow statements as-is
        # Phase 3 enhancement: could recursively optimize nested bodies
        # This would require more sophisticated analysis

        # Simple approach: keep the statement but note it contains relevant code
        return stmt

    def _extract_function_params(self):
        """Extract parameter names from function signature."""
        params = set()
        # Look for function definition (should be the AST itself for gradient functions)
        if hasattr(self.grad_func_ast, 'args'):
            for arg in self.grad_func_ast.args.args:
                if hasattr(arg, 'id'):
                    params.add(arg.id)
                elif hasattr(arg, 'arg'):
                    params.add(arg.arg)
        return params

    def _is_essential(self, stmt):
        """Check if statement is essential (return, control flow, docstring, etc.)."""
        # Return statements are essential
        if isinstance(stmt, (ast.Return, gast.Return)):
            return True

        # Function definitions are essential
        if isinstance(stmt, (ast.FunctionDef, gast.FunctionDef)):
            return True

        # Expression statements exist purely for their side effects. In
        # adjoint code these are the tape pushes (tangent.push(_stack, ...));
        # eliminating them would leave their pops orphaned and the stack
        # would underflow at runtime. Augmented assignments are likewise
        # side-effecting.
        if isinstance(stmt, (ast.Expr, gast.Expr)):
            return True
        if isinstance(stmt, (ast.AugAssign, gast.AugAssign)):
            return True

        # Writes through a subscript or attribute mutate state that the
        # def-use analysis above does not track (it only records plain Name
        # targets), so they must not be eliminated either.
        if isinstance(stmt, (ast.Assign, gast.Assign)):
            for target in stmt.targets:
                if isinstance(target, (ast.Subscript, gast.Subscript,
                                       ast.Attribute, gast.Attribute)):
                    return True

        # Docstrings are essential
        if isinstance(stmt, (ast.Expr, gast.Expr)):
            if isinstance(stmt.value, (ast.Constant, gast.Constant)):
                return True

        return False


def apply_dce(grad_func_ast, requested_grads: List[str], verbose: int = 0):
    """
    Apply Dead Code Elimination to a gradient function.

    Args:
        grad_func_ast: AST of the gradient function
        requested_grads: List of variable names we want gradients for
        verbose: Verbosity level; >= 1 prints elimination statistics

    Returns:
        Optimized AST with dead code eliminated
    """
    optimizer = GradientDCE(grad_func_ast, requested_grads, verbose=verbose)
    return optimizer.optimize()
