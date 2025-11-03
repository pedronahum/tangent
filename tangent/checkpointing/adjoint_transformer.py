"""Transform adjoint code to handle checkpointed loops.

This module provides transformations for the adjoint (backward) pass
to use checkpoints and recomputation instead of storing all intermediate values.
"""

import gast
import copy
from typing import Dict, Optional, List, Set
from tangent.analysis.checkpoint_analyzer import CheckpointingPlan, LoopInfo


class AdjointCheckpointTransformer(gast.NodeTransformer):
    """
    Transform adjoint loops to use checkpoints and recomputation.

    This transformer is applied AFTER reverse_ad.py generates the adjoint,
    modifying backward loops to:
    1. Use checkpoints when available
    2. Recompute from checkpoints when needed
    3. Skip storing O(n) intermediate values
    """

    def __init__(self, plan: CheckpointingPlan, primal_loop_map: Dict[int, gast.For]):
        """Initialize transformer.

        Args:
            plan: Checkpointing plan from analysis
            primal_loop_map: Mapping from loop IDs to primal loop nodes
        """
        self.plan = plan
        self.primal_loop_map = primal_loop_map
        self.in_adjoint_loop = False
        self.current_primal_loop_id: Optional[str] = None

    def visit_For(self, node: gast.For) -> gast.For:
        """Transform adjoint for loops."""
        # Check if this is an adjoint loop for a checkpointed primal loop
        # This is a simplified version - in full implementation we'd use annotations

        # For now, we'll detect reverse loops by checking if they iterate in reverse
        is_reverse_loop = self._is_reverse_loop(node)

        if not is_reverse_loop:
            # Not an adjoint loop, leave as-is
            return self.generic_visit(node)

        # This is an adjoint loop - transform it to use checkpoints
        return self._transform_adjoint_loop(node)

    def _is_reverse_loop(self, node: gast.For) -> bool:
        """Check if loop iterates in reverse (indicates adjoint loop)."""
        # Check if iter is reversed(...)
        if isinstance(node.iter, gast.Call):
            if isinstance(node.iter.func, gast.Name):
                if node.iter.func.id == 'reversed':
                    return True
        return False

    def _transform_adjoint_loop(self, node: gast.For) -> gast.For:
        """
        Transform adjoint loop to use checkpoints.

        Original adjoint:
            for i in reversed(range(n)):
                state = pop(_stack, 'state_id')
                # gradient computation

        Transformed adjoint:
            for _iteration in reversed(range(n)):
                # Find nearest checkpoint
                _nearest_cp = find_nearest_checkpoint(_iteration, _checkpoint_positions)

                if _nearest_cp >= 0:
                    # Restore from checkpoint
                    state = _checkpoints[_nearest_cp]['state']

                    # Recompute if needed
                    if _nearest_cp < _iteration:
                        for _j in range(_nearest_cp + 1, _iteration + 1):
                            # Execute forward step
                            state = state + 0.1
                else:
                    # Full recomputation from start
                    state = initial_state
                    for _j in range(_iteration + 1):
                        state = state + 0.1

                # Now compute gradient
                # (original adjoint body with pops removed/replaced)
        """

        # For this simplified version, we'll add checkpoint restoration logic
        # The full implementation would need to:
        # 1. Identify variables that need restoration
        # 2. Generate recomputation code
        # 3. Replace pop operations with checkpoint lookups

        new_body = []

        # Add checkpoint restoration logic at the start
        restoration_code = self._generate_checkpoint_restoration()
        new_body.extend(restoration_code)

        # Add original adjoint body (with modifications)
        # For now, we'll keep the original body as-is
        # In full implementation, we'd transform pop operations
        new_body.extend(copy.deepcopy(node.body))

        # Create new loop with modified body
        new_loop = gast.For(
            target=node.target,
            iter=node.iter,
            body=new_body,
            orelse=node.orelse
        )

        return new_loop

    def _generate_checkpoint_restoration(self) -> List[gast.AST]:
        """Generate code to restore state from checkpoints.

        Generates:
            # This is a placeholder - full implementation would generate
            # actual restoration and recomputation code
        """
        # Placeholder for now
        # Full implementation would generate:
        # - find_nearest_checkpoint call
        # - conditional restoration
        # - recomputation loop if needed

        return []


class CheckpointVariableReplacer(gast.NodeTransformer):
    """
    Replace stack pop operations with checkpoint dict lookups.

    Transforms:
        state = tangent.pop(_stack, 'state_id')

    Into:
        if _iteration in _checkpoint_positions_set:
            state = _checkpoints[_iteration]['state']
        else:
            # Restore from nearest checkpoint and recompute
            _nearest = find_nearest_checkpoint(_iteration, _checkpoint_positions)
            state = _checkpoints[_nearest]['state']
            # Recompute logic here
    """

    def __init__(self, checkpoint_dict_name: str = '_checkpoints',
                 iteration_var: str = '_iteration'):
        self.checkpoint_dict_name = checkpoint_dict_name
        self.iteration_var = iteration_var

    def visit_Assign(self, node: gast.Assign) -> gast.AST:
        """Transform assignments that pop from stack."""
        # Check if RHS is a pop operation
        if isinstance(node.value, gast.Call):
            if self._is_pop_call(node.value):
                # This is a pop - replace with checkpoint lookup
                return self._create_checkpoint_lookup(node)

        return self.generic_visit(node)

    def _is_pop_call(self, node: gast.Call) -> bool:
        """Check if call is a tangent.pop operation."""
        if isinstance(node.func, gast.Attribute):
            if isinstance(node.func.value, gast.Name):
                if node.func.value.id == 'tangent' and node.func.attr == 'pop':
                    return True
        elif isinstance(node.func, gast.Name):
            if node.func.id == 'pop':
                return True
        return False

    def _create_checkpoint_lookup(self, original_assign: gast.Assign) -> gast.AST:
        """Create checkpoint dictionary lookup to replace pop.

        This is a simplified version - full implementation would include
        recomputation logic.
        """
        # For now, return original assignment
        # Full implementation would create conditional checkpoint lookup
        return original_assign
