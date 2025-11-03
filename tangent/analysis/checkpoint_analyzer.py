"""Analyze code to determine checkpointing opportunities and requirements.

This module provides the first stage of the checkpointing pipeline:
analyzing AST to determine which loops can and should be checkpointed.
"""

import ast
import gast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
import numpy as np


@dataclass
class LoopInfo:
    """Information about a loop that could be checkpointed."""
    loop_id: str
    loop_var: Optional[str]
    num_iterations: Optional[int]  # None if dynamic
    modified_variables: Set[str]
    invariant_operations: List[gast.AST]
    variant_operations: List[gast.AST]
    nested_depth: int
    parent_loop_id: Optional[str]
    can_checkpoint: bool
    checkpoint_positions: Optional[List[int]]


@dataclass
class CheckpointingPlan:
    """Complete checkpointing plan for a function."""
    loops: Dict[str, LoopInfo] = field(default_factory=dict)
    variable_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    memory_estimate: int = 0
    recommended_strategy: str = 'none'  # 'revolve', 'dynamic', or 'none'


class CheckpointAnalyzer(gast.NodeVisitor):
    """
    First pass: Analyze AST to determine what can/should be checkpointed.

    This analyzer walks through the AST and identifies loops that are candidates
    for checkpointing based on their structure, iteration count, and operations.
    """

    def __init__(self, memory_budget: Optional[int] = None, min_loop_size: int = 100):
        self.loops: Dict[str, LoopInfo] = {}
        self.current_loop_stack: List[str] = []
        self.variable_writes: Dict[str, Set[str]] = {}  # var -> loops that modify it
        self.variable_reads: Dict[str, Set[str]] = {}   # var -> loops that read it
        self.memory_budget = memory_budget
        self.min_loop_size = min_loop_size

    def analyze(self, tree: gast.Module) -> CheckpointingPlan:
        """Main entry point for analysis."""
        self.visit(tree)
        return self._create_plan()

    def visit_For(self, node: gast.For) -> None:
        """Analyze for loops."""
        loop_id = f"loop_{id(node)}"

        # Determine iteration count if possible
        num_iterations = self._extract_iteration_count(node.iter)

        # Extract loop variable name
        if isinstance(node.target, gast.Name):
            loop_var = node.target.id
        else:
            loop_var = None

        # Analyze loop body
        modified_vars = self._find_modified_variables(node.body)
        invariant_ops, variant_ops = self._classify_operations(node.body, loop_var)

        # Check if loop can be checkpointed
        can_checkpoint = self._can_checkpoint_loop(
            modified_vars, invariant_ops, variant_ops, num_iterations
        )

        # Calculate optimal checkpoint positions if applicable
        checkpoint_positions = None
        if can_checkpoint and num_iterations:
            checkpoint_positions = self._compute_checkpoint_positions(
                num_iterations, len(modified_vars)
            )

        self.loops[loop_id] = LoopInfo(
            loop_id=loop_id,
            loop_var=loop_var,
            num_iterations=num_iterations,
            modified_variables=modified_vars,
            invariant_operations=invariant_ops,
            variant_operations=variant_ops,
            nested_depth=len(self.current_loop_stack),
            parent_loop_id=self.current_loop_stack[-1] if self.current_loop_stack else None,
            can_checkpoint=can_checkpoint,
            checkpoint_positions=checkpoint_positions
        )

        # Track variable dependencies
        for var in modified_vars:
            if var not in self.variable_writes:
                self.variable_writes[var] = set()
            self.variable_writes[var].add(loop_id)

        # Recurse into loop body
        self.current_loop_stack.append(loop_id)
        self.generic_visit(node)
        self.current_loop_stack.pop()

    def _extract_iteration_count(self, iter_node: gast.AST) -> Optional[int]:
        """Try to determine loop iteration count statically."""
        if isinstance(iter_node, gast.Call):
            if isinstance(iter_node.func, gast.Name) and iter_node.func.id == 'range':
                if len(iter_node.args) == 1:
                    # range(n)
                    if isinstance(iter_node.args[0], gast.Constant):
                        return iter_node.args[0].value
                    elif hasattr(gast, 'Num') and isinstance(iter_node.args[0], gast.Num):
                        return iter_node.args[0].n
                elif len(iter_node.args) == 2:
                    # range(start, stop)
                    arg0 = iter_node.args[0]
                    arg1 = iter_node.args[1]

                    # Try gast.Constant (gast >= 0.3.3)
                    if isinstance(arg0, gast.Constant) and isinstance(arg1, gast.Constant):
                        return arg1.value - arg0.value
                    # Try gast.Num (gast < 0.3.3)
                    elif hasattr(gast, 'Num') and isinstance(arg0, gast.Num) and isinstance(arg1, gast.Num):
                        return arg1.n - arg0.n
        return None

    def _find_modified_variables(self, body: List[gast.AST]) -> Set[str]:
        """Find all variables modified in loop body."""
        modified = set()
        for stmt in body:
            for node in gast.walk(stmt):
                if isinstance(node, gast.Assign):
                    for target in node.targets:
                        if isinstance(target, gast.Name):
                            modified.add(target.id)
                elif isinstance(node, gast.AugAssign):
                    if isinstance(node.target, gast.Name):
                        modified.add(node.target.id)
        return modified

    def _classify_operations(self, body: List[gast.AST],
                           loop_var: Optional[str]) -> Tuple[List[gast.AST], List[gast.AST]]:
        """Classify operations as loop-invariant or loop-variant."""
        invariant_ops = []
        variant_ops = []

        for stmt in body:
            if self._is_invariant_operation(stmt, loop_var):
                invariant_ops.append(stmt)
            else:
                variant_ops.append(stmt)

        return invariant_ops, variant_ops

    def _is_invariant_operation(self, node: gast.AST, loop_var: Optional[str]) -> bool:
        """Check if operation is loop-invariant (doesn't depend on loop variable)."""
        if loop_var is None:
            return False

        # Walk the node and check if it references the loop variable
        for child in gast.walk(node):
            if isinstance(child, gast.Name) and child.id == loop_var:
                return False
        return True

    def _can_checkpoint_loop(self, modified_vars: Set[str],
                            invariant_ops: List[gast.AST],
                            variant_ops: List[gast.AST],
                            num_iterations: Optional[int]) -> bool:
        """Determine if loop can be safely checkpointed."""

        # Don't checkpoint if iteration count is too small
        if num_iterations is not None and num_iterations < self.min_loop_size:
            return False

        # Check for disqualifying patterns in ALL operations (both invariant and variant)
        all_ops = invariant_ops + variant_ops
        for op in all_ops:
            # Don't checkpoint loops with I/O operations
            if self._has_io_operation(op):
                return False
            # Don't checkpoint loops with unknown function calls (for now)
            # We could relax this later
            if self._has_complex_function_call(op):
                return False

        # Loop can be checkpointed if it modifies trackable state
        return len(modified_vars) > 0

    def _has_io_operation(self, node: gast.AST) -> bool:
        """Check if node contains I/O operations."""
        for child in gast.walk(node):
            if isinstance(child, gast.Call):
                if isinstance(child.func, gast.Name):
                    # Common I/O functions
                    if child.func.id in ['print', 'open', 'input', 'write', 'read']:
                        return True
        return False

    def _has_complex_function_call(self, node: gast.AST) -> bool:
        """Check if node has complex function calls we can't handle yet."""
        for child in gast.walk(node):
            if isinstance(child, gast.Call):
                # Allow numpy and common math operations
                if isinstance(child.func, gast.Attribute):
                    # np.tanh, np.exp, etc. are OK
                    if isinstance(child.func.value, gast.Name):
                        if child.func.value.id in ['np', 'numpy', 'math']:
                            continue
                    return True  # Other attribute calls are complex
                elif isinstance(child.func, gast.Name):
                    # Allow built-in math functions
                    if child.func.id in ['abs', 'min', 'max', 'sum', 'len', 'range']:
                        continue
                    # Other function calls might have side effects
                    return True
        return False

    def _compute_checkpoint_positions(self, num_iterations: int,
                                     num_variables: int) -> List[int]:
        """Compute optimal checkpoint positions using Revolve algorithm."""
        # For now, use simple sqrt(n) checkpointing
        # Later we can integrate the full Revolve algorithm
        num_checkpoints = max(1, int(np.sqrt(num_iterations)))

        if num_iterations <= num_checkpoints:
            # Can checkpoint every iteration
            return list(range(num_iterations))

        # Evenly spaced checkpoints
        step = num_iterations / num_checkpoints
        positions = [int((i + 1) * step) - 1 for i in range(num_checkpoints)]

        return positions

    def _create_plan(self) -> CheckpointingPlan:
        """Create final checkpointing plan from analysis."""
        plan = CheckpointingPlan()
        plan.loops = self.loops
        plan.variable_dependencies = self.variable_writes

        # Determine if checkpointing is recommended
        checkpointable_loops = [
            loop for loop in self.loops.values()
            if loop.can_checkpoint
        ]

        if checkpointable_loops:
            plan.recommended_strategy = 'revolve'

            # Estimate memory savings
            total_iterations = sum(
                loop.num_iterations or 0
                for loop in checkpointable_loops
            )
            checkpointed_iterations = sum(
                len(loop.checkpoint_positions) if loop.checkpoint_positions else 0
                for loop in checkpointable_loops
            )

            if total_iterations > 0:
                # Rough estimate: assume each iteration stores 1KB
                plan.memory_estimate = (total_iterations - checkpointed_iterations) * 1024
        else:
            plan.recommended_strategy = 'none'

        return plan
