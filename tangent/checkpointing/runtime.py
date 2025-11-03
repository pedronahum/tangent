"""Runtime support functions for checkpointing.

These functions are called during the execution of checkpointed gradient code.
"""

import numpy as np
import copy
from typing import List, Dict, Any, Optional, Tuple


def find_nearest_checkpoint(iteration: int,
                           checkpoint_positions: List[int]) -> int:
    """Find the nearest checkpoint before or at iteration.

    Args:
        iteration: Current iteration index
        checkpoint_positions: List of checkpoint positions

    Returns:
        Index of nearest checkpoint, or -1 if none found
    """
    valid = [p for p in checkpoint_positions if p <= iteration]
    return max(valid) if valid else -1


def compute_checkpoint_positions(num_iterations: int,
                                num_checkpoints: Optional[int] = None) -> List[int]:
    """Compute optimal checkpoint positions.

    Uses sqrt(n) checkpointing strategy for optimal memory-time tradeoff.

    Args:
        num_iterations: Total number of loop iterations
        num_checkpoints: Number of checkpoints to use (default: sqrt(n))

    Returns:
        List of iteration indices where checkpoints should be placed
    """
    if num_checkpoints is None:
        num_checkpoints = max(1, int(np.sqrt(num_iterations)))

    if num_iterations <= 0 or num_checkpoints <= 0:
        return []

    if num_iterations <= num_checkpoints:
        # Can checkpoint every iteration
        return list(range(num_iterations))

    # Evenly spaced checkpoints
    step = num_iterations / num_checkpoints
    positions = [int((i + 1) * step) - 1 for i in range(num_checkpoints)]

    return positions


def compute_optimal_checkpoints(num_iterations: int) -> int:
    """Compute optimal number of checkpoints.

    Uses sqrt(n) for optimal memory-time tradeoff.

    Args:
        num_iterations: Total number of loop iterations

    Returns:
        Optimal number of checkpoints
    """
    return max(1, int(np.sqrt(num_iterations)))


def checkpoint_state(variables: Dict[str, Any],
                    iteration: int,
                    storage: Dict) -> None:
    """Save checkpoint of current state.

    Args:
        variables: Dictionary of variable names to values
        iteration: Current iteration index
        storage: Dictionary to store checkpoints in
    """
    for var_name, var_value in variables.items():
        storage[(iteration, var_name)] = copy.deepcopy(var_value)


def restore_state(storage: Dict,
                 iteration: int,
                 variables: List[str]) -> Dict[str, Any]:
    """Restore state from checkpoint.

    Args:
        storage: Dictionary containing checkpoints
        iteration: Iteration to restore from
        variables: List of variable names to restore

    Returns:
        Dictionary of restored variable values
    """
    restored = {}
    for var_name in variables:
        key = (iteration, var_name)
        if key in storage:
            restored[var_name] = storage[key]
    return restored


def recompute_from_checkpoint(checkpoint_idx: int,
                              target_idx: int,
                              checkpoint_state: Dict[str, Any],
                              loop_body_func,
                              loop_range: range) -> Dict[str, Any]:
    """Recompute forward pass from checkpoint to target iteration.

    Args:
        checkpoint_idx: Index of checkpoint to start from
        target_idx: Target iteration to recompute to
        checkpoint_state: State at checkpoint
        loop_body_func: Function representing loop body
        loop_range: Range object for the loop

    Returns:
        State at target iteration
    """
    # Start from checkpoint state
    current_state = copy.deepcopy(checkpoint_state)

    # Recompute forward from checkpoint to target
    for i in range(checkpoint_idx + 1, target_idx + 1):
        if i < len(loop_range):
            loop_var = loop_range[i]
            current_state = loop_body_func(current_state, loop_var)

    return current_state


class CheckpointManager:
    """Manager for checkpoint storage and retrieval.

    This class provides a high-level interface for managing checkpoints
    during forward and backward passes.
    """

    def __init__(self, num_iterations: int, num_checkpoints: Optional[int] = None):
        """Initialize checkpoint manager.

        Args:
            num_iterations: Total number of iterations
            num_checkpoints: Number of checkpoints (default: sqrt(n))
        """
        self.num_iterations = num_iterations
        self.num_checkpoints = num_checkpoints or compute_optimal_checkpoints(num_iterations)
        self.positions = compute_checkpoint_positions(num_iterations, self.num_checkpoints)
        self.positions_set = set(self.positions)
        self.storage = {}

    def should_checkpoint(self, iteration: int) -> bool:
        """Check if checkpoint should be saved at this iteration.

        Args:
            iteration: Current iteration index

        Returns:
            True if checkpoint should be saved
        """
        return iteration in self.positions_set

    def save(self, iteration: int, **variables):
        """Save checkpoint at iteration.

        Args:
            iteration: Current iteration index
            **variables: Variables to checkpoint
        """
        checkpoint_state(variables, iteration, self.storage)

    def restore(self, iteration: int, variable_names: List[str]) -> Dict[str, Any]:
        """Restore checkpoint at iteration.

        Args:
            iteration: Iteration to restore
            variable_names: Names of variables to restore

        Returns:
            Dictionary of restored variables
        """
        return restore_state(self.storage, iteration, variable_names)

    def find_nearest(self, iteration: int) -> int:
        """Find nearest checkpoint before or at iteration.

        Args:
            iteration: Current iteration

        Returns:
            Nearest checkpoint index, or -1
        """
        return find_nearest_checkpoint(iteration, self.positions)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about memory usage.

        Returns:
            Dictionary with memory statistics
        """
        return {
            'num_iterations': self.num_iterations,
            'num_checkpoints': self.num_checkpoints,
            'checkpoints_stored': len(set(k[0] for k in self.storage.keys())),
            'reduction_ratio': 1.0 - (self.num_checkpoints / max(1, self.num_iterations)),
            'positions': self.positions
        }
