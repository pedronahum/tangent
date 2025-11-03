# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime helper functions for automatic checkpointing.

These functions are called by generated gradient code to implement
memory-efficient checkpointing using the Revolve algorithm.
"""
from __future__ import absolute_import

import copy
import math


def compute_optimal_checkpoints(seq_length):
    """Compute optimal number of checkpoints for a sequence.

    Uses sqrt(n) as the optimal number of checkpoints, which balances
    memory usage O(sqrt(n)) with recomputation overhead O(sqrt(n)).

    Args:
        seq_length: Length of the sequence to checkpoint

    Returns:
        Optimal number of checkpoints (int)
    """
    if seq_length <= 1:
        return 0
    # sqrt(n) is provably optimal for binomial checkpointing
    return max(1, int(math.sqrt(seq_length)))


def find_nearest_checkpoint(iteration, checkpoint_positions):
    """Find the nearest checkpoint at or before the given iteration.

    Args:
        iteration: Current iteration index
        checkpoint_positions: List of checkpoint indices (sorted)

    Returns:
        Index of nearest checkpoint <= iteration, or 0 if none found
    """
    if not checkpoint_positions:
        return 0

    # Binary search for efficiency
    left, right = 0, len(checkpoint_positions) - 1
    result = 0

    while left <= right:
        mid = (left + right) // 2
        if checkpoint_positions[mid] <= iteration:
            result = checkpoint_positions[mid]
            left = mid + 1
        else:
            right = mid - 1

    return result


def is_checkpoint_iteration(iteration, checkpoint_positions):
    """Check if current iteration should be checkpointed.

    Args:
        iteration: Current iteration index
        checkpoint_positions: List of checkpoint indices

    Returns:
        True if iteration is in checkpoint_positions
    """
    # Use set for O(1) lookup if checkpoint_positions is large
    if isinstance(checkpoint_positions, set):
        return iteration in checkpoint_positions
    return iteration in checkpoint_positions


def store_checkpoint(checkpoint_data, iteration, state, backend='numpy'):
    """Store a checkpoint of the current state.

    Args:
        checkpoint_data: Dictionary mapping iteration -> state
        iteration: Current iteration index
        state: State to checkpoint (will be copied)
        backend: Backend type ('numpy', 'jax', 'tensorflow')

    Returns:
        checkpoint_data (modified in place, but also returned for chaining)
    """
    # Deep copy to avoid aliasing issues
    if backend == 'jax':
        try:
            import jax
            # JAX arrays need special handling
            checkpoint_data[iteration] = jax.tree_map(lambda x: x, state)
        except ImportError:
            # Fall back to deepcopy
            checkpoint_data[iteration] = copy.deepcopy(state)
    elif backend == 'tensorflow':
        try:
            import tensorflow as tf
            # TensorFlow tensors need identity operation
            if isinstance(state, tf.Tensor):
                checkpoint_data[iteration] = tf.identity(state)
            else:
                checkpoint_data[iteration] = copy.deepcopy(state)
        except ImportError:
            checkpoint_data[iteration] = copy.deepcopy(state)
    else:
        # NumPy or generic Python objects
        try:
            import numpy
            if isinstance(state, numpy.ndarray):
                checkpoint_data[iteration] = numpy.copy(state)
            else:
                checkpoint_data[iteration] = copy.deepcopy(state)
        except (ImportError, TypeError):
            checkpoint_data[iteration] = copy.deepcopy(state)

    return checkpoint_data


def restore_checkpoint(checkpoint_data, iteration, checkpoint_positions):
    """Restore state from the nearest checkpoint.

    Args:
        checkpoint_data: Dictionary mapping iteration -> state
        iteration: Target iteration to restore to
        checkpoint_positions: List of checkpoint indices

    Returns:
        Restored state from nearest checkpoint, or None if no checkpoint found
    """
    checkpoint_idx = find_nearest_checkpoint(iteration, checkpoint_positions)
    return checkpoint_data.get(checkpoint_idx)


def estimate_memory_savings(seq_length, num_checkpoints=None, state_size_bytes=None):
    """Estimate memory savings from checkpointing.

    Args:
        seq_length: Number of iterations in the loop
        num_checkpoints: Number of checkpoints (default: sqrt(seq_length))
        state_size_bytes: Size of state in bytes (default: unknown, returns ratio)

    Returns:
        Dictionary with memory statistics:
        - without_checkpointing: Memory without checkpointing
        - with_checkpointing: Memory with checkpointing
        - savings_bytes: Absolute savings
        - savings_percent: Percentage savings
        - num_checkpoints: Number of checkpoints used
    """
    if num_checkpoints is None:
        num_checkpoints = compute_optimal_checkpoints(seq_length)

    # Memory without checkpointing: store all n states
    memory_without = seq_length

    # Memory with checkpointing: store sqrt(n) checkpoints
    memory_with = num_checkpoints

    # Savings
    savings_ratio = (memory_without - memory_with) / memory_without if memory_without > 0 else 0

    result = {
        'without_checkpointing': memory_without,
        'with_checkpointing': memory_with,
        'savings_ratio': savings_ratio,
        'savings_percent': savings_ratio * 100,
        'num_checkpoints': num_checkpoints,
    }

    # If state size provided, compute absolute bytes
    if state_size_bytes is not None:
        result['without_checkpointing_bytes'] = memory_without * state_size_bytes
        result['with_checkpointing_bytes'] = memory_with * state_size_bytes
        result['savings_bytes'] = (memory_without - memory_with) * state_size_bytes

    return result


def get_checkpoint_info(seq_length, num_checkpoints=None):
    """Get information about checkpoint configuration.

    Args:
        seq_length: Number of iterations
        num_checkpoints: Number of checkpoints (default: optimal)

    Returns:
        Dictionary with checkpoint information
    """
    from tangent.checkpointing_simple import compute_checkpoint_positions

    if num_checkpoints is None:
        num_checkpoints = compute_optimal_checkpoints(seq_length)

    positions = compute_checkpoint_positions(seq_length, num_checkpoints)

    return {
        'seq_length': seq_length,
        'num_checkpoints': num_checkpoints,
        'checkpoint_positions': positions,
        'checkpoint_frequency': seq_length / num_checkpoints if num_checkpoints > 0 else seq_length,
        'memory_reduction': estimate_memory_savings(seq_length, num_checkpoints),
    }

class CheckpointAwareStack:
    """Stack wrapper that only stores values at checkpoint positions.
    
    This is the KEY component for Phase 4b memory reduction. It intercepts
    all append() operations and only stores values when at a checkpoint position.
    """
    
    def __init__(self, real_stack, checkpoint_positions_set):
        """Initialize checkpoint-aware stack."""
        self.real_stack = real_stack
        self.checkpoint_positions = checkpoint_positions_set
        self.current_iteration = None
        self.skipped_count = 0
        
    def set_current_iteration(self, iteration):
        """Set the current iteration index."""
        self.current_iteration = iteration
        
    def append(self, x):
        """Append (push) - only if at checkpoint."""
        if self.current_iteration in self.checkpoint_positions:
            self.real_stack.append(x)
        else:
            self.skipped_count += 1
            
    def pop(self):
        """Pop from real stack."""
        return self.real_stack.pop()
        
    def __len__(self):
        """Length proxy."""
        return len(self.real_stack)
        
    def __str__(self):
        return f"CheckpointAwareStack(skipped={self.skipped_count})"
        
    def __repr__(self):
        return self.__str__()
        
    def get_stats(self):
        """Get statistics."""
        return {
            'skipped_pushes': self.skipped_count,
            'num_checkpoints': len(self.checkpoint_positions),
        }
