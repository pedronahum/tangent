"""Checkpointing module for Tangent.

This module provides memory-efficient automatic differentiation through checkpointing.
"""

from tangent.checkpointing.integration import (
    grad_with_checkpointing,
    enhanced_grad
)
from tangent.checkpointing.runtime import (
    find_nearest_checkpoint,
    compute_checkpoint_positions,
    checkpoint_state,
    restore_state
)

__all__ = [
    'grad_with_checkpointing',
    'enhanced_grad',
    'find_nearest_checkpoint',
    'compute_checkpoint_positions',
    'checkpoint_state',
    'restore_state'
]
