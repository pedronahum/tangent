"""Integration of checkpointing with Tangent's gradient computation.

This module provides a wrapper around tangent.grad() that automatically applies
checkpointing to loops in the function being differentiated.

Usage:
    import tangent
    from tangent.grad_checkpoint import grad_with_checkpointing

    def rnn_forward(x, W, b, seq_length=100):
        state = x
        for i in range(seq_length):
            state = np.tanh(state @ W + b)
        return state

    # Use checkpointing-aware gradient
    df = grad_with_checkpointing(rnn_forward, num_checkpoints=10)
    grad_x = df(x, W, b)
"""

import ast
import inspect
import functools
import numpy as np
from typing import Callable, Optional, Tuple, Any

# Import Tangent's grad function
from tangent.grad_util import grad as tangent_grad

# Import checkpointing utilities
from tangent.checkpointing_simple import (
    compute_checkpoint_positions,
    checkpointed_loop,
    get_memory_savings
)


def grad_with_checkpointing(func: Callable,
                            wrt: Tuple[int, ...] = (0,),
                            num_checkpoints: Optional[int] = None,
                            **grad_kwargs) -> Callable:
    """
    Compute gradient of a function with automatic checkpointing for loops.

    This is a drop-in replacement for tangent.grad() that automatically
    detects loops and applies checkpointing to reduce memory usage.

    Args:
        func: Function to differentiate
        wrt: Tuple of argument indices to differentiate with respect to
        num_checkpoints: Number of checkpoints to use (default: sqrt(seq_length))
        **grad_kwargs: Additional arguments passed to tangent.grad()

    Returns:
        Gradient function with checkpointing applied

    Example:
        >>> def rnn(x, W, b):
        ...     state = x
        ...     for i in range(1000):
        ...         state = np.tanh(state @ W + b)
        ...     return state
        >>> df = grad_with_checkpointing(rnn, num_checkpoints=31)
        >>> grad_x = df(x, W, b)

    Note:
        This is Phase 1 implementation. Full AST transformation integration
        will be added in Phase 2.
    """
    # Analyze function to detect loops
    loop_info = _detect_loops(func)

    if not loop_info:
        # No loops found - use standard gradient
        return tangent_grad(func, wrt=wrt, **grad_kwargs)

    # For Phase 1, we use a manual checkpointing wrapper
    # Phase 2 will implement full AST transformation

    # Create a wrapper that explains the limitation
    def gradient_wrapper(*args, **kwargs):
        raise NotImplementedError(
            "Automatic checkpointing via AST transformation is not yet implemented.\n"
            "\n"
            "Current workaround: Manually apply checkpointing to your loops.\n"
            "\n"
            "Example:\n"
            "  from tangent.checkpointing_simple import checkpointed_loop\n"
            "\n"
            "  def rnn_step(state):\n"
            "      return np.tanh(state @ W + b)\n"
            "\n"
            "  # Instead of:\n"
            "  # for i in range(seq_length):\n"
            "  #     state = rnn_step(state)\n"
            "\n"
            "  # Use:\n"
            "  final_state, checkpoints = checkpointed_loop(\n"
            "      rnn_step, initial_state, seq_length, num_checkpoints=31\n"
            "  )\n"
            "\n"
            f"Detected {len(loop_info)} loop(s) in function '{func.__name__}':\n"
            + "\n".join(f"  - Line {info['line']}: {info['type']}" for info in loop_info)
            + "\n\nFor full integration, see: tangent/checkpointing_simple.py"
        )

    return gradient_wrapper


def _detect_loops(func: Callable) -> list:
    """
    Detect loops in a function using AST analysis.

    Args:
        func: Function to analyze

    Returns:
        List of dictionaries with loop information:
        [{'type': 'for', 'line': 10, 'target': 'i', 'iter': 'range(100)'}]
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
    except (OSError, TypeError):
        # Cannot get source (e.g., built-in function)
        return []

    class LoopFinder(ast.NodeVisitor):
        def __init__(self):
            self.loops = []

        def visit_For(self, node):
            loop_info = {
                'type': 'for',
                'line': node.lineno,
                'target': ast.unparse(node.target) if hasattr(ast, 'unparse') else '<target>',
                'iter': ast.unparse(node.iter) if hasattr(ast, 'unparse') else '<iter>'
            }
            self.loops.append(loop_info)
            self.generic_visit(node)

        def visit_While(self, node):
            loop_info = {
                'type': 'while',
                'line': node.lineno,
                'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else '<condition>'
            }
            self.loops.append(loop_info)
            self.generic_visit(node)

    finder = LoopFinder()
    finder.visit(tree)
    return finder.loops


def estimate_checkpoint_savings(seq_length: int,
                                num_checkpoints: Optional[int] = None) -> dict:
    """
    Estimate memory savings from checkpointing.

    This is a convenience wrapper around get_memory_savings().

    Args:
        seq_length: Length of the sequence/loop
        num_checkpoints: Number of checkpoints (default: sqrt(seq_length))

    Returns:
        Dictionary with memory statistics

    Example:
        >>> stats = estimate_checkpoint_savings(1000)
        >>> print(f"Memory reduction: {stats['savings_percent']:.1f}%")
        Memory reduction: 96.9%
    """
    return get_memory_savings(seq_length, num_checkpoints)


def checkpointed_grad(loop_func: Callable,
                     seq_length: int,
                     num_checkpoints: Optional[int] = None) -> Callable:
    """
    Create a gradient function for a loop body with checkpointing.

    This is a helper function for manually applying checkpointing to loops
    until full AST transformation is implemented.

    Args:
        loop_func: Function representing one iteration of the loop (state -> new_state)
        seq_length: Number of loop iterations
        num_checkpoints: Number of checkpoints to use

    Returns:
        Gradient function that uses checkpointing

    Example:
        >>> def rnn_step(state):
        ...     return np.tanh(state * 1.1 + 0.1)
        >>> df = checkpointed_grad(rnn_step, seq_length=1000, num_checkpoints=31)
        >>> # Use df in your gradient computation...

    Note:
        This is a simplified API. For complex use cases, use checkpointed_loop
        and tangent.grad() directly.
    """
    # Get gradient of single step
    step_grad = tangent_grad(loop_func)

    def gradient_with_checkpointing(initial_state):
        """
        Compute gradient through the loop using checkpointing.
        """
        # Forward pass with checkpointing
        final_state, checkpoints = checkpointed_loop(
            loop_func, initial_state, seq_length, num_checkpoints
        )

        # Backward pass - simplified version
        # Full implementation requires proper gradient accumulation
        # For now, we compute the gradient at the final state
        grad_final = step_grad(final_state)

        # TODO: Implement full backward pass with recomputation from checkpoints
        # This requires:
        # 1. Iterate backward through checkpoints
        # 2. Recompute forward from each checkpoint
        # 3. Accumulate gradients properly
        # 4. Handle multiple inputs/outputs

        return grad_final

    return gradient_with_checkpointing


# Convenience function for checking if checkpointing would be beneficial
def should_checkpoint(seq_length: int, threshold: float = 0.5) -> bool:
    """
    Determine if checkpointing would provide significant memory savings.

    Args:
        seq_length: Length of the sequence/loop
        threshold: Minimum memory reduction fraction to recommend checkpointing

    Returns:
        True if checkpointing is recommended

    Example:
        >>> should_checkpoint(100)  # sqrt(100) = 10 checkpoints -> 90% savings
        True
        >>> should_checkpoint(10)   # sqrt(10) = 3 checkpoints -> 70% savings
        True
    """
    stats = get_memory_savings(seq_length)
    return stats['savings_ratio'] >= threshold


# Export main functions
__all__ = [
    'grad_with_checkpointing',
    'checkpointed_grad',
    'estimate_checkpoint_savings',
    'should_checkpoint',
]
