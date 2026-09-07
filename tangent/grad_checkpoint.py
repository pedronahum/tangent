"""Checkpointing-related wrappers around tangent.grad().

What actually works today:

1. ``tangent.grad(func, checkpoint=True)`` — automatic checkpointing of
   ``for i in range(n)`` loops (constant ``n``, zero-based, at least
   ``min_length`` iterations, default 100). Only the loop *target* variable
   is stored selectively (at ~sqrt(n) checkpoint positions); every other
   intermediate value in the loop body is still pushed to the tape each
   iteration. Measured overall memory reduction on the reference benchmark
   is ~3% (97% reduction of target storage alone). Composes with
   ``optimized=True``. See docs/checkpointing_user_guide.md.

2. ``tangent.checkpointed_loop`` (tangent/checkpointing_simple.py) — a
   manual forward-pass helper that stores only O(sqrt(n)) state snapshots.
   ``tangent.grad`` cannot differentiate through it; it is a memory-saving
   utility for the forward pass only.

``grad_with_checkpointing`` in this module was intended to rewrite arbitrary
loops via AST transformation. That transformation was never implemented, so
this function raises ``NotImplementedError`` for any function containing a
loop and simply delegates to ``tangent.grad`` otherwise.
"""

import ast
import inspect
from typing import Callable, Optional, Tuple

# Import Tangent's grad function
from tangent.grad_util import grad as tangent_grad

# Import checkpointing utilities
from tangent.checkpointing_simple import get_memory_savings


def grad_with_checkpointing(
    func: Callable,
    wrt: Tuple[int, ...] = (0,),
    num_checkpoints: Optional[int] = None,
    **grad_kwargs,
) -> Callable:
    """
    Intended: gradient of `func` with checkpointing applied to its loops.

    NOT IMPLEMENTED for functions that contain loops: the AST transformation
    that would rewrite arbitrary loops into checkpointed form was never built,
    and this function raises ``NotImplementedError`` in that case (at call
    time, so the failure is immediate rather than deferred to the first
    gradient evaluation).

    For a function without loops this simply delegates to ``tangent.grad``
    (checkpointing would be a no-op anyway).

    Working alternatives:

    * ``tangent.grad(func, checkpoint=True)`` — automatic, but limited to
      ``for i in range(n)`` loops with a constant, zero-based range of at
      least ``min_length`` (default 100) iterations, and only reduces
      storage of the loop target variable (~3% overall in the reference
      benchmark).
    * ``tangent.checkpointed_loop`` — manual O(sqrt(n))-memory forward pass;
      gradients do not flow through it.

    Args:
        func: Function to differentiate
        wrt: Tuple of argument indices to differentiate with respect to
        num_checkpoints: Unused (kept for API compatibility)
        **grad_kwargs: Additional arguments passed to tangent.grad()

    Returns:
        Gradient function (only when `func` contains no loops)

    Raises:
        NotImplementedError: if `func` contains a for/while loop.
    """
    # Analyze function to detect loops
    loop_info = _detect_loops(func)

    if not loop_info:
        # No loops found - use standard gradient
        return tangent_grad(func, wrt=wrt, **grad_kwargs)

    raise NotImplementedError(
        "grad_with_checkpointing: automatic checkpointing via AST "
        "transformation is not implemented, and the function "
        f"'{func.__name__}' contains {len(loop_info)} loop(s):\n"
        + "\n".join(f"  - Line {info['line']}: {info['type']}" for info in loop_info)
        + "\n"
        "\n"
        "Working alternatives:\n"
        "\n"
        "1. tangent.grad(func, checkpoint=True)\n"
        "   Automatic, but limited: applies only to 'for i in range(n)' "
        "loops\n"
        "   with a constant, zero-based range of >= 100 iterations "
        "(configurable\n"
        "   via checkpoint_config={'min_length': ...}), and only the loop "
        "target\n"
        "   variable is stored selectively - other intermediates are still "
        "taped\n"
        "   every iteration (~3% overall memory reduction measured).\n"
        "\n"
        "2. tangent.checkpointed_loop(step_fn, initial_state, seq_length, "
        "num_checkpoints)\n"
        "   Manual forward-pass helper storing O(sqrt(n)) snapshots. Note "
        "that\n"
        "   tangent.grad cannot differentiate through it.\n"
        "\n"
        "See docs/checkpointing_user_guide.md for details."
    )


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
    except SyntaxError:
        # Indented source (e.g., a method); tangent.grad will produce its own
        # clearer error if the function is genuinely unparseable.
        return []

    class LoopFinder(ast.NodeVisitor):
        def __init__(self):
            self.loops = []

        def visit_For(self, node):
            loop_info = {
                'type': 'for',
                'line': node.lineno,
                'target': ast.unparse(node.target) if hasattr(ast, 'unparse') else '<target>',
                'iter': ast.unparse(node.iter) if hasattr(ast, 'unparse') else '<iter>',
            }
            self.loops.append(loop_info)
            self.generic_visit(node)

        def visit_While(self, node):
            loop_info = {
                'type': 'while',
                'line': node.lineno,
                'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else '<condition>',
            }
            self.loops.append(loop_info)
            self.generic_visit(node)

    finder = LoopFinder()
    finder.visit(tree)
    return finder.loops


def estimate_checkpoint_savings(seq_length: int, num_checkpoints: Optional[int] = None) -> dict:
    """
    Estimate memory savings from checkpointing.

    This is a convenience wrapper around get_memory_savings(). Note that the
    estimate counts stored *states*: it applies to the manual
    ``checkpointed_loop`` helper, not to the overall tape memory of
    ``tangent.grad(func, checkpoint=True)`` (which only stores the loop
    target selectively).

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


# Convenience function for checking if checkpointing would be beneficial
def should_checkpoint(seq_length: int, threshold: float = 0.5) -> bool:
    """
    Determine if checkpointing would provide significant memory savings.

    The estimate counts stored states (see ``estimate_checkpoint_savings``).

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
    'estimate_checkpoint_savings',
    'should_checkpoint',
]
