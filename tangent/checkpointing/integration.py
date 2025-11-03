"""Main integration point for checkpointing in Tangent.

This module provides the high-level API for using checkpointing with Tangent's
automatic differentiation.
"""

import gast
import inspect
import textwrap
from typing import Optional, Dict, Any, Callable

from tangent.analysis.checkpoint_analyzer import CheckpointAnalyzer
from tangent.preprocessing.checkpoint_preprocessor import CheckpointPreprocessor
from tangent import grad as tangent_grad


def grad_with_checkpointing(func: Callable,
                           wrt=(0,),
                           checkpoint_config: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Callable:
    """
    Create gradient function with checkpointing enabled.

    This is the main entry point for checkpointed automatic differentiation.

    Args:
        func: Function to differentiate
        wrt: Which arguments to differentiate w.r.t.
        checkpoint_config: Checkpointing configuration:
            - strategy: 'auto', 'revolve', 'dynamic', or 'none'
            - memory_budget: Maximum memory for checkpoints (bytes)
            - min_loop_size: Minimum iterations to checkpoint (default: 100)
        **kwargs: Additional arguments passed to tangent.grad

    Returns:
        Gradient function with checkpointing

    Examples:
        >>> def expensive_loop(x):
        ...     state = x
        ...     for i in range(10000):
        ...         state = state + 0.01
        ...     return state
        >>>
        >>> # Auto checkpointing
        >>> df = grad_with_checkpointing(expensive_loop)
        >>>
        >>> # With configuration
        >>> df = grad_with_checkpointing(
        ...     expensive_loop,
        ...     checkpoint_config={'min_loop_size': 50}
        ... )
    """

    # Default configuration
    config = {
        'strategy': 'auto',
        'memory_budget': None,
        'min_loop_size': 100,
        'enabled': True
    }
    if checkpoint_config:
        config.update(checkpoint_config)

    try:
        # Get function source and parse to AST
        source = inspect.getsource(func)
        # Remove leading indentation
        source = textwrap.dedent(source)
        tree = gast.parse(source)

        # Stage 1: Analyze for checkpointing opportunities
        analyzer = CheckpointAnalyzer(
            memory_budget=config.get('memory_budget'),
            min_loop_size=config.get('min_loop_size', 100)
        )
        checkpoint_plan = analyzer.analyze(tree)

        # If no checkpointing needed, fall back to standard gradient
        if checkpoint_plan.recommended_strategy == 'none':
            print(f"[Checkpointing] No loops eligible for checkpointing in {func.__name__}")
            return tangent_grad(func, wrt=wrt, **kwargs)

        # Stage 2: Preprocess AST for checkpointing
        preprocessor = CheckpointPreprocessor(checkpoint_plan)
        preprocessed_tree = preprocessor.visit(tree)

        # For now, we'll use a hybrid approach:
        # 1. The preprocessor has added checkpoint infrastructure to the primal
        # 2. We'll pass checkpoint config to tangent.grad so it knows to use templates
        # 3. The checkpointed templates in grads.py will handle the rest

        # Update checkpoint config with our plan
        config['checkpoint_plan'] = checkpoint_plan
        config['preprocessed'] = True

        # Pass to standard tangent.grad with checkpoint flag
        # This will use the checkpointed templates defined in grads.py
        grad_func = tangent_grad(
            func,
            wrt=wrt,
            checkpoint=config,
            **kwargs
        )

        return grad_func

    except Exception as e:
        # If preprocessing fails, fall back to standard gradient
        print(f"[Checkpointing] Preprocessing failed: {e}")
        print(f"[Checkpointing] Falling back to standard gradient")
        return tangent_grad(func, wrt=wrt, **kwargs)


def enhanced_grad(func: Callable,
                 wrt=(0,),
                 checkpoint=None,
                 **kwargs) -> Callable:
    """Enhanced grad function with checkpointing support.

    This is a drop-in replacement for tangent.grad with checkpoint support.

    Args:
        func: Function to differentiate
        wrt: Which arguments to differentiate w.r.t.
        checkpoint: Checkpointing configuration:
            - None: No checkpointing
            - True: Auto checkpointing with defaults
            - Dict: Custom checkpoint configuration
        **kwargs: Additional arguments passed to tangent.grad

    Returns:
        Gradient function

    Examples:
        >>> # Auto checkpointing
        >>> df = enhanced_grad(func, checkpoint=True)
        >>>
        >>> # With configuration
        >>> df = enhanced_grad(func, checkpoint={
        ...     'strategy': 'revolve',
        ...     'min_loop_size': 50
        ... })
        >>>
        >>> # No checkpointing
        >>> df = enhanced_grad(func)
    """

    if checkpoint is not None:
        if isinstance(checkpoint, bool) and checkpoint:
            # Auto checkpointing with defaults
            checkpoint_config = {'strategy': 'auto'}
        elif isinstance(checkpoint, dict):
            checkpoint_config = checkpoint
        else:
            checkpoint_config = {'strategy': str(checkpoint)}

        return grad_with_checkpointing(
            func,
            wrt=wrt,
            checkpoint_config=checkpoint_config,
            **kwargs
        )
    else:
        # Standard gradient without checkpointing
        return tangent_grad(func, wrt=wrt, **kwargs)


def analyze_checkpointing(func: Callable,
                         min_loop_size: int = 100) -> Dict[str, Any]:
    """Analyze a function to determine checkpointing opportunities.

    This is a utility function to help understand what will be checkpointed.

    Args:
        func: Function to analyze
        min_loop_size: Minimum loop size to consider

    Returns:
        Dictionary with analysis results:
            - num_loops: Total number of loops
            - checkpointable_loops: Number of loops that can be checkpointed
            - estimated_memory_savings: Estimated memory reduction in bytes
            - recommended_strategy: Recommended checkpointing strategy
            - loop_details: Details about each loop

    Example:
        >>> def my_func(x):
        ...     state = x
        ...     for i in range(1000):
        ...         state = state + 0.1
        ...     return state
        >>>
        >>> info = analyze_checkpointing(my_func)
        >>> print(f"Can checkpoint {info['checkpointable_loops']} loops")
        >>> print(f"Strategy: {info['recommended_strategy']}")
    """

    try:
        # Get function source and parse
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = gast.parse(source)

        # Analyze
        analyzer = CheckpointAnalyzer(min_loop_size=min_loop_size)
        plan = analyzer.analyze(tree)

        # Build result
        checkpointable = [loop for loop in plan.loops.values() if loop.can_checkpoint]

        result = {
            'num_loops': len(plan.loops),
            'checkpointable_loops': len(checkpointable),
            'estimated_memory_savings': plan.memory_estimate,
            'recommended_strategy': plan.recommended_strategy,
            'loop_details': []
        }

        for loop_info in checkpointable:
            detail = {
                'loop_var': loop_info.loop_var,
                'num_iterations': loop_info.num_iterations,
                'modified_variables': list(loop_info.modified_variables),
                'num_checkpoints': len(loop_info.checkpoint_positions) if loop_info.checkpoint_positions else None,
                'reduction_ratio': (
                    1.0 - len(loop_info.checkpoint_positions) / loop_info.num_iterations
                    if loop_info.checkpoint_positions and loop_info.num_iterations
                    else None
                )
            }
            result['loop_details'].append(detail)

        return result

    except Exception as e:
        return {
            'error': str(e),
            'num_loops': 0,
            'checkpointable_loops': 0,
            'estimated_memory_savings': 0,
            'recommended_strategy': 'none'
        }


# Convenience function for quick testing
def compare_memory_usage(func: Callable,
                        *args,
                        wrt=(0,),
                        **kwargs) -> Dict[str, Any]:
    """Compare memory usage with and without checkpointing.

    This is a utility function for benchmarking checkpointing benefits.

    Args:
        func: Function to test
        *args: Arguments to pass to function
        wrt: Which arguments to differentiate w.r.t.
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with comparison results
    """
    import tracemalloc

    results = {}

    try:
        # Standard gradient
        print("Testing standard gradient...")
        df_standard = tangent_grad(func, wrt=wrt)

        tracemalloc.start()
        grad_standard = df_standard(*args, **kwargs)
        current, peak_standard = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results['standard_peak_mb'] = peak_standard / 1024 / 1024
        results['standard_success'] = True

    except Exception as e:
        results['standard_peak_mb'] = 0
        results['standard_success'] = False
        results['standard_error'] = str(e)

    try:
        # Checkpointed gradient
        print("Testing checkpointed gradient...")
        df_checkpoint = enhanced_grad(func, wrt=wrt, checkpoint=True)

        tracemalloc.start()
        grad_checkpoint = df_checkpoint(*args, **kwargs)
        current, peak_checkpoint = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results['checkpoint_peak_mb'] = peak_checkpoint / 1024 / 1024
        results['checkpoint_success'] = True

        # Check gradient correctness
        if results['standard_success']:
            import numpy as np
            if np.allclose(grad_standard, grad_checkpoint, rtol=1e-5):
                results['gradients_match'] = True
            else:
                results['gradients_match'] = False

    except Exception as e:
        results['checkpoint_peak_mb'] = 0
        results['checkpoint_success'] = False
        results['checkpoint_error'] = str(e)

    # Calculate reduction
    if results.get('standard_success') and results.get('checkpoint_success'):
        standard = results['standard_peak_mb']
        checkpoint = results['checkpoint_peak_mb']
        if standard > 0:
            reduction = (1 - checkpoint / standard) * 100
            results['memory_reduction_percent'] = reduction
        else:
            results['memory_reduction_percent'] = 0

    return results
