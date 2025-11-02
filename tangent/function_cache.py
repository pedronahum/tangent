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
"""Function caching system for Tangent autodiff transformations.

This module provides an LRU cache for storing transformed gradient functions,
significantly improving performance when repeatedly computing gradients of the
same function with the same parameters.

Example usage:
    import tangent
    from tangent.function_cache import clear_cache, get_cache_stats

    # Functions are automatically cached
    df = tangent.grad(my_function)

    # Clear cache if needed
    clear_cache()

    # Get cache statistics
    stats = get_cache_stats()
    print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
"""
from __future__ import absolute_import

import functools
import hashlib
import inspect
import threading
from collections import OrderedDict


# Global cache configuration
_CACHE_SIZE = 128  # Maximum number of cached functions
_cache_lock = threading.Lock()
_cache = OrderedDict()
_cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}


def _generate_cache_key(func, wrt, motion, mode, optimized, preserve_result,
                        check_dims, input_derivative):
    """Generate a unique cache key for a function transformation.

    The cache key is based on:
    - Function source code (hashed for efficiency)
    - Function bytecode (for functions with same source but different closures)
    - Function name and module
    - All transformation parameters

    Args:
        func: The function to differentiate
        wrt: Tuple of argument indices to differentiate with respect to
        motion: 'split' or 'joint'
        mode: 'forward' or 'reverse'
        optimized: Whether to optimize the gradient function
        preserve_result: Whether to preserve the original function result
        check_dims: Whether to check dimensions
        input_derivative: Input derivative mode

    Returns:
        A hashable tuple representing the cache key
    """
    try:
        # Get function source code and hash it
        source = inspect.getsource(func)
        source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]
    except (OSError, TypeError):
        # If we can't get source (e.g., built-in function), use empty hash
        source_hash = ''

    # Get bytecode hash to distinguish functions with same source but different behavior
    # (e.g., closures with different captured variables)
    try:
        bytecode = func.__code__.co_code
        bytecode_hash = hashlib.sha256(bytecode).hexdigest()[:16]
    except (AttributeError, TypeError):
        bytecode_hash = str(id(func))

    # Get closure values to distinguish functions with same bytecode but different closures
    closure_values = []
    if hasattr(func, '__closure__') and func.__closure__:
        try:
            for cell in func.__closure__:
                try:
                    val = cell.cell_contents
                    # Try to hash the value
                    closure_values.append(hash(val))
                except (ValueError, AttributeError):
                    # If unhashable, use id
                    closure_values.append(id(val))
        except (AttributeError, ValueError):
            pass
    closure_hash = str(tuple(closure_values)) if closure_values else ''

    # Get function identity
    func_module = getattr(func, '__module__', '')
    func_name = getattr(func, '__name__', '')
    func_id = getattr(func, '__qualname__', func_name)

    # Convert input_derivative enum to string for hashability
    if hasattr(input_derivative, 'name'):
        input_derivative_str = input_derivative.name
    else:
        input_derivative_str = str(input_derivative)

    # Create cache key tuple
    cache_key = (
        source_hash,
        bytecode_hash,
        closure_hash,
        func_module,
        func_id,
        wrt,
        motion,
        mode,
        optimized,
        preserve_result,
        check_dims,
        input_derivative_str
    )

    return cache_key


def _get_from_cache(cache_key):
    """Retrieve a cached function if available.

    Args:
        cache_key: The cache key to look up

    Returns:
        The cached function, or None if not found
    """
    with _cache_lock:
        if cache_key in _cache:
            # Move to end (most recently used)
            _cache.move_to_end(cache_key)
            _cache_stats['hits'] += 1
            return _cache[cache_key]
        else:
            _cache_stats['misses'] += 1
            return None


def _add_to_cache(cache_key, func):
    """Add a transformed function to the cache.

    Implements LRU eviction when cache is full.

    Args:
        cache_key: The cache key
        func: The transformed function to cache
    """
    with _cache_lock:
        # Check if we need to evict
        if len(_cache) >= _CACHE_SIZE and cache_key not in _cache:
            # Remove oldest item (FIFO in OrderedDict)
            _cache.popitem(last=False)
            _cache_stats['evictions'] += 1

        # Add to cache
        _cache[cache_key] = func
        # Move to end (most recently used)
        _cache.move_to_end(cache_key)


def clear_cache():
    """Clear all cached functions.

    This can be useful for testing or when you want to force recompilation
    of gradient functions.
    """
    with _cache_lock:
        _cache.clear()


def get_cache_stats():
    """Get cache statistics.

    Returns:
        A dictionary with cache statistics:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - evictions: Number of cache evictions
        - size: Current cache size
        - max_size: Maximum cache size
        - hit_rate: Cache hit rate (hits / (hits + misses))
    """
    with _cache_lock:
        stats = _cache_stats.copy()
        stats['size'] = len(_cache)
        stats['max_size'] = _CACHE_SIZE

        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests
        else:
            stats['hit_rate'] = 0.0

        return stats


def reset_cache_stats():
    """Reset cache statistics counters.

    This does not clear the cache itself, only the statistics.
    """
    with _cache_lock:
        _cache_stats['hits'] = 0
        _cache_stats['misses'] = 0
        _cache_stats['evictions'] = 0


def set_cache_size(size):
    """Set the maximum cache size.

    Args:
        size: Maximum number of functions to cache (must be > 0)

    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError("Cache size must be positive")

    global _CACHE_SIZE
    with _cache_lock:
        _CACHE_SIZE = size

        # Evict items if current size exceeds new limit
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
            _cache_stats['evictions'] += 1


def get_cache_size():
    """Get the maximum cache size.

    Returns:
        The maximum number of functions that can be cached
    """
    return _CACHE_SIZE


def cached_autodiff(original_autodiff):
    """Decorator to add caching to the autodiff function.

    This wraps the original autodiff function with caching logic.

    Args:
        original_autodiff: The original autodiff function to wrap

    Returns:
        A wrapped version with caching support
    """
    @functools.wraps(original_autodiff)
    def wrapper(func, wrt=(0,), optimized=True, motion='joint', mode='reverse',
                preserve_result=False, check_dims=True,
                input_derivative=None, verbose=0):

        # Import here to avoid circular imports
        from tangent.grad_util import INPUT_DERIVATIVE

        # Handle default value for input_derivative
        if input_derivative is None:
            input_derivative = INPUT_DERIVATIVE.Required

        # Generate cache key
        cache_key = _generate_cache_key(
            func, wrt, motion, mode, optimized, preserve_result,
            check_dims, input_derivative
        )

        # Try to get from cache
        cached_func = _get_from_cache(cache_key)
        if cached_func is not None:
            if verbose >= 1:
                print(f"[Cache] Retrieved cached gradient function for {func.__name__}")
            return cached_func

        # Cache miss - compute the gradient
        if verbose >= 1:
            print(f"[Cache] Computing new gradient function for {func.__name__}")

        result = original_autodiff(
            func, wrt=wrt, optimized=optimized, motion=motion, mode=mode,
            preserve_result=preserve_result, check_dims=check_dims,
            input_derivative=input_derivative, verbose=verbose
        )

        # Add to cache
        _add_to_cache(cache_key, result)

        return result

    return wrapper


def cached_grad(original_grad):
    """Decorator to add caching to the grad function.

    This wraps the original grad function with caching logic.

    Args:
        original_grad: The original grad function to wrap

    Returns:
        A wrapped version with caching support
    """
    @functools.wraps(original_grad)
    def wrapper(func, wrt=(0,), optimized=True, preserve_result=False,
                check_dims=True, verbose=0):

        # Import here to avoid circular imports
        from tangent.grad_util import INPUT_DERIVATIVE

        # Generate cache key (grad uses specific default parameters)
        cache_key = _generate_cache_key(
            func, wrt, 'joint', 'reverse', optimized, preserve_result,
            check_dims, INPUT_DERIVATIVE.DefaultOne
        )

        # Try to get from cache
        cached_func = _get_from_cache(cache_key)
        if cached_func is not None:
            if verbose >= 1:
                print(f"[Cache] Retrieved cached gradient function for {func.__name__}")
            return cached_func

        # Cache miss - compute the gradient
        if verbose >= 1:
            print(f"[Cache] Computing new gradient function for {func.__name__}")

        result = original_grad(
            func, wrt=wrt, optimized=optimized, preserve_result=preserve_result,
            check_dims=check_dims, verbose=verbose
        )

        # Add to cache
        _add_to_cache(cache_key, result)

        return result

    return wrapper
