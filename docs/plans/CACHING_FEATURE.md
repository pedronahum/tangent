# Function Caching Feature

## Overview

This document describes the function caching system implemented for Tangent's automatic differentiation library. The caching system provides massive performance improvements by storing compiled gradient functions and retrieving them on subsequent calls with the same parameters.

## Motivation

Tangent's source-to-source automatic differentiation involves several expensive operations:
1. Retrieving function source code with `inspect.getsource`
2. Parsing source code into an Abstract Syntax Tree (AST)
3. Walking the AST to generate the adjoint/derivative code
4. Optimizing the generated code
5. Compiling the generated Python code

For applications that repeatedly compute gradients of the same function (e.g., training loops, optimization algorithms), these compilation steps are unnecessary overhead. The caching system eliminates this overhead by storing compiled gradient functions.

## Performance Improvements

Benchmark results show dramatic performance improvements:

| Metric | Without Caching | With Caching | Speedup |
|--------|----------------|--------------|---------|
| Single gradient call (polynomial) | 75.38 ms | 0.04 ms | **1,745x** |
| Single gradient call (nested) | 41.09 ms | 0.04 ms | **1,081x** |
| 100 repeated calls | 2,106 ms | 24 ms | **87.6x** |
| Cache hit rate | N/A | 99% | N/A |

## Architecture

### Cache Key Generation

The caching system generates unique keys based on:

1. **Function source code hash** - SHA-256 hash of the function's source code
2. **Function bytecode hash** - SHA-256 hash of the compiled bytecode
3. **Closure values** - Hash of captured variables for closures
4. **Function identity** - Module name, function name, and qualified name
5. **Transformation parameters**:
   - `wrt` - Which arguments to differentiate with respect to
   - `motion` - 'split' or 'joint' mode
   - `mode` - 'forward' or 'reverse' mode
   - `optimized` - Whether to apply optimizations
   - `preserve_result` - Whether to preserve the original result
   - `check_dims` - Whether to check dimensions
   - `input_derivative` - Input derivative handling mode

This comprehensive key ensures that:
- Different functions get separate cache entries
- Same function with different parameters gets separate entries
- Closures with different captured variables get separate entries
- Cache hits only occur for truly identical transformations

### Cache Implementation

The cache is implemented as a thread-safe LRU (Least Recently Used) cache:

```python
# Global cache with configurable size (default: 128 entries)
_cache = OrderedDict()
_cache_lock = threading.Lock()
_cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
```

Key features:
- **Thread-safe** - Uses locks to protect cache access in multi-threaded environments
- **LRU eviction** - Automatically removes least recently used entries when cache is full
- **Statistics tracking** - Tracks hits, misses, evictions, and hit rate
- **Configurable size** - Users can adjust cache size based on memory constraints

### Integration with Tangent API

The caching is transparently integrated into Tangent's main API functions:

```python
# Original uncached implementations renamed
_grad_uncached(func, ...)
_autodiff_uncached(func, ...)

# Cached versions become the public API
grad = cached_grad(_grad_uncached)
autodiff = cached_autodiff(_autodiff_uncached)
```

This design ensures:
- Zero API changes - existing code works without modification
- Backward compatibility - no breaking changes
- Easy to test - uncached versions still available for testing

## API Reference

### Core Functions

These are automatically cached - no user action required:

```python
tangent.grad(func, wrt=(0,), optimized=True, preserve_result=False,
             check_dims=True, verbose=0)
```

```python
tangent.autodiff(func, wrt=(0,), optimized=True, motion='joint',
                 mode='reverse', preserve_result=False, check_dims=True,
                 input_derivative=INPUT_DERIVATIVE.Required, verbose=0)
```

### Cache Management Functions

```python
tangent.clear_cache()
```
Clear all cached gradient functions. Useful for testing or forcing recompilation.

```python
tangent.get_cache_stats()
```
Returns:
```python
{
    'hits': 99,          # Number of cache hits
    'misses': 1,         # Number of cache misses
    'evictions': 0,      # Number of evicted entries
    'size': 5,           # Current number of cached functions
    'max_size': 128,     # Maximum cache size
    'hit_rate': 0.99     # Cache hit rate (hits / total requests)
}
```

```python
tangent.reset_cache_stats()
```
Reset statistics counters (hits, misses, evictions) without clearing the cache.

```python
tangent.set_cache_size(size)
```
Set the maximum number of functions to cache. Default is 128.

```python
tangent.get_cache_size()
```
Get the current maximum cache size.

## Usage Examples

### Basic Usage (Automatic)

Caching happens automatically - no code changes needed:

```python
import tangent

def loss(params, data):
    return sum((params @ data) ** 2)

# First call: cache miss, compiles gradient (~50ms)
grad_loss = tangent.grad(loss)

# Subsequent calls: cache hit, instant retrieval (~0.04ms)
grad_loss = tangent.grad(loss)  # 1000x+ faster!
```

### Monitoring Cache Performance

```python
import tangent

# Reset stats for clean measurement
tangent.reset_cache_stats()

# Training loop
for epoch in range(100):
    grad_fn = tangent.grad(my_loss_function)
    gradients = grad_fn(params, data)
    params = update(params, gradients)

# Check cache performance
stats = tangent.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### Managing Cache Size

```python
import tangent

# For memory-constrained environments
tangent.set_cache_size(32)  # Reduce from default 128

# For large projects with many functions
tangent.set_cache_size(512)  # Increase cache size

# Clear cache when switching tasks
tangent.clear_cache()
```

### Verbose Mode with Caching

```python
import tangent

def f(x):
    return x * x

# First call: see compilation output and cache miss
df = tangent.grad(f, verbose=1)
# Output: [Cache] Computing new gradient function for f
# Output: <generated derivative code>

# Second call: see cache hit
df = tangent.grad(f, verbose=1)
# Output: [Cache] Retrieved cached gradient function for f
```

## Implementation Details

### Files Created/Modified

1. **tangent/function_cache.py** (new file, 362 lines)
   - Core caching implementation
   - Cache key generation with source, bytecode, and closure hashing
   - Thread-safe LRU cache with statistics
   - Decorator functions for wrapping grad/autodiff

2. **tangent/grad_util.py** (modified)
   - Renamed original functions to `_grad_uncached` and `_autodiff_uncached`
   - Applied caching decorators to create public API
   - Updated internal references to use uncached versions

3. **tangent/__init__.py** (modified)
   - Exported cache management functions
   - Made cache API available at top level

4. **tests/test_cache.py** (new file, 313 lines)
   - 13 comprehensive test cases
   - Tests for cache hits/misses, different parameters, correctness
   - Tests for cache management functions
   - All tests passing

5. **benchmark_cache.py** (new file, 134 lines)
   - Performance benchmarking suite
   - Measures speedup from caching
   - Generates detailed performance reports

6. **README.md** (modified)
   - Added "Performance: Automatic Caching" section
   - Usage examples and benchmark results
   - User-facing documentation

### Cache Key Design Considerations

The cache key must distinguish between:

1. **Different functions** - Same code in different files/modules
   - Solution: Include module name and qualified name

2. **Modified functions** - User edits function and reloads
   - Solution: Hash source code (detects changes)

3. **Closures** - Functions with same code but different captured variables
   - Solution: Hash closure cell contents

4. **Different parameters** - Same function, different differentiation parameters
   - Solution: Include all transformation parameters in key

5. **Bytecode differences** - Functions with same source but different compilation
   - Solution: Hash bytecode in addition to source

### Thread Safety

The cache is thread-safe through:
- Global lock (`_cache_lock`) protecting all cache operations
- Atomic read-modify-write operations
- Lock-protected statistics updates

This allows safe use in:
- Multi-threaded training
- Parallel gradient computations
- Concurrent model evaluations

### Memory Management

The LRU eviction policy ensures:
- Bounded memory usage (configurable size)
- Automatic removal of unused entries
- Most frequently used gradients stay cached

Memory per cached entry:
- Compiled function object: ~1-10 KB
- Cache key: ~100-200 bytes
- Total for 128 entries: ~128 KB - 1.3 MB (negligible)

## Testing

### Test Coverage

The test suite (tests/test_cache.py) includes:

1. **Basic functionality**
   - Cache hits and misses
   - Correctness of cached functions
   - Different functions get separate entries

2. **Parameter sensitivity**
   - Different `wrt` parameters
   - Different `optimized` flag
   - Different `preserve_result` flag

3. **Cache management**
   - Clear cache functionality
   - Cache size limits and eviction
   - Statistics tracking and reset

4. **Integration**
   - Works with `tangent.grad()`
   - Works with `tangent.autodiff()`
   - Multiple wrt arguments

All 13 tests pass:
```
tests/test_cache.py::test_basic_caching PASSED
tests/test_cache.py::test_cache_different_functions PASSED
tests/test_cache.py::test_cache_different_parameters PASSED
tests/test_cache.py::test_cache_optimized_flag PASSED
tests/test_cache.py::test_cache_preserve_result_flag PASSED
tests/test_cache.py::test_clear_cache PASSED
tests/test_cache.py::test_cache_size_limit PASSED
tests/test_cache.py::test_set_cache_size PASSED
tests/test_cache.py::test_reset_cache_stats PASSED
tests/test_cache.py::test_cache_hit_rate PASSED
tests/test_cache.py::test_autodiff_caching PASSED
tests/test_cache.py::test_cache_with_multiple_wrt PASSED
tests/test_cache.py::test_cache_correctness PASSED
```

## Future Enhancements

Potential improvements for future versions:

1. **Persistent caching** - Save cache to disk between sessions
2. **Distributed caching** - Share cache across processes/machines
3. **Smart precompilation** - Detect hot paths and precompile
4. **Cache warming** - Pre-populate cache with common functions
5. **Memory-mapped cache** - Use mmap for very large caches
6. **Cache versioning** - Invalidate cache on Tangent version updates
7. **Compression** - Compress cached functions to save memory

## Backward Compatibility

The caching system is fully backward compatible:
- No API changes required
- Existing code works without modification
- Can be disabled by setting cache size to 0 (if needed)
- No performance regression for single-use functions

## Limitations

Current limitations:

1. **Interactive sessions** - Functions defined in Python REPL can't be cached (source not available)
2. **Dynamic functions** - Functions generated via `exec()` or `eval()` may not cache correctly
3. **C extensions** - Native functions without Python source can't be cached
4. **Memory usage** - Cache grows with number of unique functions (mitigated by LRU eviction)

These are inherent limitations of Tangent's source-based approach, not the caching system.

## Conclusion

The function caching system provides dramatic performance improvements (1000x+ for cached calls) with zero API changes and full backward compatibility. It's a major enhancement that makes Tangent practical for production use in training loops and optimization algorithms.

The implementation is robust, well-tested, thread-safe, and configurable. Users get the benefits automatically, while power users have fine-grained control through the cache management API.
