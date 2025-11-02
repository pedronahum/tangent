"""Benchmark script for function caching performance.

This script measures the performance improvement from caching gradient functions.
"""
import time
import tangent


def benchmark_function_caching():
    """Benchmark the performance improvement from function caching."""

    def polynomial(x):
        """A moderately complex polynomial function."""
        return 3.0 * x**4 + 2.0 * x**3 - 5.0 * x**2 + 7.0 * x - 1.0

    def matrix_function(x):
        """Matrix operations."""
        import numpy as np
        y = x @ x.T
        return np.sum(y * y)

    def nested_function(x):
        """Function with nested operations."""
        y = x * x
        z = y + 2.0 * x
        return z * z + y

    functions = [
        ("polynomial", polynomial),
        ("nested_function", nested_function),
    ]

    print("=" * 80)
    print("Function Caching Benchmark")
    print("=" * 80)

    for func_name, func in functions:
        print(f"\n{func_name}:")
        print("-" * 80)

        # Clear cache and stats
        tangent.clear_cache()
        tangent.reset_cache_stats()

        # Measure first compilation (cache miss)
        start = time.time()
        df1 = tangent.grad(func)
        first_time = time.time() - start

        stats_after_first = tangent.get_cache_stats()

        # Measure subsequent compilations (cache hits)
        times = []
        for i in range(10):
            start = time.time()
            df = tangent.grad(func)
            times.append(time.time() - start)

        avg_cached_time = sum(times) / len(times)

        stats_after_cached = tangent.get_cache_stats()

        # Calculate speedup
        speedup = first_time / avg_cached_time if avg_cached_time > 0 else float('inf')

        # Print results
        print(f"First compilation (cache miss):  {first_time*1000:.2f} ms")
        print(f"Avg cached compilation:          {avg_cached_time*1000:.2f} ms")
        print(f"Speedup:                         {speedup:.1f}x")
        print(f"\nCache stats after first call:    {stats_after_first}")
        print(f"Cache stats after cached calls:  {stats_after_cached}")

        # Verify correctness
        result1 = df1(2.0)
        result2 = df(2.0)
        assert abs(result1 - result2) < 1e-10, "Results don't match!"
        print(f"✓ Results verified: df(2.0) = {result1}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    final_stats = tangent.get_cache_stats()
    print(f"Total cache hits:     {final_stats['hits']}")
    print(f"Total cache misses:   {final_stats['misses']}")
    print(f"Hit rate:             {final_stats['hit_rate']:.1%}")
    print(f"Cache size:           {final_stats['size']}/{final_stats['max_size']}")
    print(f"Evictions:            {final_stats['evictions']}")
    print("=" * 80)


def benchmark_repeated_calls():
    """Benchmark repeated gradient calls with and without caching."""

    def test_function(x):
        return x * x * x

    print("\n" + "=" * 80)
    print("Repeated Calls Benchmark")
    print("=" * 80)

    num_calls = 100

    # Test with caching enabled (default)
    tangent.clear_cache()
    tangent.reset_cache_stats()

    start = time.time()
    for i in range(num_calls):
        df = tangent.grad(test_function)
        _ = df(2.0)
    cached_time = time.time() - start

    stats = tangent.get_cache_stats()

    print(f"\nWith caching (100 calls):")
    print(f"  Total time:           {cached_time*1000:.2f} ms")
    print(f"  Average per call:     {cached_time*1000/num_calls:.3f} ms")
    print(f"  Cache hits:           {stats['hits']}")
    print(f"  Cache misses:         {stats['misses']}")
    print(f"  Hit rate:             {stats['hit_rate']:.1%}")

    # Estimate time without caching (first call time * num_calls)
    tangent.clear_cache()
    start = time.time()
    df = tangent.grad(test_function)
    _ = df(2.0)
    single_uncached_time = time.time() - start
    estimated_uncached_time = single_uncached_time * num_calls

    print(f"\nEstimated without caching:")
    print(f"  Total time:           {estimated_uncached_time*1000:.2f} ms")
    print(f"  Average per call:     {estimated_uncached_time*1000/num_calls:.3f} ms")

    speedup = estimated_uncached_time / cached_time
    print(f"\nOverall speedup:        {speedup:.1f}x")
    print(f"Time saved:             {(estimated_uncached_time - cached_time)*1000:.2f} ms")
    print("=" * 80)


if __name__ == '__main__':
    benchmark_function_caching()
    benchmark_repeated_calls()
