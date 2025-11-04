"""
Comprehensive benchmark comparing DCE phases.
Measures performance with and without optimizations.
"""
import time
import tracemalloc
import tangent
import numpy as np
import sys


def time_it(func, *args, iterations=1000, warmup=100):
    """Time a function with warmup."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Time
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()

    return (end - start) / iterations * 1000  # Convert to ms


def benchmark_selective_gradient():
    """Benchmark: Gradient w.r.t. 1 parameter out of many."""
    print("\n" + "="*80)
    print("BENCHMARK 1: Selective Gradient (1 of 10 parameters)")
    print("="*80)

    def model(x, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
        result = x
        result = result * p1 + p1 * p1
        result = result * p2 + p2 * p2
        result = result * p3 + p3 * p3
        result = result * p4 + p4 * p4
        result = result * p5 + p5 * p5
        result = result * p6 + p6 * p6
        result = result * p7 + p7 * p7
        result = result * p8 + p8 * p8
        result = result * p9 + p9 * p9
        result = result * p10 + p10 * p10
        return result

    args = tuple([float(i) for i in range(11)])

    # No optimization
    grad_none = tangent.grad(model, wrt=(0,), optimized=False)
    time_none = time_it(grad_none, *args)

    # Basic optimization only
    grad_basic = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': False})
    time_basic = time_it(grad_basic, *args)

    # Full DCE (all phases)
    grad_dce = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': True})
    time_dce = time_it(grad_dce, *args)

    print(f"No optimization:    {time_none:.4f} ms")
    print(f"Basic optimization: {time_basic:.4f} ms  ({time_none/time_basic:.2f}x speedup)")
    print(f"Full DCE:           {time_dce:.4f} ms  ({time_none/time_dce:.2f}x speedup)")
    print(f"DCE vs Basic:       {time_basic/time_dce:.2f}x additional speedup")

    return {
        'none': time_none,
        'basic': time_basic,
        'dce': time_dce,
        'speedup_total': time_none / time_dce,
        'speedup_dce': time_basic / time_dce
    }


def benchmark_unused_computation():
    """Benchmark: Function with unused variables."""
    print("\n" + "="*80)
    print("BENCHMARK 2: Unused Computation Elimination")
    print("="*80)

    def model(x, y, z):
        # Used computations
        a = x**2 + x**3 + x**4 + x**5
        b = y**2 + y**3 + y**4 + y**5

        # UNUSED computation (z not in wrt)
        c = z**2 + z**3 + z**4 + z**5

        return a + b

    args = (2.0, 3.0, 4.0)

    # No optimization
    grad_none = tangent.grad(model, wrt=(0,), optimized=False)
    time_none = time_it(grad_none, *args)

    # Basic optimization only
    grad_basic = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': False})
    time_basic = time_it(grad_basic, *args)

    # Full DCE
    grad_dce = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': True})
    time_dce = time_it(grad_dce, *args)

    print(f"No optimization:    {time_none:.4f} ms")
    print(f"Basic optimization: {time_basic:.4f} ms  ({time_none/time_basic:.2f}x speedup)")
    print(f"Full DCE:           {time_dce:.4f} ms  ({time_none/time_dce:.2f}x speedup)")
    print(f"DCE vs Basic:       {time_basic/time_dce:.2f}x additional speedup")

    return {
        'none': time_none,
        'basic': time_basic,
        'dce': time_dce,
        'speedup_total': time_none / time_dce,
        'speedup_dce': time_basic / time_dce
    }


def benchmark_loop_optimization():
    """Benchmark: Loop with unused variables."""
    print("\n" + "="*80)
    print("BENCHMARK 3: Loop Optimization")
    print("="*80)

    def model(x, y):
        result = 0.0
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            result = result + x * i

        # Unused loop
        unused = 0.0
        for j in [1.0, 2.0, 3.0, 4.0, 5.0]:
            unused = unused + y * j

        return result

    args = (2.0, 5.0)

    # No optimization
    grad_none = tangent.grad(model, wrt=(0,), optimized=False)
    time_none = time_it(grad_none, *args)

    # Basic optimization only
    grad_basic = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': False})
    time_basic = time_it(grad_basic, *args)

    # Full DCE
    grad_dce = tangent.grad(model, wrt=(0,), optimized=True, optimizations={'dce': True})
    time_dce = time_it(grad_dce, *args)

    print(f"No optimization:    {time_none:.4f} ms")
    print(f"Basic optimization: {time_basic:.4f} ms  ({time_none/time_basic:.2f}x speedup)")
    print(f"Full DCE:           {time_dce:.4f} ms  ({time_none/time_dce:.2f}x speedup)")
    print(f"DCE vs Basic:       {time_basic/time_dce:.2f}x additional speedup")

    return {
        'none': time_none,
        'basic': time_basic,
        'dce': time_dce,
        'speedup_total': time_none / time_dce,
        'speedup_dce': time_basic / time_dce
    }


def benchmark_complex_function():
    """Benchmark: Complex real-world-like function."""
    print("\n" + "="*80)
    print("BENCHMARK 4: Complex Function (Realistic ML Model)")
    print("="*80)

    def ml_model(x, w1, w2, w3, learning_rate, momentum, debug_flag):
        # Constants
        scale = 0.01 * 100.0  # Will be folded to 1.0

        # Forward pass
        h1 = x * w1 * scale
        h2 = h1 * w2
        output = h2 * w3

        # Regularization (computed but not used in return!)
        reg = w1**2 + w2**2 + w3**2

        # Hyperparameters (not in wrt)
        adjusted_lr = learning_rate * 0.1
        vel = momentum * 0.9

        # Debug (uses debug_flag but not relevant)
        if debug_flag > 0.5:
            debug_val = output * 1000.0

        return output

    args = (2.0, 0.5, 0.3, 0.7, 0.01, 0.9, 0.0)

    # No optimization
    grad_none = tangent.grad(ml_model, wrt=(1,), optimized=False)
    time_none = time_it(grad_none, *args)

    # Basic optimization only
    grad_basic = tangent.grad(ml_model, wrt=(1,), optimized=True, optimizations={'dce': False})
    time_basic = time_it(grad_basic, *args)

    # Full DCE
    grad_dce = tangent.grad(ml_model, wrt=(1,), optimized=True, optimizations={'dce': True})
    time_dce = time_it(grad_dce, *args)

    print(f"No optimization:    {time_none:.4f} ms")
    print(f"Basic optimization: {time_basic:.4f} ms  ({time_none/time_basic:.2f}x speedup)")
    print(f"Full DCE:           {time_dce:.4f} ms  ({time_none/time_dce:.2f}x speedup)")
    print(f"DCE vs Basic:       {time_basic/time_dce:.2f}x additional speedup")

    return {
        'none': time_none,
        'basic': time_basic,
        'dce': time_dce,
        'speedup_total': time_none / time_dce,
        'speedup_dce': time_basic / time_dce
    }


def print_summary(results):
    """Print overall summary."""
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    benchmarks = [
        "Selective Gradient",
        "Unused Computation",
        "Loop Optimization",
        "Complex Function"
    ]

    print(f"\n{'Benchmark':<25} {'Total Speedup':<15} {'DCE vs Basic'}")
    print("-" * 80)

    for name, result in zip(benchmarks, results):
        speedup_total = result['speedup_total']
        speedup_dce = result['speedup_dce']
        print(f"{name:<25} {speedup_total:.2f}x{'':<12} {speedup_dce:.2f}x")

    # Calculate averages
    avg_total = sum(r['speedup_total'] for r in results) / len(results)
    avg_dce = sum(r['speedup_dce'] for r in results) / len(results)

    print("-" * 80)
    print(f"{'AVERAGE':<25} {avg_total:.2f}x{'':<12} {avg_dce:.2f}x")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"✓ Full DCE provides {avg_total:.2f}x average speedup over no optimization")
    print(f"✓ DCE adds {avg_dce:.2f}x additional speedup over basic optimization")
    print(f"✓ Combined optimization pipeline is {((avg_total - 1) * 100):.0f}% faster than baseline")
    print("="*80)


def main():
    print("="*80)
    print("COMPREHENSIVE DCE PERFORMANCE BENCHMARK")
    print("="*80)
    print("Comparing: No optimization vs Basic optimization vs Full DCE")
    print("="*80)

    results = []

    try:
        results.append(benchmark_selective_gradient())
    except Exception as e:
        print(f"Error in selective gradient benchmark: {e}")

    try:
        results.append(benchmark_unused_computation())
    except Exception as e:
        print(f"Error in unused computation benchmark: {e}")

    try:
        results.append(benchmark_loop_optimization())
    except Exception as e:
        print(f"Error in loop optimization benchmark: {e}")

    try:
        results.append(benchmark_complex_function())
    except Exception as e:
        print(f"Error in complex function benchmark: {e}")

    if results:
        print_summary(results)

    # Save results
    import json
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'performance_results.json')

    summary = {
        'benchmarks': [
            {'name': 'Selective Gradient', 'results': results[0] if len(results) > 0 else {}},
            {'name': 'Unused Computation', 'results': results[1] if len(results) > 1 else {}},
            {'name': 'Loop Optimization', 'results': results[2] if len(results) > 2 else {}},
            {'name': 'Complex Function', 'results': results[3] if len(results) > 3 else {}},
        ],
        'summary': {
            'avg_total_speedup': sum(r['speedup_total'] for r in results) / len(results) if results else 0,
            'avg_dce_speedup': sum(r['speedup_dce'] for r in results) / len(results) if results else 0,
        }
    }

    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
