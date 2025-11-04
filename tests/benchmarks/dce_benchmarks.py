"""
Benchmark suite for Dead Code Elimination.
Tests various scenarios where DCE should provide benefits.
"""
import time
import tracemalloc
import tangent
import numpy as np


class Benchmark:
    """Base class for DCE benchmarks."""

    def __init__(self, name):
        self.name = name
        self.results = {}

    def time_it(self, func, *args, iterations=100, warmup=10):
        """Time a function with warmup."""
        # Warmup
        for _ in range(warmup):
            func(*args)

        # Time
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        end = time.perf_counter()

        return (end - start) / iterations

    def memory_it(self, func, *args):
        """Measure peak memory usage."""
        tracemalloc.start()
        func(*args)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024 / 1024  # Convert to MB

    def run(self):
        """Override in subclasses."""
        raise NotImplementedError


class SelectiveGradientBenchmark(Benchmark):
    """
    Benchmark: Computing gradient w.r.t. only 1 parameter out of many.
    Expected speedup: 2-5× with DCE
    """

    def __init__(self):
        super().__init__("Selective Gradient (1 of N parameters)")

    def setup(self, n_params=10):
        """Create function with many parameters (reduced to 10 for testing)."""
        def model(x, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
            # Chain computation through all parameters
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

        self.func = model
        self.x = 5.0
        self.params = [float(i + 1) for i in range(n_params)]
        self.n_params = n_params

    def run(self):
        # Baseline: gradient w.r.t. x only
        grad_func = tangent.grad(self.func, wrt=(0,))

        time_baseline = self.time_it(grad_func, self.x, *self.params)
        mem_baseline = self.memory_it(grad_func, self.x, *self.params)

        self.results['baseline_time'] = time_baseline
        self.results['baseline_memory'] = mem_baseline
        self.results['n_params'] = self.n_params

        return self.results


class UnusedComputationBenchmark(Benchmark):
    """
    Benchmark: Function with unused branches/computations.
    Expected speedup: 1.5-3× with DCE
    """

    def __init__(self):
        super().__init__("Unused Computation Elimination")

    def setup(self):
        def model(x, y, z):
            # Expensive computation on x (used)
            a = x**0 + x**1 + x**2 + x**3 + x**4 + x**5 + x**6 + x**7 + x**8 + x**9

            # Expensive computation on y (used)
            b = y**0 + y**1 + y**2 + y**3 + y**4 + y**5 + y**6 + y**7 + y**8 + y**9

            # Expensive computation on z (UNUSED!)
            c = z**0 + z**1 + z**2 + z**3 + z**4 + z**5 + z**6 + z**7 + z**8 + z**9

            return a + b  # c is never used!

        self.func = model
        self.x = 2.0
        self.y = 3.0
        self.z = 4.0

    def run(self):
        # Gradient w.r.t. x only (y and z computations could be eliminated)
        grad_func = tangent.grad(self.func, wrt=(0,))

        time_baseline = self.time_it(grad_func, self.x, self.y, self.z)
        mem_baseline = self.memory_it(grad_func, self.x, self.y, self.z)

        self.results['baseline_time'] = time_baseline
        self.results['baseline_memory'] = mem_baseline

        return self.results


class UnusedRegularizationBenchmark(Benchmark):
    """
    Benchmark: Regularization term computed but not used.
    Expected speedup: 1.5-2× with DCE
    """

    def __init__(self):
        super().__init__("Unused Regularization Term")

    def setup(self):
        def neural_net(x, w1, w2, w3, reg_weight):
            # Forward pass
            h1 = x * w1
            h2 = h1 * w2
            output = h2 * w3

            # Regularization (expensive, but not used in return!)
            reg = reg_weight * (w1**2 + w2**2 + w3**2)

            # Oops! Forgot to add regularization to loss
            return output  # Should be: output + reg

        self.func = neural_net
        self.x = 1.0
        self.w1 = 0.5
        self.w2 = 0.3
        self.w3 = 0.7
        self.reg_weight = 0.01

    def run(self):
        grad_func = tangent.grad(self.func, wrt=(1,))  # w.r.t. w1

        time_baseline = self.time_it(
            grad_func, self.x, self.w1, self.w2, self.w3, self.reg_weight
        )
        mem_baseline = self.memory_it(
            grad_func, self.x, self.w1, self.w2, self.w3, self.reg_weight
        )

        self.results['baseline_time'] = time_baseline
        self.results['baseline_memory'] = mem_baseline

        return self.results


class ConditionalBranchBenchmark(Benchmark):
    """
    Benchmark: Dead branches in conditionals.
    Expected speedup: 1.3-2× with DCE
    """

    def __init__(self):
        super().__init__("Conditional Dead Branch")

    def setup(self):
        def model(x, y, flag):
            if flag > 0:
                a = x * x * x  # Expensive
                b = y  # Simple
            else:
                a = x  # Simple
                b = y * y * y  # Expensive (but this branch never taken!)

            return a + b

        self.func = model
        self.x = 2.0
        self.y = 3.0
        self.flag = 1.0  # Always positive, else branch is dead

    def run(self):
        grad_func = tangent.grad(self.func, wrt=(0,))  # w.r.t. x

        time_baseline = self.time_it(grad_func, self.x, self.y, self.flag)
        mem_baseline = self.memory_it(grad_func, self.x, self.y, self.flag)

        self.results['baseline_time'] = time_baseline
        self.results['baseline_memory'] = mem_baseline

        return self.results


def run_all_benchmarks():
    """Run all benchmarks and display results."""
    benchmarks = [
        SelectiveGradientBenchmark(),
        UnusedComputationBenchmark(),
        UnusedRegularizationBenchmark(),
        ConditionalBranchBenchmark(),
    ]

    print("=" * 80)
    print("TANGENT DCE BENCHMARK SUITE - BASELINE")
    print("=" * 80)
    print()

    all_results = {}

    for bench in benchmarks:
        print(f"Running: {bench.name}")
        bench.setup()
        try:
            results = bench.run()
            all_results[bench.name] = results

            print(f"  Time: {results['baseline_time']*1000:.3f} ms")
            print(f"  Memory: {results['baseline_memory']:.2f} MB")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

    return all_results


if __name__ == "__main__":
    import os
    results = run_all_benchmarks()

    # Save baseline results in the benchmarks directory
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Baseline results saved to {results_path}")
