"""
Realistic performance benchmarks showing CSE and Algebraic Simplification impact.

These tests target specific patterns where symbolic optimizations provide benefit.
"""
import unittest
import tangent
import time
import statistics


def benchmark_function(func, *args, iterations=1000, warmup=100):
    """Benchmark a function by running it multiple times."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'result': result
    }


class TestRealisticPerformance(unittest.TestCase):
    """Performance tests on realistic gradient patterns."""

    def test_backward_pass_redundancy(self):
        """
        Test CSE on backward pass where redundant gradient expressions appear.

        In reverse-mode AD, the backward pass often computes the same
        expression multiple times (e.g., bc * x for product rule).
        CSE should optimize these.
        """
        def forward_function(x, y, z):
            # Forward pass creates expressions that lead to redundancy in backward
            a = x * y * z
            b = x * y * z  # Same computation
            c = a + b
            return c * c  # Square it

        print("\n" + "=" * 70)
        print("REALISTIC BENCHMARK: Backward Pass Redundancy")
        print("=" * 70)
        print("Function: f(x,y,z) = ((x*y*z) + (x*y*z))²")
        print("=" * 70)

        # Generate gradients w.r.t. x
        grad_no_opt = tangent.grad(forward_function, wrt=(0,), optimized=False, verbose=0)
        grad_standard = tangent.grad(forward_function, wrt=(0,), optimized=True,
                                      optimizations={'dce': True, 'cse': False},
                                      verbose=0)
        grad_with_cse = tangent.grad(forward_function, wrt=(0,), optimized=True,
                                       optimizations={'dce': True, 'cse': True},
                                       verbose=0)

        # Benchmark
        test_inputs = (2.0, 3.0, 4.0)
        stats_no_opt = benchmark_function(grad_no_opt, *test_inputs, iterations=2000)
        stats_standard = benchmark_function(grad_standard, *test_inputs, iterations=2000)
        stats_with_cse = benchmark_function(grad_with_cse, *test_inputs, iterations=2000)

        # Verify correctness
        self.assertAlmostEqual(stats_no_opt['result'], stats_standard['result'], places=5)
        self.assertAlmostEqual(stats_no_opt['result'], stats_with_cse['result'], places=5)

        # Calculate speedups
        speedup_standard = stats_no_opt['mean'] / stats_standard['mean']
        speedup_cse = stats_no_opt['mean'] / stats_with_cse['mean']
        speedup_cse_vs_standard = stats_standard['mean'] / stats_with_cse['mean']

        print(f"\n{'Configuration':<25} {'Time (μs)':<15} {'Speedup':<15}")
        print("-" * 70)
        print(f"{'No optimization':<25} {stats_no_opt['mean']:>10.2f} ± {stats_no_opt['std']:<8.2f} {'baseline':<15}")
        print(f"{'Standard (DCE)':<25} {stats_standard['mean']:>10.2f} ± {stats_standard['std']:<8.2f} {speedup_standard:<15.2f}×")
        print(f"{'CSE + DCE':<25} {stats_with_cse['mean']:>10.2f} ± {stats_with_cse['std']:<8.2f} {speedup_cse:<15.2f}×")

        print(f"\n{'Improvement':<40} {'Factor':<15}")
        print("-" * 70)
        print(f"{'CSE over standard:':<40} {speedup_cse_vs_standard:<15.2f}×")
        print(f"{'Gradient value (correctness check):':<40} {stats_with_cse['result']:<15.2f}")
        print("=" * 70)

    def test_chain_rule_optimization(self):
        """
        Test on chain rule where intermediate values are reused.
        """
        def chain_function(x):
            # f(x) = ((x²)² + (x²)²)²
            # This creates redundant x² computations in gradient
            x_squared = x * x
            term1 = x_squared * x_squared
            term2 = x_squared * x_squared  # Redundant
            sum_terms = term1 + term2
            return sum_terms * sum_terms

        print("\n" + "=" * 70)
        print("REALISTIC BENCHMARK: Chain Rule with Redundancy")
        print("=" * 70)
        print("Function: f(x) = ((x²)² + (x²)²)²")
        print("=" * 70)

        # Generate gradients
        grad_no_opt = tangent.grad(chain_function, wrt=(0,), optimized=False, verbose=0)
        grad_standard = tangent.grad(chain_function, wrt=(0,), optimized=True,
                                      optimizations={'dce': True, 'cse': False},
                                      verbose=0)
        grad_with_cse = tangent.grad(chain_function, wrt=(0,), optimized=True,
                                       optimizations={'dce': True, 'cse': True},
                                       verbose=0)

        # Benchmark
        test_input = 2.0
        stats_no_opt = benchmark_function(grad_no_opt, test_input, iterations=2000)
        stats_standard = benchmark_function(grad_standard, test_input, iterations=2000)
        stats_with_cse = benchmark_function(grad_with_cse, test_input, iterations=2000)

        # Verify correctness
        self.assertAlmostEqual(stats_no_opt['result'], stats_standard['result'], places=5)
        self.assertAlmostEqual(stats_no_opt['result'], stats_with_cse['result'], places=5)

        # Calculate speedups
        speedup_standard = stats_no_opt['mean'] / stats_standard['mean']
        speedup_cse = stats_no_opt['mean'] / stats_with_cse['mean']
        speedup_cse_vs_standard = stats_standard['mean'] / stats_with_cse['mean']

        print(f"\n{'Configuration':<25} {'Time (μs)':<15} {'Speedup':<15}")
        print("-" * 70)
        print(f"{'No optimization':<25} {stats_no_opt['mean']:>10.2f} ± {stats_no_opt['std']:<8.2f} {'baseline':<15}")
        print(f"{'Standard (DCE)':<25} {stats_standard['mean']:>10.2f} ± {stats_standard['std']:<8.2f} {speedup_standard:<15.2f}×")
        print(f"{'CSE + DCE':<25} {stats_with_cse['mean']:>10.2f} ± {stats_with_cse['std']:<8.2f} {speedup_cse:<15.2f}×")

        print(f"\n{'Improvement':<40} {'Factor':<15}")
        print("-" * 70)
        print(f"{'CSE over standard:':<40} {speedup_cse_vs_standard:<15.2f}×")
        print(f"{'Gradient value (correctness check):':<40} {stats_with_cse['result']:<15.2f}")
        print("=" * 70)

    def test_all_optimizations_combined(self):
        """
        Test all optimizations together on a realistic function.
        """
        def realistic_ml_function(x, w1, w2):
            # A more realistic ML-style function
            # Hidden layer 1
            h1 = x * w1
            a1 = h1 * h1  # Square activation

            # Hidden layer 2
            h2 = a1 * w2
            a2 = h2 * h2  # Square activation

            # Add identity operations that algebraic could simplify
            result = a2 * 1.0 + 0.0
            return result

        print("\n" + "=" * 70)
        print("REALISTIC BENCHMARK: All Optimizations Combined")
        print("=" * 70)
        print("Function: 2-layer network with identity operations")
        print("=" * 70)

        # Generate gradients w.r.t. w1
        grad_no_opt = tangent.grad(realistic_ml_function, wrt=(1,), optimized=False, verbose=0)
        grad_standard = tangent.grad(realistic_ml_function, wrt=(1,), optimized=True,
                                      optimizations={'dce': True, 'cse': False, 'algebraic': False},
                                      verbose=0)
        grad_with_cse = tangent.grad(realistic_ml_function, wrt=(1,), optimized=True,
                                       optimizations={'dce': True, 'cse': True, 'algebraic': False},
                                       verbose=0)
        grad_all_opts = tangent.grad(realistic_ml_function, wrt=(1,), optimized=True,
                                       optimizations={'dce': True, 'cse': True, 'algebraic': True},
                                       verbose=0)

        # Benchmark
        test_inputs = (2.0, 0.5, 0.3)
        stats_no_opt = benchmark_function(grad_no_opt, *test_inputs, iterations=2000)
        stats_standard = benchmark_function(grad_standard, *test_inputs, iterations=2000)
        stats_with_cse = benchmark_function(grad_with_cse, *test_inputs, iterations=2000)
        stats_all_opts = benchmark_function(grad_all_opts, *test_inputs, iterations=2000)

        # Verify correctness
        self.assertAlmostEqual(stats_no_opt['result'], stats_standard['result'], places=5)
        self.assertAlmostEqual(stats_no_opt['result'], stats_with_cse['result'], places=5)
        self.assertAlmostEqual(stats_no_opt['result'], stats_all_opts['result'], places=5)

        # Calculate speedups
        speedup_standard = stats_no_opt['mean'] / stats_standard['mean']
        speedup_cse = stats_no_opt['mean'] / stats_with_cse['mean']
        speedup_all = stats_no_opt['mean'] / stats_all_opts['mean']

        print(f"\n{'Configuration':<30} {'Time (μs)':<15} {'Speedup':<15}")
        print("-" * 70)
        print(f"{'No optimization':<30} {stats_no_opt['mean']:>10.2f} ± {stats_no_opt['std']:<8.2f} {'baseline':<15}")
        print(f"{'DCE only':<30} {stats_standard['mean']:>10.2f} ± {stats_standard['std']:<8.2f} {speedup_standard:<15.2f}×")
        print(f"{'DCE + CSE':<30} {stats_with_cse['mean']:>10.2f} ± {stats_with_cse['std']:<8.2f} {speedup_cse:<15.2f}×")
        print(f"{'DCE + CSE + Algebraic':<30} {stats_all_opts['mean']:>10.2f} ± {stats_all_opts['std']:<8.2f} {speedup_all:<15.2f}×")

        print(f"\n{'Improvement':<40} {'Factor':<15}")
        print("-" * 70)
        print(f"{'DCE baseline:':<40} {speedup_standard:<15.2f}×")
        print(f"{'CSE added benefit:':<40} {speedup_cse / speedup_standard:<15.2f}×")
        print(f"{'All optimizations:':<40} {speedup_all:<15.2f}×")
        print(f"{'Gradient value (correctness check):':<40} {stats_all_opts['result']:<15.2f}")
        print("=" * 70)


class TestOptimizationImpact(unittest.TestCase):
    """Measure the cumulative impact of optimizations."""

    def test_optimization_breakdown(self):
        """Show contribution of each optimization."""
        def test_function(x, y):
            # Function designed to benefit from all optimizations
            a = x * x + y * y
            b = x * x + y * y  # CSE opportunity
            c = a * 1.0  # Algebraic opportunity
            d = b + 0.0  # Algebraic opportunity
            return c * d

        print("\n" + "=" * 70)
        print("OPTIMIZATION BREAKDOWN ANALYSIS")
        print("=" * 70)
        print("Function with CSE and algebraic opportunities")
        print("=" * 70)

        # Generate gradients with different optimization combinations
        configs = [
            ("No opt", {'dce': False, 'cse': False, 'algebraic': False}),
            ("DCE only", {'dce': True, 'cse': False, 'algebraic': False}),
            ("DCE+CSE", {'dce': True, 'cse': True, 'algebraic': False}),
            ("DCE+Alg", {'dce': True, 'cse': False, 'algebraic': True}),
            ("All", {'dce': True, 'cse': True, 'algebraic': True}),
        ]

        results = []
        test_inputs = (2.0, 3.0)

        for name, opts in configs:
            grad_func = tangent.grad(test_function, wrt=(0,), optimized=True,
                                      optimizations=opts, verbose=0)
            stats = benchmark_function(grad_func, *test_inputs, iterations=1000)
            results.append((name, stats))

        # Print results
        baseline = results[0][1]['mean']
        print(f"\n{'Configuration':<20} {'Time (μs)':<15} {'Speedup vs baseline':<20}")
        print("-" * 70)
        for name, stats in results:
            speedup = baseline / stats['mean']
            print(f"{name:<20} {stats['mean']:>10.2f} ± {stats['std']:<8.2f} {speedup:>15.2f}×")

        # Verify all produce same result
        reference_result = results[0][1]['result']
        print(f"\n{'Correctness Check':<40}")
        print("-" * 70)
        all_correct = True
        for name, stats in results:
            match = abs(stats['result'] - reference_result) < 1e-5
            status = "✓" if match else "✗"
            print(f"{name:<20} {status} {stats['result']:.6f}")
            if not match:
                all_correct = False

        self.assertTrue(all_correct, "All optimizations should produce same result")
        print("=" * 70)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
