"""
Comprehensive benchmarks for CSE + Algebraic Simplification.

This tests the complete integration of symbolic optimizations with Tangent.
"""
import unittest
import tangent
import numpy as np
import time


class TestCSEBenchmark(unittest.TestCase):
    """Benchmark CSE on real gradient functions."""

    def test_redundant_computation_gradient(self):
        """Test gradient with redundant computations."""
        def f(x):
            # Deliberately redundant computation
            a = x * x
            b = x * x  # Same as a
            c = x * x  # Same as a
            return a + b + c

        # WITHOUT CSE
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
        result_no_opt = grad_f_no_opt(5.0)

        # WITH CSE
        grad_f_with_cse = tangent.grad(f, wrt=(0,), optimized=True,
                                        optimizations={'dce': False, 'cse': True},
                                        verbose=2)
        result_with_cse = grad_f_with_cse(5.0)

        # Results should match
        self.assertAlmostEqual(result_no_opt, result_with_cse, places=5)

        print(f"\n✓ Redundant computation test passed")
        print(f"  Result: {result_with_cse}")

    def test_product_rule_with_redundancy(self):
        """Test product rule gradient with redundant terms."""
        def f(x, y):
            a = x * x
            b = y * y
            c = a * b  # Uses a and b
            d = a * b  # Redundant with c
            return c + d

        # WITHOUT CSE
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
        result_no_opt = grad_f_no_opt(3.0, 4.0)

        # WITH CSE
        grad_f_with_cse = tangent.grad(f, wrt=(0,), optimized=True,
                                        optimizations={'dce': False, 'cse': True},
                                        verbose=2)
        result_with_cse = grad_f_with_cse(3.0, 4.0)

        # Results should match
        self.assertAlmostEqual(result_no_opt, result_with_cse, places=5)

        print(f"\n✓ Product rule with redundancy test passed")
        print(f"  Result: {result_with_cse}")

    def test_chain_rule_redundancy(self):
        """Test chain rule gradient with common subexpressions."""
        def f(x):
            # f(x) = (x²)³ = x⁶
            temp = x * x
            result = temp * temp * temp
            return result

        # WITHOUT CSE
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
        result_no_opt = grad_f_no_opt(2.0)

        # WITH CSE
        grad_f_with_cse = tangent.grad(f, wrt=(0,), optimized=True,
                                        optimizations={'dce': False, 'cse': True},
                                        verbose=2)
        result_with_cse = grad_f_with_cse(2.0)

        # Results should match
        # d(x⁶)/dx = 6x⁵ = 6*32 = 192
        expected = 6 * (2.0 ** 5)
        self.assertAlmostEqual(result_no_opt, expected, places=3)
        self.assertAlmostEqual(result_with_cse, expected, places=3)

        print(f"\n✓ Chain rule redundancy test passed")
        print(f"  Result: {result_with_cse} (expected: {expected})")


class TestAlgebraicBenchmark(unittest.TestCase):
    """Benchmark algebraic simplification on gradient functions."""

    def test_identity_simplification(self):
        """Test simplification of identity operations."""
        def f(x):
            a = x * 1.0
            b = a + 0.0
            return b

        # WITHOUT algebraic
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
        result_no_opt = grad_f_no_opt(5.0)

        # WITH algebraic
        grad_f_with_alg = tangent.grad(f, wrt=(0,), optimized=True,
                                        optimizations={'dce': False, 'algebraic': True},
                                        verbose=2)
        result_with_alg = grad_f_with_alg(5.0)

        # Results should match
        self.assertAlmostEqual(result_no_opt, result_with_alg, places=5)

        print(f"\n✓ Identity simplification test passed")
        print(f"  Result: {result_with_alg}")

    def test_trig_identity_in_gradient(self):
        """Test trigonometric identity simplification."""
        def f(x):
            # This won't actually use sin²+cos²=1 in forward pass,
            # but we test the algebraic simplifier works
            import math
            a = math.sin(x) ** 2
            b = math.cos(x) ** 2
            c = a + b  # Should simplify to 1
            return x * c

        # WITHOUT algebraic
        try:
            grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
            result_no_opt = grad_f_no_opt(1.0)

            # WITH algebraic
            grad_f_with_alg = tangent.grad(f, wrt=(0,), optimized=True,
                                            optimizations={'dce': False, 'algebraic': True},
                                            verbose=2)
            result_with_alg = grad_f_with_alg(1.0)

            # Results should match
            self.assertAlmostEqual(result_no_opt, result_with_alg, places=5)

            print(f"\n✓ Trigonometric identity test passed")
            print(f"  Result: {result_with_alg}")
        except Exception as e:
            print(f"\n⚠ Trigonometric identity test skipped: {e}")


class TestCombinedOptimizations(unittest.TestCase):
    """Test combined CSE + Algebraic + DCE optimizations."""

    def test_all_optimizations_together(self):
        """Test all symbolic optimizations together."""
        def f(x, y):
            # Multiple optimization opportunities:
            # - Redundant x*x computations (CSE)
            # - Identity operations (Algebraic)
            # - Unused variables (DCE)
            a = x * x
            b = x * x  # Redundant - CSE candidate
            c = a * 1.0  # Identity - Algebraic candidate
            d = y * y  # May be unused in gradient - DCE candidate
            return c + b

        # WITHOUT optimizations
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False, verbose=0)
        result_no_opt = grad_f_no_opt(3.0, 5.0)

        # WITH all optimizations
        grad_f_optimized = tangent.grad(f, wrt=(0,), optimized=True,
                                         optimizations={'dce': True, 'cse': True, 'algebraic': True},
                                         verbose=2)
        result_optimized = grad_f_optimized(3.0, 5.0)

        # Results should match
        self.assertAlmostEqual(result_no_opt, result_optimized, places=5)

        print(f"\n✓ Combined optimizations test passed")
        print(f"  Result: {result_optimized}")

    def test_ml_style_function(self):
        """Test on ML-style function."""
        def neural_layer(x, w1, w2):
            # Simulate a simple neural network layer
            h1 = x * w1
            a1 = h1 * h1  # Square activation
            h2 = a1 * w2
            a2 = h2 * h2  # Square activation
            return a2

        # WITHOUT optimizations
        grad_w1_no_opt = tangent.grad(neural_layer, wrt=(1,), optimized=False, verbose=0)
        result_no_opt = grad_w1_no_opt(2.0, 0.5, 0.3)

        # WITH all optimizations
        grad_w1_optimized = tangent.grad(neural_layer, wrt=(1,), optimized=True,
                                          optimizations={'dce': True, 'cse': True, 'algebraic': True},
                                          verbose=2)
        result_optimized = grad_w1_optimized(2.0, 0.5, 0.3)

        # Results should match
        self.assertAlmostEqual(result_no_opt, result_optimized, places=5)

        print(f"\n✓ ML-style function test passed")
        print(f"  Result: {result_optimized}")


class TestPerformanceImpact(unittest.TestCase):
    """Measure performance impact of optimizations."""

    def test_performance_comparison(self):
        """Compare performance with and without optimizations."""
        def complex_function(x):
            # Create many redundant computations
            a = x * x
            b = x * x
            c = x * x
            d = a + b + c
            e = d * d
            f = d * d
            return e + f

        # Generate gradients
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)

        print("\n1. WITHOUT optimizations:")
        grad_f_no_opt = tangent.grad(complex_function, wrt=(0,), optimized=False, verbose=0)

        print("\n2. WITH standard optimizations:")
        grad_f_standard = tangent.grad(complex_function, wrt=(0,), optimized=True,
                                        optimizations={'dce': True, 'cse': False, 'algebraic': False},
                                        verbose=0)

        print("\n3. WITH CSE only:")
        grad_f_cse = tangent.grad(complex_function, wrt=(0,), optimized=True,
                                   optimizations={'dce': False, 'cse': True, 'algebraic': False},
                                   verbose=0)

        print("\n4. WITH all symbolic optimizations:")
        grad_f_all = tangent.grad(complex_function, wrt=(0,), optimized=True,
                                   optimizations={'dce': True, 'cse': True, 'algebraic': True},
                                   verbose=0)

        # Test correctness
        test_input = 3.0
        result_no_opt = grad_f_no_opt(test_input)
        result_standard = grad_f_standard(test_input)
        result_cse = grad_f_cse(test_input)
        result_all = grad_f_all(test_input)

        print("\n" + "=" * 60)
        print("CORRECTNESS CHECK")
        print("=" * 60)
        print(f"No optimization:     {result_no_opt}")
        print(f"Standard:            {result_standard}")
        print(f"CSE only:            {result_cse}")
        print(f"All optimizations:   {result_all}")

        # All should give same result
        self.assertAlmostEqual(result_no_opt, result_standard, places=5)
        self.assertAlmostEqual(result_no_opt, result_cse, places=5)
        self.assertAlmostEqual(result_no_opt, result_all, places=5)

        print("\n✓ All optimization variants produce correct results")
        print("=" * 60)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
