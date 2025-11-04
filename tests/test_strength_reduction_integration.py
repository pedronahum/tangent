"""
Integration tests for Strength Reduction with Tangent gradients.
"""
import unittest
import tangent


class TestStrengthReductionIntegration(unittest.TestCase):
    """Test strength reduction integration with Tangent."""

    def test_square_with_gradient(self):
        """Test x**2 strength reduction in gradient."""
        def f(x):
            return x ** 2

        # Without strength reduction
        grad_no_sr = tangent.grad(f, wrt=(0,), optimized=True,
                                   optimizations={'strength_reduction': False},
                                   verbose=0)

        # With strength reduction
        grad_with_sr = tangent.grad(f, wrt=(0,), optimized=True,
                                      optimizations={'strength_reduction': True},
                                      verbose=2)

        # Test correctness
        result_no_sr = grad_no_sr(3.0)
        result_with_sr = grad_with_sr(3.0)

        # d(x²)/dx = 2x = 6
        expected = 6.0
        self.assertAlmostEqual(result_no_sr, expected, places=5)
        self.assertAlmostEqual(result_with_sr, expected, places=5)

        print(f"\n✓ Square reduction test passed: {result_with_sr}")

    def test_strength_reduction_plus_cse(self):
        """Test strength reduction combined with CSE."""
        def f(x):
            # x**2 appears twice - strength reduction creates x*x twice
            # Then CSE should optimize the duplicate x*x
            a = x ** 2
            b = x ** 2
            return a + b

        # With both optimizations
        grad_f = tangent.grad(f, wrt=(0,), optimized=True,
                               optimizations={
                                   'strength_reduction': True,
                                   'cse': True
                               },
                               verbose=2)

        # Test correctness
        result = grad_f(3.0)
        # d(2x²)/dx = 4x = 12
        expected = 12.0
        self.assertAlmostEqual(result, expected, places=5)

        print(f"\n✓ Strength + CSE test passed: {result}")

    def test_polynomial_with_strength_reduction(self):
        """Test polynomial gradient with strength reduction."""
        def f(x):
            # Polynomial: x² + x³
            return x ** 2 + x ** 3

        grad_f = tangent.grad(f, wrt=(0,), optimized=True,
                               optimizations={
                                   'strength_reduction': True,
                                   'cse': True
                               },
                               verbose=2)

        # Test correctness
        result = grad_f(2.0)
        # d(x² + x³)/dx = 2x + 3x² = 4 + 12 = 16
        expected = 2 * 2.0 + 3 * (2.0 ** 2)
        self.assertAlmostEqual(result, expected, places=5)

        print(f"\n✓ Polynomial test passed: {result}")

    def test_division_to_multiplication(self):
        """Test division by constant to multiplication."""
        def f(x):
            # Division should be converted to multiplication
            return x / 2.0

        grad_f = tangent.grad(f, wrt=(0,), optimized=True,
                               optimizations={'strength_reduction': True},
                               verbose=2)

        # Test correctness
        result = grad_f(4.0)
        # d(x/2)/dx = 1/2 = 0.5
        expected = 0.5
        self.assertAlmostEqual(result, expected, places=5)

        print(f"\n✓ Division test passed: {result}")

    def test_all_optimizations_combined(self):
        """Test all symbolic optimizations together."""
        def f(x):
            # Combines: strength reduction, CSE, algebraic
            a = x ** 2
            b = x ** 2  # Duplicate (CSE opportunity)
            c = a * 1.0  # Identity (algebraic opportunity)
            d = b / 2.0  # Division (strength reduction opportunity)
            return c + d

        grad_f = tangent.grad(f, wrt=(0,), optimized=True,
                               optimizations={
                                   'dce': True,
                                   'strength_reduction': True,
                                   'cse': True,
                                   'algebraic': True
                               },
                               verbose=2)

        # Test correctness
        result = grad_f(3.0)

        # Expected: d/dx(x² + x²/2) = d/dx(1.5x²) = 3x = 9
        expected = 3 * 3.0
        self.assertAlmostEqual(result, expected, places=5)

        print(f"\n✓ All optimizations test passed: {result}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
