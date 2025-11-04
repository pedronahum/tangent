"""
Tests for unified optimization pipeline (Phase 4).
"""
import unittest
import tangent


class TestUnifiedOptimization(unittest.TestCase):
    """Test the unified optimization pipeline."""

    def test_constant_folding_with_dce(self):
        """Test that constant folding creates opportunities for DCE."""
        def f(x, y):
            # Constants that get folded
            a = 2.0 * 3.0  # Will be folded to 6.0
            b = x + a
            # Unused computation
            c = y * y
            return b

        grad_f = tangent.grad(f, wrt=(0,), optimized=True)
        result = grad_f(5.0, 10.0)

        # d(x + 6)/dx = 1
        self.assertAlmostEqual(result, 1.0)

    def test_assignment_propagation_with_dce(self):
        """Test that assignment propagation enhances DCE."""
        def f(x, y):
            a = x * x
            b = a  # Single-use assignment (can be propagated)
            c = y * y  # Unused
            return b

        grad_f = tangent.grad(f, wrt=(0,), optimized=True)
        result = grad_f(3.0, 5.0)

        # d(x*x)/dx = 2x = 6
        self.assertAlmostEqual(result, 6.0)

    def test_multi_pass_optimization(self):
        """Test that multiple optimization passes find more opportunities."""
        def f(x, y, z):
            # First pass: constant folding
            factor = 1.0 * 2.0  # Folds to 2.0
            a = x * factor

            # Second pass: DCE removes y
            b = y * y

            # Third pass: propagation
            c = a
            result = c

            # z completely unused
            unused = z * z * z

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimized=True)
        result = grad_f(3.0, 4.0, 5.0)

        # d(x * 2)/dx = 2
        self.assertAlmostEqual(result, 2.0)

    def test_unified_vs_basic_optimization(self):
        """Verify unified optimization provides better results than basic."""
        def f(x, y, z):
            # Complex computation
            a = x * x
            b = y * y
            c = z * z
            d = a + 10.0  # Constant that could be folded later
            e = b + c  # Unused
            return d

        # With unified optimization
        grad_f_unified = tangent.grad(f, wrt=(0,), optimized=True, optimizations={'dce': True})

        # Without advanced DCE
        grad_f_basic = tangent.grad(f, wrt=(0,), optimized=True, optimizations={'dce': False})

        x, y, z = 2.0, 3.0, 4.0
        result_unified = grad_f_unified(x, y, z)
        result_basic = grad_f_basic(x, y, z)

        # Results should be identical
        self.assertAlmostEqual(result_unified, result_basic, places=10)
        # And equal to d(x*x)/dx = 2x = 4
        self.assertAlmostEqual(result_unified, 4.0)

    def test_loop_optimization(self):
        """Test optimization with loops."""
        def f(x, y):
            result = 0.0
            for i in [1.0, 2.0, 3.0]:
                # Constant expression in loop
                factor = 1.0 + 1.0  # Should be folded
                result = result + x * factor

            # Unused
            unused_loop = 0.0
            for j in [1.0, 2.0]:
                unused_loop = unused_loop + y

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimized=True)
        result = grad_f(2.0, 5.0)

        # d(3 * x * 2)/dx = 6
        self.assertAlmostEqual(result, 6.0)

    def test_conditional_optimization(self):
        """Test optimization with conditionals."""
        def f(x, y, flag):
            # Constants
            c1 = 2.0 * 2.0
            c2 = 3.0 * 3.0

            if flag > 0:
                result = x * c1
            else:
                result = x * c2

            # Unused
            unused = y * y

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimized=True)

        # True branch
        result = grad_f(2.0, 5.0, 1.0)
        # d(x * 4)/dx = 4
        self.assertAlmostEqual(result, 4.0)

        # False branch
        result = grad_f(2.0, 5.0, -1.0)
        # d(x * 9)/dx = 9
        self.assertAlmostEqual(result, 9.0)


class TestOptimizationCorrectness(unittest.TestCase):
    """Verify optimization doesn't break correctness."""

    def test_complex_function_correctness(self):
        """Test a complex function with multiple optimization opportunities."""
        def complex_model(x, w1, w2, w3, learning_rate, momentum):
            # Forward pass
            h1 = x * w1
            h2 = h1 * w2
            output = h2 * w3

            # Constants
            scale = 1.0 / 100.0

            # Unused computations
            reg = w1 * w1 + w2 * w2 + w3 * w3
            velocity = momentum * 0.9
            adjusted_lr = learning_rate * scale

            return output

        # Get gradient w.r.t. w1 with full optimization
        grad_optimized = tangent.grad(complex_model, wrt=(1,), optimized=True, optimizations={'dce': True})

        # Get gradient without optimization
        grad_basic = tangent.grad(complex_model, wrt=(1,), optimized=False)

        # Test values
        args = (2.0, 0.5, 0.3, 0.7, 0.01, 0.9)

        result_opt = grad_optimized(*args)
        result_basic = grad_basic(*args)

        # Results must be identical
        self.assertAlmostEqual(result_opt, result_basic, places=8)

    def test_numerical_stability(self):
        """Verify optimizations don't affect numerical results."""
        def f(x, y, z):
            a = x * x + y * y
            b = a / 2.0
            c = b + z
            unused = x * y * z  # Dead code
            return c - z  # Should simplify to a/2

        grad_f_opt = tangent.grad(f, wrt=(0,), optimized=True)
        grad_f_no_opt = tangent.grad(f, wrt=(0,), optimized=False)

        for x in [1.0, 10.0, 100.0]:
            for y in [1.0, 10.0, 100.0]:
                for z in [1.0, 10.0, 100.0]:
                    r_opt = grad_f_opt(x, y, z)
                    r_no_opt = grad_f_no_opt(x, y, z)
                    self.assertAlmostEqual(r_opt, r_no_opt, places=10)


if __name__ == '__main__':
    unittest.main()
