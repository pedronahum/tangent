"""
Tests for DCE with control flow (if/for/while).
"""
import unittest
import tangent


class TestControlFlowDCE(unittest.TestCase):
    """Test DCE with control flow structures."""

    def test_if_statement_with_unused_branch(self):
        """Test that DCE handles if statements correctly."""
        def f(x, y, flag):
            if flag > 0:
                a = x * x
                result = a
            else:
                b = y * y  # This branch not taken when flag > 0
                result = b

            return result

        # Test with flag > 0 (else branch not taken at runtime)
        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(3.0, 4.0, 1.0)

        # Should get d(x*x)/dx = 2x = 6
        self.assertAlmostEqual(result, 6.0)

    def test_for_loop_with_accumulation(self):
        """Test DCE with for loops."""
        def f(x, y):
            result = 0.0
            for i in [1.0, 2.0, 3.0]:
                result = result + x * i

            # y unused
            unused = y * y

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(2.0, 5.0)

        # d(sum(x * i for i in [1,2,3]))/dx = sum(i) = 1 + 2 + 3 = 6
        self.assertAlmostEqual(result, 6.0)

    def test_while_loop(self):
        """Test DCE with while loops."""
        def f(x, y):
            result = x
            counter = 0.0

            while counter < 3.0:
                result = result + x
                counter = counter + 1.0

            # y unused
            unused = y * y

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(2.0, 5.0)

        # result = x + x + x + x = 4x, so gradient = 4
        self.assertAlmostEqual(result, 4.0)

    def test_nested_if(self):
        """Test DCE with nested if statements."""
        def f(x, y, z):
            if x > 0:
                if y > 0:
                    result = x * y
                else:
                    result = x * 2.0
            else:
                result = 0.0

            # z unused
            unused = z * z

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(3.0, 4.0, 5.0)

        # d(x*y)/dx = y = 4
        self.assertAlmostEqual(result, 4.0)

    def test_for_loop_with_conditional(self):
        """Test DCE with loop containing conditional."""
        def f(x, y):
            result = 0.0
            for i in [1.0, 2.0, 3.0]:
                if i > 1.5:
                    result = result + x * i
                else:
                    result = result + x

            # y unused
            unused = y * y

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(2.0, 5.0)

        # i=1: result += x (gradient: 1)
        # i=2: result += x*2 (gradient: 2)
        # i=3: result += x*3 (gradient: 3)
        # Total gradient: 1 + 2 + 3 = 6
        self.assertAlmostEqual(result, 6.0)

    def test_conditional_eliminates_unused_vars(self):
        """Test that variables only used in dead branches are eliminated."""
        def f(x, y, z):
            # Compute many things
            a = x * x
            b = y * y
            c = z * z

            # Only use a
            if True:
                result = a
            else:
                result = b + c  # Dead branch

            return result

        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        result = grad_f(3.0, 4.0, 5.0)

        # d(x*x)/dx = 2x = 6
        self.assertAlmostEqual(result, 6.0)


class TestControlFlowCorrectness(unittest.TestCase):
    """Verify DCE doesn't break correctness with control flow."""

    def test_loop_correctness(self):
        """Ensure loop gradients are correct with DCE."""
        def f(x):
            result = 0.0
            for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
                result = result + x * i

            return result

        # With DCE
        grad_f_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        # Without DCE
        grad_f_no_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': False})

        x = 2.5
        result_dce = grad_f_dce(x)
        result_no_dce = grad_f_no_dce(x)

        # Should be equal
        self.assertAlmostEqual(result_dce, result_no_dce, places=5)
        # And equal to sum(i for i in [1,2,3,4,5]) = 15
        self.assertAlmostEqual(result_dce, 15.0)

    def test_conditional_correctness(self):
        """Ensure conditional gradients are correct with DCE."""
        def f(x, flag):
            if flag > 0.5:
                result = x * x * x
            else:
                result = x * x

            return result

        # With DCE
        grad_f_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        # Without DCE
        grad_f_no_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': False})

        x = 3.0

        # Test true branch
        result_dce = grad_f_dce(x, 1.0)
        result_no_dce = grad_f_no_dce(x, 1.0)
        self.assertAlmostEqual(result_dce, result_no_dce, places=5)
        # d(x^3)/dx = 3x^2 = 27
        self.assertAlmostEqual(result_dce, 27.0)

        # Test false branch
        result_dce = grad_f_dce(x, 0.0)
        result_no_dce = grad_f_no_dce(x, 0.0)
        self.assertAlmostEqual(result_dce, result_no_dce, places=5)
        # d(x^2)/dx = 2x = 6
        self.assertAlmostEqual(result_dce, 6.0)


if __name__ == '__main__':
    unittest.main()
