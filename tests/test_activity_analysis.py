"""
Tests for activity analysis.
"""
import unittest
import gast
from tangent.optimizations.dce import ActivityAnalyzer, DefUseAnalyzer


class TestActivityAnalysis(unittest.TestCase):

    def test_forward_activity(self):
        """Test forward activity propagation."""
        code = """
def f(x, y):
    a = x + 1
    b = a * 2
    c = y + 1
    return b
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = ActivityAnalyzer(func, {'x'}, {'b'})
        active = analyzer.forward_analysis()

        # x, a, b should be active (chain from x)
        self.assertIn('x', active)
        self.assertIn('a', active)
        self.assertIn('b', active)

    def test_backward_activity(self):
        """Test backward activity propagation."""
        code = """
def f(x, y):
    a = x + 1
    b = y + 1
    c = a + b
    return c
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = ActivityAnalyzer(func, {'x', 'y'}, {'c'})
        active = analyzer.backward_analysis()

        # All variables should be active (all affect c)
        self.assertIn('c', active)
        self.assertIn('a', active)
        self.assertIn('b', active)

    def test_unused_variable(self):
        """Test that truly unused variables are identified."""
        code = """
def f(x, y, z):
    a = x + 1
    b = y + 1
    c = z + 1
    return a + b
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = ActivityAnalyzer(func, {'x', 'y', 'z'}, set())

        # Find what's returned
        for stmt in func.body:
            if isinstance(stmt, gast.Return):
                if stmt.value:
                    from tangent.optimizations.dce import VariableCollector
                    return_vars = VariableCollector.collect(stmt.value)

        analyzer.active_outputs = return_vars
        backward = analyzer.backward_analysis()

        # c should not be in backward active (doesn't affect output)
        self.assertNotIn('c', backward)
        # a and b should be active
        self.assertIn('a', backward)
        self.assertIn('b', backward)

    def test_compute_active_variables(self):
        """Test combined forward and backward analysis."""
        code = """
def f(x, y, z):
    a = x + 1
    b = y + 1
    c = z + 1
    d = a + b
    return d
        """
        tree = gast.parse(code)
        func = tree.body[0]

        # Only x is an active input
        analyzer = ActivityAnalyzer(func, {'x'}, {'d'})
        active = analyzer.compute_active_variables()

        # x, a, d should be active (x flows forward to d)
        self.assertIn('x', active)
        self.assertIn('a', active)
        self.assertIn('d', active)

        # y is not an active input, but is used in d
        # So b should be backward active but not forward active from x
        # Therefore b should NOT be in the intersection
        self.assertNotIn('c', active)  # c definitely not active

    def test_selective_activity(self):
        """Test activity analysis with selective inputs."""
        code = """
def f(x, y):
    a = x * x
    b = y * y
    c = a + b
    return c
        """
        tree = gast.parse(code)
        func = tree.body[0]

        # Only x is active input, but c depends on both
        analyzer = ActivityAnalyzer(func, {'x'}, {'c'})

        forward_active = analyzer.forward_analysis()
        # From x: x, a, c should be forward active
        self.assertIn('x', forward_active)
        self.assertIn('a', forward_active)
        self.assertIn('c', forward_active)

        backward_active = analyzer.backward_analysis()
        # For c: c, a, b, x, y should all be backward active
        self.assertIn('c', backward_active)
        self.assertIn('a', backward_active)
        self.assertIn('b', backward_active)
        self.assertIn('x', backward_active)
        self.assertIn('y', backward_active)

        # Combined (intersection): Only x, a, c
        active = analyzer.compute_active_variables()
        self.assertIn('x', active)
        self.assertIn('a', active)
        self.assertIn('c', active)
        # y and b are backward active but not forward active from x
        self.assertNotIn('y', active)
        self.assertNotIn('b', active)


class TestActivityIntegration(unittest.TestCase):
    """Integration tests with Tangent."""

    def test_with_tangent_gradient(self):
        """Test activity analysis improves DCE for gradients."""
        import tangent

        def f(x, y, z):
            # Complex function
            a = x * x
            b = y * y
            c = z * z  # Unused
            temp1 = a + b
            temp2 = c + 100  # Unused chain
            return temp1

        # Get gradient with activity analysis
        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True}, verbose=0)

        # Should work correctly
        result = grad_f(3.0, 4.0, 5.0)
        self.assertAlmostEqual(result, 6.0)  # d(x*x)/dx = 2x = 6


if __name__ == '__main__':
    unittest.main()
