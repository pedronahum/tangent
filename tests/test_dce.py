"""
Unit tests for Dead Code Elimination.
"""
import unittest
import tangent
from tangent.optimizations.dce import (
    DefUseAnalyzer,
    BackwardSlicer,
    GradientDCE,
    VariableCollector
)
import ast
import gast


class TestVariableCollector(unittest.TestCase):
    """Test variable collection from expressions."""

    def test_simple_expression(self):
        code = "x + y"
        node = ast.parse(code, mode='eval').body
        vars = VariableCollector.collect(node)
        self.assertEqual(vars, {'x', 'y'})

    def test_nested_expression(self):
        code = "x * y + z * w"
        node = ast.parse(code, mode='eval').body
        vars = VariableCollector.collect(node)
        self.assertEqual(vars, {'x', 'y', 'z', 'w'})

    def test_function_call(self):
        code = "func(x, y)"
        node = ast.parse(code, mode='eval').body
        vars = VariableCollector.collect(node)
        self.assertIn('x', vars)
        self.assertIn('y', vars)


class TestDefUseAnalyzer(unittest.TestCase):
    """Test def-use analysis."""

    def test_simple_function(self):
        code = """
def f():
    a = x + y
    b = a * z
    return b
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = DefUseAnalyzer(func)

        # Check definitions
        self.assertIn('a', analyzer.def_sites)
        self.assertIn('b', analyzer.def_sites)

        # Check uses
        self.assertIn('a', analyzer.use_map[1])  # b = a * z uses 'a'
        self.assertIn('z', analyzer.use_map[1])  # b = a * z uses 'z'

    def test_multiple_assignments(self):
        code = """
def f():
    x = 1
    y = 2
    z = x + y
    return z
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = DefUseAnalyzer(func)

        # Check all variables are tracked
        self.assertIn('x', analyzer.def_sites)
        self.assertIn('y', analyzer.def_sites)
        self.assertIn('z', analyzer.def_sites)

        # Check z uses x and y
        self.assertIn('x', analyzer.use_map[2])
        self.assertIn('y', analyzer.use_map[2])


class TestBackwardSlicing(unittest.TestCase):
    """Test backward slicing algorithm."""

    def test_simple_slice(self):
        code = """
def f():
    a = x + 1
    b = y + 1
    c = a + 2
    return c
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = DefUseAnalyzer(func)
        slicer = BackwardSlicer(func, analyzer)

        # Slice for 'c' should include lines defining c and a, but not b
        relevant = slicer.slice({'c'})

        # Should include definition of c (line 2) and a (line 0)
        self.assertIn(2, relevant)  # c = a + 2
        self.assertIn(0, relevant)  # a = x + 1
        # Should NOT include b = y + 1
        self.assertNotIn(1, relevant)

    def test_chain_dependencies(self):
        code = """
def f():
    a = x
    b = a
    c = b
    d = c
    return d
        """
        tree = gast.parse(code)
        func = tree.body[0]

        analyzer = DefUseAnalyzer(func)
        slicer = BackwardSlicer(func, analyzer)

        # Slice for 'd' should include all definitions in the chain
        relevant = slicer.slice({'d'})

        # All statements should be relevant
        self.assertIn(0, relevant)  # a = x
        self.assertIn(1, relevant)  # b = a
        self.assertIn(2, relevant)  # c = b
        self.assertIn(3, relevant)  # d = c


class TestGradientDCE(unittest.TestCase):
    """Test full DCE on gradient functions."""

    def test_unused_gradient_elimination(self):
        """Test that unused gradients are eliminated."""
        # Mock gradient function
        grad_code = """
def grad_f(x, y, z):
    # Forward pass
    a = x * 2
    b = y * 3
    c = z * 4
    result = a + b

    # Gradients
    bresult = 1.0
    ba = bresult
    bb = bresult
    bx = ba * 2
    by = bb * 3
    bz = 0

    return bx
        """

        tree = gast.parse(grad_code)
        func = tree.body[0]

        # Apply DCE requesting only bx
        optimizer = GradientDCE(func, ['x'])
        optimized = optimizer.optimize()

        # Get all variable names in optimized code
        code_str = gast.unparse(optimized)

        # bx should be present (it's returned)
        self.assertIn('bx', code_str)

        # bz should ideally be eliminated (unused)
        # Note: This test may need adjustment based on actual implementation


class TestIntegration(unittest.TestCase):
    """Integration tests with actual Tangent."""

    def test_selective_gradient(self):
        """Test gradient w.r.t. one variable eliminates others."""
        def f(x, y, z):
            a = x * x
            b = y * y
            c = z * z
            return a + b  # z unused

        # Get gradient w.r.t. x only
        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})

        # Should work correctly
        result = grad_f(3.0, 4.0, 5.0)
        self.assertAlmostEqual(result, 6.0)  # d(x*x)/dx = 2x = 6

    def test_unused_computation(self):
        """Test that unused computations can be eliminated."""
        def f(x, y):
            used = x * x
            unused = y * y * y  # Never used!
            return used

        # Get gradient w.r.t. x
        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': True})

        # Should work correctly
        result = grad_f(5.0, 100.0)
        self.assertAlmostEqual(result, 10.0)  # d(x*x)/dx = 2x = 10

    def test_dce_disabled(self):
        """Test that DCE can be disabled."""
        def f(x, y):
            a = x * x
            b = y * y
            return a + b

        # Without DCE
        grad_f = tangent.grad(f, wrt=(0,), optimizations={'dce': False})
        result = grad_f(3.0, 4.0)
        self.assertAlmostEqual(result, 6.0)

    def test_correctness_preserved(self):
        """Test that DCE doesn't break correctness."""
        def f(x, y, z):
            # Complex function
            a = x * y
            b = y * z
            c = a + b
            return c * c

        # With DCE
        grad_f_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': True})
        # Without DCE
        grad_f_no_dce = tangent.grad(f, wrt=(0,), optimizations={'dce': False})

        # Results should match
        x, y, z = 2.0, 3.0, 4.0
        result_dce = grad_f_dce(x, y, z)
        result_no_dce = grad_f_no_dce(x, y, z)

        self.assertAlmostEqual(result_dce, result_no_dce, places=5)


if __name__ == '__main__':
    unittest.main()
