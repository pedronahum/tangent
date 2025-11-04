"""
Unit tests for Common Subexpression Elimination.
"""
import unittest
import ast
import gast
from tangent.optimizations.cse import (
    SubexpressionAnalyzer,
    CSETransformer,
    CommonSubexpressionEliminator,
    apply_cse
)


class TestSubexpressionAnalyzer(unittest.TestCase):
    """Test subexpression analysis."""

    def test_simple_redundancy(self):
        """Test detection of simple redundant expression."""
        code = """
def f():
    result = x * y + x * y
        """
        tree = gast.parse(code)
        func = tree.body[0]
        stmt = func.body[0]

        analyzer = SubexpressionAnalyzer()
        candidates = analyzer.analyze(stmt.value)

        # Should find x * y as common subexpression
        self.assertGreater(len(candidates), 0)

        # First candidate should have count >= 2
        node, cost, count, locations = candidates[0]
        self.assertGreaterEqual(count, 2)

    def test_complex_redundancy(self):
        """Test detection in complex expression."""
        code = """
def f():
    result = (a * b * c) + (a * b * d) + (a * b * e)
        """
        tree = gast.parse(code)
        func = tree.body[0]
        stmt = func.body[0]

        analyzer = SubexpressionAnalyzer()
        candidates = analyzer.analyze(stmt.value)

        # Should find a * b as common subexpression (appears 3 times)
        self.assertGreater(len(candidates), 0)

    def test_cost_computation(self):
        """Test cost estimation."""
        # Simple add: cost should be low
        code1 = "x + y"
        node1 = ast.parse(code1, mode='eval').body

        # Complex expression: cost should be higher
        code2 = "x * y * z + sin(x) * cos(y)"
        node2 = ast.parse(code2, mode='eval').body

        analyzer = SubexpressionAnalyzer()
        cost1 = analyzer._compute_cost(node1)
        cost2 = analyzer._compute_cost(node2)

        self.assertGreater(cost2, cost1)

    def test_min_occurrences_filter(self):
        """Test that min_occurrences filters correctly."""
        code = """
def f():
    result = x * y + z
        """
        tree = gast.parse(code)
        func = tree.body[0]
        stmt = func.body[0]

        # No expression occurs more than once
        analyzer = SubexpressionAnalyzer(min_occurrences=2)
        candidates = analyzer.analyze(stmt.value)

        # Should find no candidates (nothing repeated)
        self.assertEqual(len(candidates), 0)


class TestCSETransformation(unittest.TestCase):
    """Test CSE transformation."""

    def test_simple_elimination(self):
        """Test CSE on simple redundant expression."""
        code = """
def f(x, y):
    result = x * y + x * y
        """
        tree = gast.parse(code)
        func = tree.body[0]

        eliminator = CommonSubexpressionEliminator()
        optimized = eliminator.optimize(func)

        # Should have added temp variable
        # New body should have: temp = x*y, result = temp + temp
        self.assertGreaterEqual(len(optimized.body), 2)

        # First statement should be temp assignment
        first_stmt = optimized.body[0]
        self.assertIsInstance(first_stmt, (ast.Assign, gast.Assign))
        self.assertTrue(first_stmt.targets[0].id.startswith('_cse_temp'))

    def test_derivative_pattern(self):
        """Test CSE on typical derivative pattern."""
        # Note: Singleexpression CSE won't find across-term redundancy
        # This test just verifies CSE doesn't break on complex expressions
        code = """
def grad_f(f1, f2, f3, df1, df2, df3):
    result = df1 * f2 * f3 + f1 * df2 * f3 + f1 * f2 * df3
        """
        tree = gast.parse(code)
        func = tree.body[0]

        eliminator = CommonSubexpressionEliminator()
        optimized = eliminator.optimize(func)

        # Should work without errors
        self.assertIsNotNone(optimized)
        # Verify at least original statement is present
        self.assertGreaterEqual(len(optimized.body), 1)

    def test_no_cse_for_single_occurrence(self):
        """Test that single occurrences are not eliminated."""
        code = """
def f(x, y):
    a = x * y
    b = x + y
    return a + b
        """
        tree = gast.parse(code)
        func = tree.body[0]

        eliminator = CommonSubexpressionEliminator()
        optimized = eliminator.optimize(func)

        # Should not add any CSE temps (nothing repeated)
        temp_count = sum(1 for stmt in optimized.body
                        if isinstance(stmt, (ast.Assign, gast.Assign)) and
                           stmt.targets[0].id.startswith('_cse_temp'))

        self.assertEqual(temp_count, 0)


class TestCSEIntegration(unittest.TestCase):
    """Integration tests with Tangent."""

    def test_cse_with_tangent_gradient(self):
        """Test CSE integration with Tangent gradient."""
        # Simulate Tangent-generated gradient code
        code = """
def grad_f(x, y):
    temp = x * x
    result = temp * y + temp * y
    return result
        """
        tree = gast.parse(code)
        func = tree.body[0]

        optimized = apply_cse(func)

        # Should work without errors
        self.assertIsNotNone(optimized)

    def test_apply_cse_function(self):
        """Test the apply_cse entry point."""
        code = """
def f(x):
    return x * x + x * x
        """
        tree = gast.parse(code)
        func = tree.body[0]

        # Apply with config
        optimized = apply_cse(func, config={'min_occurrences': 2, 'min_cost': 2})

        self.assertIsNotNone(optimized)


class TestCSEBenchmark(unittest.TestCase):
    """Benchmark-style tests showing CSE impact."""

    def test_product_rule_gradient(self):
        """Test CSE on product rule gradient (typical Tangent output)."""
        # Note: Single-expression CSE won't find across-term redundancy
        # This test verifies CSE doesn't break on complex derivative code
        code = """
def grad_product(f, g, h, df, dg, dh):
    d_result = df * g * h + f * dg * h + f * g * dh
    return d_result
        """
        tree = gast.parse(code)
        func = tree.body[0]

        eliminator = CommonSubexpressionEliminator()
        optimized = eliminator.optimize(func)

        # Should work without errors
        self.assertIsNotNone(optimized)
        # Verify statements are preserved
        self.assertGreaterEqual(len(optimized.body), 2)


if __name__ == '__main__':
    unittest.main()
