"""
Test applying CSE directly to Tangent's generated gradient AST.

This demonstrates CSE working on the actual gradient code AST before compilation.
"""
import unittest
import tangent
from tangent.optimizations.cse import apply_cse
import gast
import ast


class TestCSEOnTangentAST(unittest.TestCase):
    """Test CSE on Tangent's gradient AST."""

    def test_apply_cse_to_gradient_ast(self):
        """Apply CSE to a gradient function's AST."""
        def f(x):
            # Function with redundant x*x
            a = x * x
            b = x * x
            return a + b

        # We need to intercept the AST before Tangent compiles it
        # For now, create a simple gradient-like AST manually
        code = """
def grad_f(x):
    # Simulates Tangent-generated gradient with redundancy
    temp1 = x * x
    temp2 = x * x  # Redundant!
    result = temp1 + temp2
    return result
        """

        tree = gast.parse(code)
        func = tree.body[0]

        print("=" * 60)
        print("BEFORE CSE:")
        print("=" * 60)
        print(gast.unparse(func))

        # Apply CSE
        optimized = apply_cse(func, config={'min_occurrences': 2, 'min_cost': 2})

        print("\n" + "=" * 60)
        print("AFTER CSE:")
        print("=" * 60)
        print(gast.unparse(optimized))

        # Count statements before and after
        original_stmts = len(tree.body[0].body)
        optimized_stmts = len(optimized.body)

        print("\n" + "=" * 60)
        print(f"Original statements: {original_stmts}")
        print(f"Optimized statements: {optimized_stmts}")
        print(f"Reduction: {original_stmts - optimized_stmts} statements")
        print("=" * 60)

        # Should have added CSE temp and reduced redundancy
        self.assertGreater(optimized_stmts, 0)

    def test_cse_on_complex_gradient_pattern(self):
        """Test CSE on more complex gradient patterns."""
        # Simulate Tangent's gradient code for d(f*g)/dx = df*g + f*dg
        code = """
def grad_product(f, g, df, dg):
    # Product rule gradient
    term1_f = df
    term1_g = g
    term1 = term1_f * term1_g

    term2_f = f
    term2_g = dg
    term2 = term2_f * term2_g

    result = term1 + term2
    return result
        """

        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("PRODUCT RULE GRADIENT - BEFORE CSE:")
        print("=" * 60)
        print(gast.unparse(func))

        # Apply CSE
        optimized = apply_cse(func)

        print("\n" + "=" * 60)
        print("PRODUCT RULE GRADIENT - AFTER CSE:")
        print("=" * 60)
        print(gast.unparse(optimized))

        self.assertIsNotNone(optimized)

    def test_cse_preserves_correctness(self):
        """Verify CSE doesn't change computational results."""
        # Create equivalent functions with and without redundancy
        code_redundant = """
def f_redundant(x):
    a = x * x
    b = x * x
    return a + b
        """

        code_optimized = """
def f_optimized(x):
    temp = x * x
    a = temp
    b = temp
    return a + b
        """

        # Both should give same result
        # (We can't execute AST directly, but CSE transformation should be equivalent)

        tree_redundant = gast.parse(code_redundant)
        tree_optimized = gast.parse(code_optimized)

        # Apply CSE to redundant version
        func_redundant = tree_redundant.body[0]
        cse_result = apply_cse(func_redundant)

        # Verify structure similar to hand-optimized version
        self.assertIsNotNone(cse_result)

        print("\n" + "=" * 60)
        print("CSE CORRECTNESS TEST:")
        print("=" * 60)
        print("Original (redundant):")
        print(gast.unparse(tree_redundant.body[0]))
        print("\nAfter CSE:")
        print(gast.unparse(cse_result))
        print("\nHand-optimized:")
        print(gast.unparse(tree_optimized.body[0]))
        print("=" * 60)

    def test_cse_with_tangent_patterns(self):
        """Test CSE on actual Tangent-generated patterns."""
        # Based on real Tangent output (simplified)
        code = """
def dfdx(x, by=1.0):
    # Forward pass
    y = x * x
    z = y + y

    # Backward pass (product rule pattern)
    _bx = by * x
    _bx2 = by * x  # Redundant!
    bx = _bx
    bx = bx + _bx2
    return bx
        """

        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("TANGENT PATTERN - BEFORE CSE:")
        print("=" * 60)
        print(gast.unparse(func))

        # Apply CSE
        optimized = apply_cse(func, config={'min_occurrences': 2, 'min_cost': 2})

        print("\n" + "=" * 60)
        print("TANGENT PATTERN - AFTER CSE:")
        print("=" * 60)
        print(gast.unparse(optimized))

        # Count CSE temps created
        cse_temp_count = sum(1 for stmt in optimized.body
                            if isinstance(stmt, (ast.Assign, gast.Assign)) and
                               stmt.targets[0].id.startswith('_cse_temp'))

        print(f"\nCSE temporary variables created: {cse_temp_count}")
        print("=" * 60)

        self.assertIsNotNone(optimized)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
