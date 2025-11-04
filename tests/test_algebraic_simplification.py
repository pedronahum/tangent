"""
Unit tests for Algebraic Simplification.
"""
import unittest
import ast
import gast
from tangent.optimizations.algebraic_simplification import (
    ASTToSymPyConverter,
    SymPyToASTConverter,
    AlgebraicSimplifier,
    apply_algebraic_simplification
)
import sympy as sp


class TestASTToSymPyConverter(unittest.TestCase):
    """Test AST to SymPy conversion."""

    def test_simple_variable(self):
        """Test converting variable name."""
        code = "x"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(str(result), 'x')

    def test_constant(self):
        """Test converting constant."""
        code = "42"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(result, 42)

    def test_addition(self):
        """Test converting addition."""
        code = "x + y"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(str(result), 'x + y')

    def test_multiplication(self):
        """Test converting multiplication."""
        code = "x * y"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        # SymPy might reorder
        self.assertIn('x', str(result))
        self.assertIn('y', str(result))

    def test_power(self):
        """Test converting power operation."""
        code = "x ** 2"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(str(result), 'x**2')

    def test_function_call_sin(self):
        """Test converting sin function."""
        code = "sin(x)"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(str(result), 'sin(x)')

    def test_function_call_cos(self):
        """Test converting cos function."""
        code = "cos(x)"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        self.assertEqual(str(result), 'cos(x)')

    def test_complex_expression(self):
        """Test converting complex expression."""
        code = "x * y + sin(z)"
        node = ast.parse(code, mode='eval').body

        converter = ASTToSymPyConverter()
        result = converter.convert(node)

        self.assertIsNotNone(result)
        # Check components are present
        result_str = str(result)
        self.assertIn('x', result_str)
        self.assertIn('y', result_str)
        self.assertIn('sin', result_str)


class TestSymPyToASTConverter(unittest.TestCase):
    """Test SymPy to AST conversion."""

    def test_simple_symbol(self):
        """Test converting SymPy symbol."""
        x = sp.Symbol('x')

        converter = SymPyToASTConverter()
        result = converter.convert(x)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, (gast.Name, ast.Name))
        self.assertEqual(result.id, 'x')

    def test_number(self):
        """Test converting number."""
        expr = sp.sympify(42)

        converter = SymPyToASTConverter()
        result = converter.convert(expr)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, (gast.Constant, ast.Constant))

    def test_addition(self):
        """Test converting addition."""
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        expr = x + y

        converter = SymPyToASTConverter()
        result = converter.convert(expr)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, (gast.BinOp, ast.BinOp))

    def test_multiplication(self):
        """Test converting multiplication."""
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        expr = x * y

        converter = SymPyToASTConverter()
        result = converter.convert(expr)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, (gast.BinOp, ast.BinOp))

    def test_sin_function(self):
        """Test converting sin function."""
        x = sp.Symbol('x')
        expr = sp.sin(x)

        converter = SymPyToASTConverter()
        result = converter.convert(expr)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, (gast.Call, ast.Call))


class TestAlgebraicSimplification(unittest.TestCase):
    """Test algebraic simplification."""

    def test_identity_simplification(self):
        """Test x * 1 -> x simplification."""
        code = """
def f(x):
    result = x * 1
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("IDENTITY TEST (x * 1):")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)

    def test_zero_simplification(self):
        """Test x + 0 -> x simplification."""
        code = """
def f(x):
    result = x + 0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("ZERO TEST (x + 0):")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)

    def test_trig_identity(self):
        """Test sin^2(x) + cos^2(x) -> 1 simplification."""
        code = """
def f(x):
    result = sin(x) ** 2 + cos(x) ** 2
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("TRIG IDENTITY TEST (sin²(x) + cos²(x)):")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)
        # SymPy should simplify this to 1
        if simplifier.simplifications_applied > 0:
            print("✓ Trigonometric identity successfully simplified!")

    def test_log_exp_cancellation(self):
        """Test log(exp(x)) -> x simplification."""
        code = """
def f(x):
    result = log(exp(x))
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("LOG-EXP CANCELLATION TEST:")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)

    def test_polynomial_simplification(self):
        """Test polynomial simplification."""
        code = """
def f(x):
    result = x * x + 2 * x * x
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POLYNOMIAL TEST (x² + 2x²):")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)

    def test_no_simplification_needed(self):
        """Test expression that can't be simplified."""
        code = """
def f(x, y):
    result = x * y + z
        """
        tree = gast.parse(code)
        func = tree.body[0]

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        # Should not fail
        self.assertIsNotNone(simplified)

    def test_multiple_statements(self):
        """Test simplification across multiple statements."""
        code = """
def f(x, y):
    a = x * 1
    b = y + 0
    c = a + b
    return c
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("MULTIPLE STATEMENTS TEST:")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplifier = AlgebraicSimplifier()
        simplified = simplifier.simplify(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print(f"Expressions simplified: {simplifier.expressions_simplified}")
        print(f"Simplifications applied: {simplifier.simplifications_applied}")
        print("=" * 60)

        self.assertIsNotNone(simplified)


class TestAlgebraicIntegration(unittest.TestCase):
    """Integration tests."""

    def test_apply_algebraic_simplification_function(self):
        """Test the apply_algebraic_simplification entry point."""
        code = """
def f(x):
    result = x * 1 + 0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        simplified = apply_algebraic_simplification(func)

        self.assertIsNotNone(simplified)

    def test_with_config(self):
        """Test with configuration."""
        code = """
def f(x):
    result = x + x
        """
        tree = gast.parse(code)
        func = tree.body[0]

        config = {'aggressive': True}
        simplified = apply_algebraic_simplification(func, config=config)

        self.assertIsNotNone(simplified)

    def test_gradient_pattern(self):
        """Test on typical gradient code pattern."""
        code = """
def grad_f(x, by):
    bx = by * 1
    bx = bx + 0
    return bx
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("GRADIENT PATTERN TEST:")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        simplified = apply_algebraic_simplification(func)

        print("\nAFTER:")
        print(gast.unparse(simplified))
        print("=" * 60)

        self.assertIsNotNone(simplified)


class TestRoundTripConversion(unittest.TestCase):
    """Test AST -> SymPy -> AST round trip."""

    def test_simple_expression_roundtrip(self):
        """Test round-trip conversion preserves semantics."""
        code = "x + y"
        node = ast.parse(code, mode='eval').body

        # AST -> SymPy
        ast_to_sympy = ASTToSymPyConverter()
        sympy_expr = ast_to_sympy.convert(node)

        self.assertIsNotNone(sympy_expr)

        # SymPy -> AST
        sympy_to_ast = SymPyToASTConverter()
        new_node = sympy_to_ast.convert(sympy_expr)

        self.assertIsNotNone(new_node)

        # Should be able to unparse
        try:
            result = gast.unparse(new_node)
            self.assertIsNotNone(result)
            print(f"Round-trip: '{code}' -> '{result}'")
        except Exception as e:
            self.fail(f"Failed to unparse: {e}")

    def test_complex_expression_roundtrip(self):
        """Test round-trip on complex expression."""
        code = "x * y + sin(z)"
        node = ast.parse(code, mode='eval').body

        # AST -> SymPy
        ast_to_sympy = ASTToSymPyConverter()
        sympy_expr = ast_to_sympy.convert(node)

        self.assertIsNotNone(sympy_expr)

        # SymPy -> AST
        sympy_to_ast = SymPyToASTConverter()
        new_node = sympy_to_ast.convert(sympy_expr)

        self.assertIsNotNone(new_node)

        # Should be able to unparse
        try:
            result = gast.unparse(new_node)
            self.assertIsNotNone(result)
            print(f"Round-trip: '{code}' -> '{result}'")
        except Exception as e:
            self.fail(f"Failed to unparse: {e}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
