"""
Unit tests for Strength Reduction optimization.
"""
import unittest
import ast
import gast
from tangent.optimizations.strength_reduction import (
    StrengthReducer,
    StrengthReductionOptimizer,
    apply_strength_reduction
)


class TestPowerReduction(unittest.TestCase):
    """Test power operation reduction."""

    def test_square_reduction(self):
        """Test x**2 -> x*x."""
        code = """
def f(x):
    result = x ** 2
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**2 -> x*x")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)
        # Check that result contains multiplication
        result_stmt = optimized.body[0]
        self.assertIsInstance(result_stmt.value.op, (gast.Mult, ast.Mult))

    def test_cube_reduction(self):
        """Test x**3 -> x*x*x."""
        code = """
def f(x):
    result = x ** 3
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**3 -> x*x*x")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_fourth_power_reduction(self):
        """Test x**4 -> (x*x)*(x*x)."""
        code = """
def f(x):
    result = x ** 4
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**4 -> (x*x)*(x*x)")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("Note: CSE will further optimize x*x duplicates")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_sqrt_reduction(self):
        """Test x**0.5 -> sqrt(x)."""
        code = """
def f(x):
    result = x ** 0.5
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**0.5 -> sqrt(x)")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)
        # Check that result uses sqrt
        result_stmt = optimized.body[0]
        self.assertIsInstance(result_stmt.value, (gast.Call, ast.Call))
        self.assertEqual(result_stmt.value.func.id, 'sqrt')

    def test_reciprocal_reduction(self):
        """Test x**-1 -> 1.0/x."""
        code = """
def f(x):
    result = x ** -1
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**-1 -> 1.0/x")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_identity_power(self):
        """Test x**1 -> x."""
        code = """
def f(x):
    result = x ** 1
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**1 -> x")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_zero_power(self):
        """Test x**0 -> 1."""
        code = """
def f(x):
    result = x ** 0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("POWER REDUCTION TEST: x**0 -> 1")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)


class TestDivisionReduction(unittest.TestCase):
    """Test division to multiplication reduction."""

    def test_division_by_constant(self):
        """Test x / 2.0 -> x * 0.5."""
        code = """
def f(x):
    result = x / 2.0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("DIVISION REDUCTION TEST: x / 2.0 -> x * 0.5")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)
        # Check that result contains multiplication
        result_stmt = optimized.body[0]
        self.assertIsInstance(result_stmt.value.op, (gast.Mult, ast.Mult))

    def test_division_by_ten(self):
        """Test x / 10.0 -> x * 0.1."""
        code = """
def f(x):
    result = x / 10.0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("DIVISION REDUCTION TEST: x / 10.0 -> x * 0.1")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_no_division_by_variable(self):
        """Test that x / y is NOT reduced (y is not constant)."""
        code = """
def f(x, y):
    result = x / y
        """
        tree = gast.parse(code)
        func = tree.body[0]

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        # Should not reduce variable division
        self.assertEqual(optimizer.total_reductions, 0)


class TestCombinedReductions(unittest.TestCase):
    """Test multiple reductions in one function."""

    def test_multiple_powers(self):
        """Test multiple power reductions in one function."""
        code = """
def f(x, y):
    a = x ** 2
    b = y ** 3
    c = x ** 0.5
    return a + b + c
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("COMBINED REDUCTION TEST: Multiple powers")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        # Should reduce all three powers
        self.assertGreaterEqual(optimizer.total_reductions, 3)

    def test_powers_and_divisions(self):
        """Test combined power and division reductions."""
        code = """
def f(x):
    a = x ** 2
    b = a / 2.0
    c = x ** 3
    d = c / 10.0
    return b + d
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("COMBINED REDUCTION TEST: Powers and divisions")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        # Should reduce 2 powers + 2 divisions = 4 reductions
        self.assertGreaterEqual(optimizer.total_reductions, 4)


class TestGradientPatterns(unittest.TestCase):
    """Test strength reduction on typical gradient patterns."""

    def test_gradient_with_squares(self):
        """Test gradient code with x**2 patterns."""
        # Simulates Tangent-generated gradient with squared terms
        code = """
def grad_f(x, by):
    # Forward: y = x**2
    # Backward: bx = by * 2*x, computed as by * x + by * x
    # But with x**2 in mix
    temp = x ** 2
    bx = by * x
    bx2 = by * x
    bx_total = bx + bx2
    return bx_total
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("GRADIENT PATTERN TEST: Squared terms")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        self.assertGreater(optimizer.total_reductions, 0)

    def test_polynomial_gradient(self):
        """Test gradient of polynomial with powers."""
        code = """
def grad_polynomial(x, by):
    # Gradient of x**2 + x**3 + x**4
    term1 = 2 * x
    term2 = 3 * x ** 2
    term3 = 4 * x ** 3
    bx = by * (term1 + term2 + term3)
    return bx
        """
        tree = gast.parse(code)
        func = tree.body[0]

        print("\n" + "=" * 60)
        print("GRADIENT PATTERN TEST: Polynomial")
        print("=" * 60)
        print("BEFORE:")
        print(gast.unparse(func))

        optimizer = StrengthReductionOptimizer()
        optimized = optimizer.optimize(func)

        print("\nAFTER:")
        print(gast.unparse(optimized))
        print(f"Reductions applied: {optimizer.total_reductions}")
        print("=" * 60)

        # Should reduce x**2 and x**3
        self.assertGreaterEqual(optimizer.total_reductions, 2)


class TestConfigurationOptions(unittest.TestCase):
    """Test configuration options."""

    def test_disable_power_reduction(self):
        """Test disabling power reduction."""
        code = """
def f(x):
    result = x ** 2
        """
        tree = gast.parse(code)
        func = tree.body[0]

        optimizer = StrengthReductionOptimizer(enable_power_reduction=False)
        optimized = optimizer.optimize(func)

        # Should not reduce
        self.assertEqual(optimizer.total_reductions, 0)

    def test_disable_division_reduction(self):
        """Test disabling division reduction."""
        code = """
def f(x):
    result = x / 2.0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        optimizer = StrengthReductionOptimizer(enable_division_to_multiply=False)
        optimized = optimizer.optimize(func)

        # Should not reduce
        self.assertEqual(optimizer.total_reductions, 0)

    def test_apply_strength_reduction_function(self):
        """Test the apply_strength_reduction entry point."""
        code = """
def f(x):
    result = x ** 2 + x / 2.0
        """
        tree = gast.parse(code)
        func = tree.body[0]

        config = {
            'enable_power_reduction': True,
            'enable_division_to_multiply': True
        }
        optimized = apply_strength_reduction(func, config)

        self.assertIsNotNone(optimized)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
