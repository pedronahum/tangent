"""Examples demonstrating class support in Tangent.

This file showcases various ways to use classes with Tangent's automatic
differentiation, including:
- Simple methods
- Instance attributes
- Method chaining
- NumPy integration
- Multi-parameter methods
"""
import tangent
import numpy as np


# =============================================================================
# Class Definitions (must be at module level)
# =============================================================================

class Calculator:
    def square(self, x):
        """Return x squared."""
        return x ** 2


class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor


class Polynomial:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        """Evaluate polynomial: a*x^2 + b*x + c."""
        return self.a * x ** 2 + self.b * x + self.c


class ChainedCalculator:
    def square(self, x):
        return x ** 2

    def double(self, x):
        return x * 2

    def square_then_double(self, x):
        """Call square, then double the result."""
        return self.double(self.square(x))


class NumpyCalculator:
    def sin_plus_square(self, x):
        """Compute sin(x) + x^2."""
        return np.sin(x) + x ** 2


class MultiParameter:
    def multiply_add(self, x, y):
        """Compute x * y + x^2."""
        return x * y + x ** 2


class DenseLayer:
    """Simple dense layer."""

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        """Apply linear transformation."""
        return x * self.weight + self.bias


# =============================================================================
# Examples
# =============================================================================

def example_1_simple_method():
    """Example 1: Simple class method."""
    print("\n" + "="*70)
    print("Example 1: Simple Method")
    print("="*70)

    def f(x):
        calc = Calculator()
        return calc.square(x)

    # Gradient of x^2 is 2x
    df = tangent.grad(f)

    print("\nFunction: f(x) = x²")
    print("Gradient: f'(x) = 2x")
    print(f"\nAt x=3.0:")
    print(f"  f'(3.0) = {df(3.0)} (expected: 6.0)")
    print(f"\nAt x=5.0:")
    print(f"  f'(5.0) = {df(5.0)} (expected: 10.0)")


def example_2_instance_attributes():
    """Example 2: Instance attributes."""
    print("\n" + "="*70)
    print("Example 2: Instance Attributes")
    print("="*70)

    def f(x):
        scaler = Scaler(2.5)
        return scaler.scale(x)

    # Gradient of 2.5*x is 2.5
    df = tangent.grad(f)

    print("\nFunction: f(x) = 2.5 * x")
    print("Gradient: f'(x) = 2.5")
    print(f"\nAt x=3.0:")
    print(f"  f'(3.0) = {df(3.0)} (expected: 2.5)")
    print(f"\nAt x=10.0:")
    print(f"  f'(10.0) = {df(10.0)} (expected: 2.5)")


def example_3_polynomial():
    """Example 3: Polynomial evaluation."""
    print("\n" + "="*70)
    print("Example 3: Polynomial with Multiple Attributes")
    print("="*70)

    def f(x):
        poly = Polynomial(2.0, 3.0, 1.0)  # 2x^2 + 3x + 1
        return poly.evaluate(x)

    # Gradient of 2x^2 + 3x + 1 is 4x + 3
    df = tangent.grad(f)

    print("\nFunction: f(x) = 2x² + 3x + 1")
    print("Gradient: f'(x) = 4x + 3")
    print(f"\nAt x=2.0:")
    print(f"  f'(2.0) = {df(2.0)} (expected: 11.0)")
    print(f"\nAt x=5.0:")
    print(f"  f'(5.0) = {df(5.0)} (expected: 23.0)")


def example_4_method_chaining():
    """Example 4: Method chaining (methods calling methods)."""
    print("\n" + "="*70)
    print("Example 4: Method Chaining")
    print("="*70)

    def f(x):
        calc = ChainedCalculator()
        return calc.square_then_double(x)

    # square_then_double(x) = 2 * (x^2) = 2x^2
    # Gradient is 4x
    df = tangent.grad(f)

    print("\nFunction: f(x) = 2 * (x²) = 2x²")
    print("Gradient: f'(x) = 4x")
    print(f"\nAt x=3.0:")
    print(f"  f'(3.0) = {df(3.0)} (expected: 12.0)")
    print(f"\nAt x=5.0:")
    print(f"  f'(5.0) = {df(5.0)} (expected: 20.0)")


def example_5_numpy_integration():
    """Example 5: NumPy operations in methods."""
    print("\n" + "="*70)
    print("Example 5: NumPy Integration")
    print("="*70)

    def f(x):
        calc = NumpyCalculator()
        return calc.sin_plus_square(x)

    # Gradient of sin(x) + x^2 is cos(x) + 2x
    df = tangent.grad(f)

    x_test = 1.0
    expected = np.cos(x_test) + 2 * x_test
    result = df(x_test)

    print("\nFunction: f(x) = sin(x) + x²")
    print("Gradient: f'(x) = cos(x) + 2x")
    print(f"\nAt x={x_test}:")
    print(f"  f'({x_test}) = {result:.6f}")
    print(f"  expected = cos({x_test}) + 2*{x_test} = {expected:.6f}")
    print(f"  error = {abs(result - expected):.10f}")


def example_6_multiple_parameters():
    """Example 6: Multi-parameter methods."""
    print("\n" + "="*70)
    print("Example 6: Multi-Parameter Methods")
    print("="*70)

    def f(x, y):
        calc = MultiParameter()
        return calc.multiply_add(x, y)

    # Gradient w.r.t. x of (x*y + x^2) is y + 2x
    df_dx = tangent.grad(f, wrt=(0,))

    # Gradient w.r.t. y of (x*y + x^2) is x
    df_dy = tangent.grad(f, wrt=(1,))

    print("\nFunction: f(x, y) = x*y + x²")
    print("Gradient w.r.t. x: ∂f/∂x = y + 2x")
    print("Gradient w.r.t. y: ∂f/∂y = x")
    print(f"\nAt x=3.0, y=4.0:")
    print(f"  ∂f/∂x = {df_dx(3.0, 4.0)} (expected: 10.0)")
    print(f"  ∂f/∂y = {df_dy(3.0, 4.0)} (expected: 3.0)")


def example_7_real_world_neural_network():
    """Example 7: Simple neural network layer."""
    print("\n" + "="*70)
    print("Example 7: Neural Network Layer")
    print("="*70)

    def loss(x):
        """Compute loss for a simple network."""
        layer = DenseLayer(weight=2.0, bias=-1.0)
        output = layer.forward(x)
        # MSE loss with target 5.0
        return (output - 5.0) ** 2

    # Gradient of the loss
    # L = (2x - 1 - 5)² = (2x - 6)²
    # dL/dx = 2(2x - 6) * 2 = 4(2x - 6) = 8x - 24
    dloss = tangent.grad(loss)

    print("\nNeural Network:")
    print("  Layer: f(x) = 2.0*x - 1.0")
    print("  Loss: L = (f(x) - 5.0)² = (2x - 6)²")
    print("  Gradient: dL/dx = 8x - 24")
    print("\nTesting gradient at different points:")

    test_points = [1.0, 2.0, 3.0, 4.0]
    for x in test_points:
        grad = dloss(x)
        expected = 8*x - 24
        layer_output = 2.0*x - 1.0
        print(f"\nAt x={x}:")
        print(f"  Layer output: {layer_output}")
        print(f"  dL/dx = {grad} (expected: {expected})")


if __name__ == '__main__':
    """Run all examples."""
    print("\n" + "="*70)
    print("TANGENT CLASS SUPPORT EXAMPLES")
    print("="*70)
    print("\nDemonstrating automatic differentiation through user-defined classes")

    example_1_simple_method()
    example_2_instance_attributes()
    example_3_polynomial()
    example_4_method_chaining()
    example_5_numpy_integration()
    example_6_multiple_parameters()
    example_7_real_world_neural_network()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✅ Classes work seamlessly with Tangent")
    print("  ✅ Instance attributes are properly substituted")
    print("  ✅ Method chaining is fully supported")
    print("  ✅ NumPy operations work in methods")
    print("  ✅ Multi-parameter methods work correctly")
    print("  ✅ Real-world patterns (like neural networks) are supported")
    print()
