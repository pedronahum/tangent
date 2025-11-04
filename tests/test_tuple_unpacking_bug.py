"""Test to demonstrate and fix the tuple unpacking gradient bug."""
import numpy as np
import tangent
from tangent import grad


def test_tuple_unpacking_current_behavior():
    """Demonstrate the current (incorrect) behavior of tuple unpacking."""

    def f(x):
        # Tuple unpacking - currently gives incorrect gradient
        a, b = x ** 2, x * 3
        return a + b

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(x^2 + 3x) = 2x + 3 = 2*2 + 3 = 7
    # Current (wrong): Sums gradients somehow
    print(f"Tuple unpacking gradient at x=2: {result}")
    print(f"Expected: 7.0")
    print(f"Match: {np.isclose(result, 7.0)}")

    return result


def test_separate_assignments_correct_behavior():
    """Show that separate assignments work correctly."""

    def f(x):
        # Separate assignments - works correctly
        a = x ** 2
        b = x * 3
        return a + b

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(x^2 + 3x) = 2x + 3 = 2*2 + 3 = 7
    print(f"\nSeparate assignments gradient at x=2: {result}")
    print(f"Expected: 7.0")
    print(f"Match: {np.isclose(result, 7.0)}")

    assert np.isclose(result, 7.0), f"Expected 7.0, got {result}"
    return result


def test_tuple_unpacking_multiple_uses():
    """Test tuple unpacking where both values are used."""

    def f(x):
        a, b = x ** 2, x * 3
        return a * 2 + b * 5

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(2*x^2 + 5*3x) = 4x + 15 = 4*2 + 15 = 23
    print(f"\nMultiple uses gradient at x=2: {result}")
    print(f"Expected: 23.0")
    print(f"Match: {np.isclose(result, 23.0)}")

    return result


def test_tuple_unpacking_with_array():
    """Test tuple unpacking with arrays."""

    def f(x):
        a, b = x ** 2, x * 3
        return np.sum(a) + np.sum(b)

    x = np.array([1.0, 2.0])
    df = grad(f)
    result = df(x)

    # Expected: d/dx[0](sum(x^2) + sum(3x)) = 2*x[0] + 3 = 2*1 + 3 = 5
    # Expected: d/dx[1](sum(x^2) + sum(3x)) = 2*x[1] + 3 = 2*2 + 3 = 7
    print(f"\nArray tuple unpacking gradient at x=[1, 2]: {result}")
    print(f"Expected: [5.0, 7.0]")
    expected = np.array([5.0, 7.0])
    print(f"Match: {np.allclose(result, expected)}")

    return result


if __name__ == '__main__':
    print("=" * 80)
    print("TUPLE UNPACKING GRADIENT BUG INVESTIGATION")
    print("=" * 80)

    # Test 1: Current behavior
    print("\n1. Current (incorrect) tuple unpacking behavior:")
    print("-" * 80)
    try:
        test_tuple_unpacking_current_behavior()
    except Exception as e:
        print(f"ERROR: {e}")

    # Test 2: Correct behavior with separate assignments
    print("\n2. Correct behavior with separate assignments:")
    print("-" * 80)
    try:
        test_separate_assignments_correct_behavior()
    except Exception as e:
        print(f"ERROR: {e}")

    # Test 3: Multiple uses
    print("\n3. Tuple unpacking with multiple uses:")
    print("-" * 80)
    try:
        test_tuple_unpacking_multiple_uses()
    except Exception as e:
        print(f"ERROR: {e}")

    # Test 4: Arrays
    print("\n4. Tuple unpacking with arrays:")
    print("-" * 80)
    try:
        test_tuple_unpacking_with_array()
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "=" * 80)
