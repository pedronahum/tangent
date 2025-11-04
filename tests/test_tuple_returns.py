"""Test tuple returns and multi-argument gradients."""
import numpy as np
import tangent
from tangent import grad


def test_return_tuple():
    """Test returning a tuple from a function."""

    def f(x):
        a = x ** 2
        b = x * 3
        return a, b  # Return tuple

    try:
        df = grad(f)
        result = df(2.0)
        print(f"Return tuple gradient: {result}")
        print(f"Type: {type(result)}")
        return result
    except Exception as e:
        print(f"❌ Return tuple ERROR: {e}")
        return None


def test_multi_argument_gradient():
    """Test gradient with respect to multiple arguments."""

    def f(x, y):
        return x ** 2 + y ** 2

    try:
        # Try to get gradient wrt both x and y
        df = grad(f, wrt=(0, 1))
        result = df(2.0, 3.0)
        print(f"\nMulti-argument gradient: {result}")
        print(f"Type: {type(result)}")
        print(f"Expected: (4.0, 6.0) or similar structure")
        return result
    except Exception as e:
        print(f"\n❌ Multi-argument gradient ERROR: {e}")
        return None


def test_unpack_function_return():
    """Test unpacking a function that returns a tuple."""

    def g(x):
        return x ** 2, x * 3

    def f(x):
        # Unpack the return value of another function
        a, b = g(x)
        return a + b

    try:
        df = grad(f)
        result = df(2.0)
        print(f"\nUnpack function return gradient: {result}")
        print(f"Expected: 7.0")
        print(f"Match: {np.isclose(result, 7.0)}")

        if not np.isclose(result, 7.0):
            print(f"❌ BUG: Expected 7.0, got {result}")

        return result
    except Exception as e:
        print(f"\n❌ Unpack function return ERROR: {e}")
        return None


def test_tuple_in_tuple_unpacking():
    """Test unpacking where one side is a tuple expression."""

    def f(x):
        a, b = (x ** 2, x * 3)  # Explicit tuple on RHS
        return a + b

    try:
        df = grad(f)
        result = df(2.0)
        print(f"\nExplicit tuple RHS gradient: {result}")
        print(f"Expected: 7.0")
        print(f"Match: {np.isclose(result, 7.0)}")
        return result
    except Exception as e:
        print(f"\n❌ Explicit tuple RHS ERROR: {e}")
        return None


def test_multiple_return_gradient():
    """Test if we can differentiate a function that returns multiple values."""

    def f(x):
        return x ** 2, x * 3  # Return two values

    try:
        # What happens if we try to differentiate this?
        df = grad(f)
        result = df(2.0)
        print(f"\nMultiple return gradient: {result}")
        print(f"Type: {type(result)}")

        # If it works, what does it compute?
        # Maybe it sums the outputs? d/dx(x^2 + 3x) = 2x + 3 = 7?
        if isinstance(result, tuple):
            print(f"  Returns tuple of gradients: {result}")
        else:
            print(f"  Returns single value: {result}")
            if np.isclose(result, 7.0):
                print(f"  → Appears to sum the outputs!")

        return result
    except Exception as e:
        print(f"\n❌ Multiple return gradient ERROR: {e}")
        return None


if __name__ == '__main__':
    print("=" * 80)
    print("TUPLE RETURNS AND MULTI-ARGUMENT GRADIENTS")
    print("=" * 80)

    test_return_tuple()
    test_multi_argument_gradient()
    test_unpack_function_return()
    test_tuple_in_tuple_unpacking()
    test_multiple_return_gradient()

    print("\n" + "=" * 80)
