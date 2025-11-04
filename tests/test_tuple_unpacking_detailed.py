"""Detailed tests to find the tuple unpacking gradient bug."""
import numpy as np
import tangent
from tangent import grad


def test_case_1_only_use_first():
    """Use only the first element of tuple unpacking."""

    def f(x):
        a, b = x ** 2, x * 3
        return a  # Only use 'a', not 'b'

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(x^2) = 2x = 2*2 = 4
    # If it sums: might get 2x + 3 = 7 (wrong!)
    print(f"Use only first: gradient = {result}")
    print(f"Expected: 4.0 (gradient of x^2)")
    print(f"Match: {np.isclose(result, 4.0)}")

    if not np.isclose(result, 4.0):
        print(f"❌ BUG FOUND: Expected 4.0, got {result}")
        if np.isclose(result, 7.0):
            print(f"   Looks like it's summing gradients (2x + 3 = 7)")

    return result


def test_case_2_only_use_second():
    """Use only the second element of tuple unpacking."""

    def f(x):
        a, b = x ** 2, x * 3
        return b  # Only use 'b', not 'a'

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(3x) = 3
    # If it sums: might get 2x + 3 = 7 (wrong!)
    print(f"\nUse only second: gradient = {result}")
    print(f"Expected: 3.0 (gradient of 3x)")
    print(f"Match: {np.isclose(result, 3.0)}")

    if not np.isclose(result, 3.0):
        print(f"❌ BUG FOUND: Expected 3.0, got {result}")
        if np.isclose(result, 7.0):
            print(f"   Looks like it's summing gradients (2x + 3 = 7)")

    return result


def test_case_3_weighted_sum():
    """Use both but with different weights."""

    def f(x):
        a, b = x ** 2, x * 3
        return 10 * a + 1 * b  # Weight 'a' much more

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(10*x^2 + 3x) = 20x + 3 = 20*2 + 3 = 43
    print(f"\nWeighted sum (10*a + b): gradient = {result}")
    print(f"Expected: 43.0")
    print(f"Match: {np.isclose(result, 43.0)}")

    if not np.isclose(result, 43.0):
        print(f"❌ BUG FOUND: Expected 43.0, got {result}")

    return result


def test_case_4_swap_order():
    """Swap the order to see if that matters."""

    def f(x):
        b, a = x * 3, x ** 2  # Swapped order
        return a + b

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(x^2 + 3x) = 2x + 3 = 7 (same as before)
    print(f"\nSwapped order: gradient = {result}")
    print(f"Expected: 7.0")
    print(f"Match: {np.isclose(result, 7.0)}")

    return result


def test_case_5_triple_unpacking():
    """Three-way unpacking."""

    def f(x):
        a, b, c = x ** 2, x * 3, x + 1
        return a  # Only use first one

    df = grad(f)
    result = df(2.0)

    # Expected: d/dx(x^2) = 2x = 4
    # If it sums all: might get 2x + 3 + 1 = 9 (wrong!)
    print(f"\nTriple unpacking (use only first): gradient = {result}")
    print(f"Expected: 4.0")
    print(f"Match: {np.isclose(result, 4.0)}")

    if not np.isclose(result, 4.0):
        print(f"❌ BUG FOUND: Expected 4.0, got {result}")
        if np.isclose(result, 9.0):
            print(f"   Looks like it's summing all gradients")

    return result


def test_case_6_nested_tuple():
    """Nested tuple unpacking."""

    def f(x):
        # This might not even work
        (a, b), c = (x ** 2, x * 3), x + 1
        return a + c

    try:
        df = grad(f)
        result = df(2.0)

        # Expected: d/dx(x^2 + x + 1) = 2x + 1 = 5
        print(f"\nNested tuple: gradient = {result}")
        print(f"Expected: 5.0")
        print(f"Match: {np.isclose(result, 5.0)}")

        return result
    except Exception as e:
        print(f"\nNested tuple: ERROR - {e}")
        return None


def compare_with_separate():
    """Compare tuple unpacking vs separate assignments for same computation."""

    def f_tuple(x):
        a, b = x ** 2, x * 3
        return a

    def f_separate(x):
        a = x ** 2
        b = x * 3
        return a

    df_tuple = grad(f_tuple)
    df_separate = grad(f_separate)

    result_tuple = df_tuple(2.0)
    result_separate = df_separate(2.0)

    print(f"\n\nComparison:")
    print(f"Tuple unpacking:      {result_tuple}")
    print(f"Separate assignments: {result_separate}")
    print(f"Match: {np.isclose(result_tuple, result_separate)}")

    if not np.isclose(result_tuple, result_separate):
        print(f"❌ CONFIRMED BUG: Tuple unpacking gives different result!")


if __name__ == '__main__':
    print("=" * 80)
    print("DETAILED TUPLE UNPACKING INVESTIGATION")
    print("=" * 80)

    test_case_1_only_use_first()
    test_case_2_only_use_second()
    test_case_3_weighted_sum()
    test_case_4_swap_order()
    test_case_5_triple_unpacking()
    test_case_6_nested_tuple()
    compare_with_separate()

    print("\n" + "=" * 80)
