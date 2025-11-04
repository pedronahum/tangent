"""Test that the warning is issued for tuple returns."""
import warnings
import tangent
from tangent.function_cache import clear_cache
import numpy as np


def test_tuple_return_issues_warning():
    """Test that a warning is issued when differentiating a tuple-returning function."""

    # Clear cache to ensure the warning is issued
    clear_cache()

    def f(x):
        return x ** 2, x * 3

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        df = tangent.grad(f)

        # Check that a warning was issued
        # Note: There may be deprecation warnings from ast module, so filter for our warning
        tuple_warnings = [warning for warning in w if "returns a tuple" in str(warning.message)]

        assert len(tuple_warnings) >= 1, f"Expected tuple warning, got {len(w)} warnings: {[str(x.message) for x in w]}"
        assert issubclass(tuple_warnings[0].category, UserWarning)
        assert "sum of all outputs" in str(tuple_warnings[0].message)

        print("Warning message:")
        print(str(tuple_warnings[0].message))

    # Verify it still works correctly
    result = df(2.0)
    assert np.isclose(result, 7.0)


def test_scalar_return_no_warning():
    """Test that no warning is issued for scalar returns."""

    # Clear cache
    clear_cache()

    def g(x):  # Use different function name to avoid conflicts
        return x ** 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        df = tangent.grad(g)

        # Filter for tuple warnings only (ignore deprecation warnings)
        tuple_warnings = [warning for warning in w if "returns a tuple" in str(warning.message)]

        # No tuple warning should be issued for scalar returns
        assert len(tuple_warnings) == 0, f"Unexpected tuple warning for scalar return"


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING TUPLE RETURN WARNING")
    print("=" * 80)
    print()

    print("Test 1: Tuple return should issue warning")
    print("-" * 80)
    test_tuple_return_issues_warning()
    print("✓ Warning issued correctly")
    print()

    print("Test 2: Scalar return should not issue warning")
    print("-" * 80)
    test_scalar_return_no_warning()
    print("✓ No warning for scalar returns")
    print()

    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)
