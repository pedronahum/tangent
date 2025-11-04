"""Test and document tuple return behavior in Tangent.

This test documents how Tangent handles functions that return tuples.
When a function returns multiple values, tangent.grad() automatically
seeds the backward pass with (1.0, 1.0, ...), effectively computing
the gradient of the sum of the outputs.

This is mathematically correct behavior for multi-output functions.
"""
import numpy as np
import tangent
from tangent import grad
import pytest


def test_scalar_return():
    """Test normal scalar return - baseline behavior."""

    def f(x):
        return x ** 2 + x * 3

    df = grad(f)
    result = df(2.0)

    # d/dx(x^2 + 3x) = 2x + 3 = 7
    assert np.isclose(result, 7.0)


def test_tuple_return_sums_gradients():
    """Document that tuple returns are auto-summed.

    When a function returns (a, b), tangent.grad() computes d/dx(a + b).
    """

    def f(x):
        return x ** 2, x * 3  # Returns tuple

    df = grad(f)
    result = df(2.0)

    # Tangent treats this as: d/dx(x^2 + 3x) = 2x + 3 = 7
    assert np.isclose(result, 7.0), \
        "Tuple returns are auto-summed: grad computes d/dx(sum of outputs)"


def test_tuple_return_mathematical_interpretation():
    """Explain the mathematical interpretation.

    For f: R -> R^2 defined as f(x) = (f1(x), f2(x)), the gradient
    with default seed (1,1) computes:  d/dx(f1(x) + f2(x))
    """

    def f(x):
        # f1(x) = x^3, f2(x) = x^2
        return x ** 3, x ** 2

    df = grad(f)
    result = df(2.0)

    # With seed (1,1): d/dx(1*x^3 + 1*x^2) = 3x^2 + 2x = 3*4 + 2*2 = 16
    expected = 3 * (2.0 ** 2) + 2 * 2.0
    assert np.isclose(result, expected), \
        f"Expected {expected}, got {result}"


def test_triple_return():
    """Test with three return values."""

    def f(x):
        return x, x ** 2, x ** 3

    df = grad(f)
    result = df(2.0)

    # d/dx(x + x^2 + x^3) = 1 + 2x + 3x^2 = 1 + 4 + 12 = 17
    expected = 1 + 2 * 2.0 + 3 * (2.0 ** 2)
    assert np.isclose(result, expected)


def test_weighted_tuple_return():
    """Show that different output magnitudes affect the gradient."""

    def f(x):
        # One large output, one small output
        return 1000 * x, 0.001 * x ** 2

    df = grad(f)
    result = df(2.0)

    # d/dx(1000*x + 0.001*x^2) = 1000 + 0.002*x = 1000.004
    expected = 1000 + 0.002 * 2.0
    assert np.isclose(result, expected)


def test_comparison_with_explicit_sum():
    """Show that tuple return is equivalent to explicitly summing."""

    def f_tuple(x):
        return x ** 2, x * 3

    def f_explicit_sum(x):
        a = x ** 2
        b = x * 3
        return a + b

    df_tuple = grad(f_tuple)
    df_explicit = grad(f_explicit_sum)

    x = 2.0
    result_tuple = df_tuple(x)
    result_explicit = df_explicit(x)

    assert np.isclose(result_tuple, result_explicit), \
        "Tuple return should give same gradient as explicit sum"


def test_tuple_with_arrays():
    """Test tuple returns with array inputs."""

    def f(x):
        return np.sum(x ** 2), np.sum(x * 3)

    x = np.array([1.0, 2.0, 3.0])
    df = grad(f)
    result = df(x)

    # d/dx[i](sum(x^2) + sum(3x)) = 2*x[i] + 3
    expected = 2 * x + 3
    assert np.allclose(result, expected)


def test_why_this_behavior_makes_sense():
    """Explain why this is the right default behavior.

    For scalar-valued loss functions in ML, we typically sum or average
    multiple outputs before computing gradients. This automatic summing
    matches that common pattern.

    Example: Loss = MSE + Regularization
    """

    def loss(params):
        mse = params ** 2  # Simulated MSE
        reg = 0.01 * params ** 2  # L2 regularization
        return mse, reg

    df = grad(loss)
    result = df(10.0)

    # Total gradient: d/dparam(mse + reg) = d/dparam(1.01 * param^2) = 2.02 * param
    expected = 2.02 * 10.0
    assert np.isclose(result, expected)


def test_how_to_get_individual_gradients():
    """Show the correct way to get individual gradients.

    If you need separate gradients, define separate functions.
    """

    def f1(x):
        return x ** 2

    def f2(x):
        return x * 3

    df1 = grad(f1)
    df2 = grad(f2)

    x = 2.0
    grad1 = df1(x)  # = 2x = 4
    grad2 = df2(x)  # = 3

    assert np.isclose(grad1, 4.0)
    assert np.isclose(grad2, 3.0)

    # If you returned (f1, f2), you'd get grad1 + grad2 = 7
    def f_combined(x):
        return x ** 2, x * 3

    df_combined = grad(f_combined)
    grad_combined = df_combined(x)

    assert np.isclose(grad_combined, grad1 + grad2)


# Def test_vjp_allows_custom_seeds():
#     """Show that vjp() allows custom gradient seeds for multi-output functions."""

#     def f(x):
#         return x ** 2, x * 3

#     # Use vjp to specify custom gradient seeds
#     from tangent import vjp
#     df = vjp(f)

#     x = 2.0
#     # Seed with (2.0, 0.0) to get gradient of just 2*f1
#     result = df(x, (2.0, 0.0))

#     # d/dx(2.0 * x^2) = 4x = 8
#     expected = 4.0 * 2.0
#     assert np.isclose(result, expected), \
#         "vjp() should allow custom gradient seeds"


def test_documentation_example():
    """Example for documentation: What users should know."""

    # ❌ Common mistake: Expecting separate gradients
    def model(x):
        prediction = x ** 2
        confidence = x * 0.5
        return prediction, confidence

    # This computes d/dx(prediction + confidence)
    d_model = grad(model)
    gradient = d_model(3.0)

    # gradient = d/dx(x^2 + 0.5x) = 2x + 0.5 = 6.5
    assert np.isclose(gradient, 6.5)

    # ✅ Correct: Sum explicitly if that's your intent
    def combined_loss(x):
        prediction = x ** 2
        confidence = x * 0.5
        return prediction + confidence  # Explicit sum

    d_loss = grad(combined_loss)
    gradient2 = d_loss(3.0)

    assert np.isclose(gradient, gradient2)

    # ✅ Or define separate functions for separate gradients
    def just_prediction(x):
        return x ** 2

    d_prediction = grad(just_prediction)
    pred_gradient = d_prediction(3.0)  # = 2*3 = 6

    assert np.isclose(pred_gradient, 6.0)


if __name__ == '__main__':
    print("=" * 80)
    print("TUPLE RETURN BEHAVIOR IN TANGENT")
    print("=" * 80)
    print()
    print("Key Finding: When a function returns a tuple (a, b, c, ...):")
    print("  tangent.grad() computes: d/dx(a + b + c + ...)")
    print()
    print("This is mathematically correct for multi-output functions,")
    print("where the default gradient seed is (1, 1, 1, ...).")
    print()
    print("=" * 80)
    print()

    # Run all tests
    test_scalar_return()
    print("✓ test_scalar_return")

    test_tuple_return_sums_gradients()
    print("✓ test_tuple_return_sums_gradients")

    test_tuple_return_mathematical_interpretation()
    print("✓ test_tuple_return_mathematical_interpretation")

    test_triple_return()
    print("✓ test_triple_return")

    test_weighted_tuple_return()
    print("✓ test_weighted_tuple_return")

    test_comparison_with_explicit_sum()
    print("✓ test_comparison_with_explicit_sum")

    test_tuple_with_arrays()
    print("✓ test_tuple_with_arrays")

    test_why_this_behavior_makes_sense()
    print("✓ test_why_this_behavior_makes_sense")

    test_how_to_get_individual_gradients()
    print("✓ test_how_to_get_individual_gradients")

    test_documentation_example()
    print("✓ test_documentation_example")

    print()
    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)
