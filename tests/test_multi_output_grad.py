"""Test multiple output gradient functionality."""

import numpy as np
import tangent
from tangent import grad
import pytest


def test_output_index_first():
    """Test getting gradient of first output only."""

    def f(x):
        return x**2, x * 3

    # Gradient of first output only
    df = grad(f, output_index=0)
    result = df(2.0)

    # Expected: d/dx(x^2) = 2x = 4.0
    assert np.isclose(result, 4.0), f"Expected 4.0, got {result}"


def test_output_index_second():
    """Test getting gradient of second output only."""

    def f(x):
        return x**2, x * 3

    # Gradient of second output only
    df = grad(f, output_index=1)
    result = df(2.0)

    # Expected: d/dx(3x) = 3
    assert np.isclose(result, 3.0), f"Expected 3.0, got {result}"


def test_output_weights():
    """Test custom output weights."""

    def f(x):
        return x**2, x * 3

    # Weighted combination: 0.5*out1 + 2.0*out2
    df = grad(f, output_weights=(0.5, 2.0))
    result = df(2.0)

    # Expected: d/dx(0.5*x^2 + 2.0*3x) = 0.5*2x + 2.0*3 = x + 6 = 8.0
    expected = 0.5 * 2 * 2.0 + 2.0 * 3
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_three_outputs():
    """Test with three outputs."""

    def f(x):
        return x, x**2, x**3

    # Gradient of second output
    df = grad(f, output_index=1)
    result = df(2.0)

    # Expected: d/dx(x^2) = 2x = 4.0
    assert np.isclose(result, 4.0)


def test_with_preserve_result():
    """Test output_index with preserve_result=True."""

    def f(x):
        return x**2, x * 3

    # Gradient of first output, preserve both outputs
    df = grad(f, output_index=0, preserve_result=True)
    grad_result, outputs = df(2.0)

    # Gradient should be 4.0
    assert np.isclose(grad_result, 4.0)

    # Outputs should be (4.0, 6.0)
    assert len(outputs) == 2
    assert np.isclose(outputs[0], 4.0)  # x^2 at x=2
    assert np.isclose(outputs[1], 6.0)  # 3x at x=2


def test_loss_and_accuracy_example():
    """Real-world example: differentiate loss but not accuracy."""

    def loss_and_accuracy(params):
        # Simulated: compute loss and accuracy
        loss = params**2  # MSE-like
        accuracy = 1.0 / (1.0 + abs(params))  # Accuracy metric
        return loss, accuracy

    # Only differentiate loss (output 0)
    dloss = grad(loss_and_accuracy, output_index=0, preserve_result=True)

    params = 3.0
    grad_params, (loss_val, acc_val) = dloss(params)

    # Gradient of loss: d/dx(x^2) = 2x = 6.0
    assert np.isclose(grad_params, 6.0)

    # Loss value: 3^2 = 9.0
    assert np.isclose(loss_val, 9.0)

    # Accuracy value: 1/(1+3) = 0.25
    assert np.isclose(acc_val, 0.25)


def test_regularized_loss_example():
    """Multi-task learning with weighted outputs."""

    def model_with_reg(params):
        pred_loss = params**2
        reg_loss = 0.01 * params**2
        return pred_loss, reg_loss

    # Weighted combination: total_loss = pred_loss + 0.01*reg_loss
    # But reg_loss already has 0.01 in it, so weights should be (1.0, 1.0)
    df = grad(model_with_reg, output_weights=(1.0, 1.0))

    result = df(10.0)

    # d/dx(x^2 + 0.01*x^2) = d/dx(1.01*x^2) = 2.02*x = 20.2
    expected = 2.02 * 10.0
    assert np.isclose(result, expected)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multi_output_check_dims_false():
    """Multi-output seed must not depend on the check_dims shapes_match assert.

    Regression test: output arity used to be inferred by scanning the
    generated primal for the shapes_match assert, so check_dims=False
    silently produced a scalar seed for tuple outputs (crashing with
    "'float' object is not subscriptable" or giving wrong gradients).
    The arity is now recorded as an annotation during reverse-mode AD.
    """

    def f(x):
        return x**2, x * 3

    df = grad(f, check_dims=False)
    # Default seed: gradient of the sum of outputs, d/dx(x^2 + 3x) = 2x + 3
    assert np.isclose(df(2.0), 7.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multi_output_check_dims_false_unoptimized():
    """Same regression with the optimizer disabled."""

    def f(x):
        return x**2, x * 3

    df = grad(f, check_dims=False, optimized=False)
    assert np.isclose(df(2.0), 7.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multi_output_optimized():
    """Multi-output seed must survive optimization (DCE etc.)."""

    def f(x):
        return x**2, x * 3

    df = grad(f, optimized=True)
    assert np.isclose(df(2.0), 7.0)
    df = grad(f, check_dims=False, optimized=True)
    assert np.isclose(df(2.0), 7.0)


def test_output_index_check_dims_false():
    """output_index must build the one-hot seed even with check_dims=False."""

    def f(x):
        return x**2, x * 3

    df0 = grad(f, output_index=0, check_dims=False)
    df1 = grad(f, output_index=1, check_dims=False)
    assert np.isclose(df0(2.0), 4.0)
    assert np.isclose(df1(2.0), 3.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multi_output_corpus_function_check_dims_false():
    """A corpus multi-output function under the previously-broken configs."""
    import functions

    df = grad(functions.fn_multiple_return, check_dims=False, optimized=True)
    # fn_multiple_return(a) = (2*a, a); d/da(2a + a) = 3
    assert np.isclose(df(5.0), 3.0)
    df = grad(functions.fn_multiple_return, check_dims=False, optimized=False)
    assert np.isclose(df(5.0), 3.0)


def test_single_output_check_dims_false_still_scalar_seed():
    """A single-output function must keep its scalar 1.0 default seed."""

    def f(x):
        return x**2

    df = grad(f, check_dims=False)
    assert np.isclose(df(3.0), 6.0)


def test_output_arity_annotation_recorded():
    """The joint function carries the output arity annotation from reverse AD."""
    from tangent import annotations as anno
    from tangent import grad_util

    def f(x):
        return x**2, x * 3

    def g(x):
        return x**2

    for fn, expected in ((f, 2), (g, None)):
        node, _ = grad_util.autodiff_ast(
            fn,
            wrt=(0,),
            motion='joint',
            mode='reverse',
            preserve_result=False,
            check_dims=False,
            verbose=0,
        )
        fwdbwd = node.body[0]
        assert anno.hasanno(fwdbwd, 'output_arity')
        assert anno.getanno(fwdbwd, 'output_arity') == expected


def test_error_both_params():
    """Test that providing both output_index and output_weights raises error."""

    def f(x):
        return x**2, x * 3

    with pytest.raises(ValueError, match="Cannot specify both"):
        grad(f, output_index=0, output_weights=(1.0, 1.0))


if __name__ == '__main__':
    print("=" * 80)
    print("MULTI-OUTPUT GRADIENT TESTS")
    print("=" * 80)

    tests = [
        ("output_index_first", test_output_index_first),
        ("output_index_second", test_output_index_second),
        ("output_weights", test_output_weights),
        ("three_outputs", test_three_outputs),
        ("with_preserve_result", test_with_preserve_result),
        ("loss_and_accuracy_example", test_loss_and_accuracy_example),
        ("regularized_loss_example", test_regularized_loss_example),
        ("error_both_params", test_error_both_params),
    ]

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    print("=" * 80)
