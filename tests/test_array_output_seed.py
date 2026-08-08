"""Array-valued outputs accept the default (scalar) seed derivative.

Reverse-mode AD needs a seed derivative (cotangent) for the output. For a scalar
output the default seed ``1.0`` is unambiguous. For an array output a scalar
seed broadcasts - it is equivalent to a ones cotangent, i.e. the gradient of the
sum of the outputs. Optimized mode already accepted this (the shape assertion is
elided there); the unoptimized shape check now accepts it too, so both modes
behave the same. Genuinely mismatched *array* seeds are still rejected.
"""
import numpy as np
import pytest

import tangent


def _both_modes(f):
    return (tangent.grad(f, optimized=True),
            tangent.grad(f, optimized=False))


def test_elementwise_array_output_default_seed():
    def f(x):
        return np.tanh(x)

    x = np.array([0.5, 1.0, 1.5])
    expected = 1 - np.tanh(x) ** 2  # gradient of sum(tanh(x))
    for df in _both_modes(f):
        assert np.allclose(df(x), expected)


def test_optimized_and_unoptimized_agree():
    def f(x):
        return x * x

    x = np.array([1.0, 2.0, 3.0])
    dopt, dunopt = _both_modes(f)
    assert np.allclose(dopt(x), dunopt(x))


def test_explicit_array_seed_still_works():
    def f(x):
        return np.exp(x)

    x = np.array([0.0, 1.0])
    seed = np.array([1.0, 1.0])
    for optimized in (True, False):
        df = tangent.grad(f, optimized=optimized)
        assert np.allclose(df(x, seed), np.exp(x))


def test_mismatched_array_seed_is_rejected():
    def f(x):
        return x

    x = np.array([1.0, 2.0, 3.0])
    df = tangent.grad(f, optimized=False)
    with pytest.raises(AssertionError):
        df(x, np.ones(5))  # wrong-shaped array seed must still be caught


def test_scalar_output_unaffected():
    def f(x):
        return np.sum(x * x)

    x = np.array([1.0, 2.0, 3.0])
    for df in _both_modes(f):
        assert np.allclose(df(x), 2 * x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
