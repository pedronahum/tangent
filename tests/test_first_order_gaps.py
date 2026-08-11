"""First-order reverse-mode gradients for functions excluded from the harness.

A few functions in the shared corpus are excluded from the parametrized harness
(see conftest.py) because they are not yet supported across *every* mode the
harness exercises (forward mode, reverse-over-reverse, ...). Several of them do,
however, produce correct gradients in first-order reverse mode - typically as a
result of the subscript-scatter broadcast-reduce fix. This file pins those down
so the progress is locked in and visible, even though the functions are not yet
ready for the full harness.

When one of these starts passing the full harness, move it out of the
conftest blacklist and delete its entry here.
"""
import numpy as np
import pytest

import functions
import tangent


def test_fn_multiple_return_first_order():
  # f(a) = (2a, a); the summed gradient is 2 + 1 = 3.
  assert tangent.grad(functions.fn_multiple_return)(2.0) == pytest.approx(3.0)


def test_active_subscript_first_order():
  # y[i] = x[i] elementwise; d/dx sum(y) = ones.
  grad = tangent.grad(functions.active_subscript)(np.array([1.0, 2.0, 3.0]))
  assert np.allclose(grad, [1.0, 1.0, 1.0])


def test_init_array_grad_maybe_active_first_order():
  # h[t] = x[t] broadcasts a scalar into a row of 3; each x[t] contributes 3.
  grad = tangent.grad(functions.init_array_grad_maybe_active)(
      np.array([1.0, 2.0, 3.0]))
  assert np.allclose(grad, [3.0, 3.0, 3.0])


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
