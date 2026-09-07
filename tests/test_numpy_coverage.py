# Copyright 2026 Tangent contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NumPy gradient coverage: ufunc arithmetic, math, shape ops and linalg.

Every op exercised here was previously unimplemented (raising
ReverseNotImplementedError / ForwardNotImplementedError or failing to
resolve). Each gradient is validated against the central finite-difference
oracle from tests/utils.py (`numeric_grad`), which computes the gradient of
sum(f) - so all test functions reduce to a scalar with weights that make
the gradients non-trivial.

Reverse mode compares the full gradient; forward mode compares the
directional derivative along an all-ones direction (seed 1.0), which equals
the sum of the gradient entries.
"""
import numpy as np
import pytest

import tangent
from utils import numeric_grad


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _fd_grad(f, args, i):
  """Finite-difference gradient of scalar-valued f w.r.t. args[i]."""

  def reordered(x, *rest):
    full = list(rest[:i]) + [x] + list(rest[i:])
    return f(*full)

  rest = args[:i] + args[i + 1:]
  return numeric_grad(reordered)(args[i], *rest)


def check_reverse(f, args, wrt=(0,), tol=1e-4):
  """Reverse-mode gradient of scalar-valued f vs. finite differences."""
  for i in wrt:
    df = tangent.grad(f, wrt=(i,))
    got = df(*args)
    expected = _fd_grad(f, args, i)
    assert np.allclose(got, expected, atol=tol, rtol=tol), (
        'reverse wrt arg %d: got %s expected %s' % (i, got, expected))


def check_forward(f, args, wrt=(0,), tol=1e-4):
  """Forward-mode directional derivative (all-ones direction) vs. FD."""
  for i in wrt:
    df = tangent.autodiff(
        f, mode='forward', wrt=(i,), preserve_result=False)
    got = df(*(args + (1.0,)))
    expected = np.sum(_fd_grad(f, args, i))
    assert np.allclose(got, expected, atol=tol, rtol=tol), (
        'forward wrt arg %d: got %s expected %s' % (i, got, expected))


def check_both(f, args, wrt=(0,), tol=1e-4):
  check_reverse(f, args, wrt, tol)
  check_forward(f, args, wrt, tol)


# Shared inputs. Weights make the gradients position-dependent so that
# permutation/replication ops cannot pass by accident.
V1 = np.array([1.0, 2.0, 3.0])
V2 = np.array([0.5, 1.5, 2.5])
W3 = np.array([0.7, -1.3, 2.1])
M22 = np.array([[1.0, 2.0], [3.0, 4.0]])
M23 = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
W23 = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]])


# ---------------------------------------------------------------------------
# 1. Ufunc spellings of basic arithmetic
# ---------------------------------------------------------------------------


def add_ufunc(a, b):
  return np.sum(np.add(a, b) * np.array([1.0, -2.0, 3.0]))


def add_broadcast(a, b):
  return np.sum(np.add(a, b) * np.array([[0.5, 1.5, -0.5],
                                         [2.0, -1.0, 0.25]]))


def subtract_ufunc(a, b):
  return np.sum(np.subtract(a, b) * np.array([1.0, -2.0, 3.0]))


def subtract_broadcast(a, b):
  return np.sum(np.subtract(a, b) * np.array([[0.5, 1.5, -0.5],
                                              [2.0, -1.0, 0.25]]))


def divide_ufunc(a, b):
  return np.sum(np.divide(a, b) * np.array([1.0, -2.0, 3.0]))


def divide_broadcast(a, b):
  return np.sum(np.divide(a, b) * np.array([[0.5, 1.5, -0.5],
                                            [2.0, -1.0, 0.25]]))


def true_divide_ufunc(a, b):
  return np.sum(np.true_divide(a, b) * np.array([1.0, -2.0, 3.0]))


def negative_ufunc(a):
  return np.sum(np.negative(a) * np.array([1.0, -2.0, 3.0]))


def power_ufunc(a, b):
  return np.sum(np.power(a, b))


def power_scalar_exponent(a):
  return np.sum(np.power(a, 2.0))


def float_power_ufunc(a, b):
  return np.sum(np.float_power(a, b))


def multiply_broadcast(a, b):
  return np.sum(np.multiply(a, b) * np.array([[0.5, 1.5, -0.5],
                                              [2.0, -1.0, 0.25]]))


def test_add_ufunc():
  check_both(add_ufunc, (V1, V2), wrt=(0, 1))


def test_add_broadcast():
  # (2, 3) + (3,): the gradient w.r.t. b must be unbroadcast (summed).
  check_both(add_broadcast, (M23, V1), wrt=(0, 1))


def test_add_scalar_broadcast():
  check_reverse(add_ufunc, (V1, 2.0), wrt=(1,))


def test_subtract_ufunc():
  check_both(subtract_ufunc, (V1, V2), wrt=(0, 1))


def test_subtract_broadcast():
  check_both(subtract_broadcast, (M23, V1), wrt=(0, 1))


def test_divide_ufunc():
  check_both(divide_ufunc, (V1, V2), wrt=(0, 1))


def test_divide_broadcast():
  check_both(divide_broadcast, (M23, V1), wrt=(0, 1))


def test_true_divide_ufunc():
  check_both(true_divide_ufunc, (V1, V2), wrt=(0, 1))


def test_negative_ufunc():
  check_both(negative_ufunc, (V1,))


def test_power_ufunc():
  # Positive bases so the exponent gradient (log term) is well-defined.
  check_both(power_ufunc, (V1, V2), wrt=(0, 1))


def test_power_scalar_exponent_negative_base():
  # x ** 2 with negative entries: the log term must not poison the
  # gradient w.r.t. the base when the exponent is inactive.
  x = np.array([-2.0, -0.5, 1.5])
  check_both(power_scalar_exponent, (x,))


def test_float_power_ufunc():
  check_both(float_power_ufunc, (V1, V2), wrt=(0, 1))


def test_multiply_broadcast():
  # np.multiply's adjoint now unbroadcasts like the * operator does.
  check_both(multiply_broadcast, (M23, V1), wrt=(0, 1))


# ---------------------------------------------------------------------------
# 2. Common math functions
# ---------------------------------------------------------------------------


def arctan2_f(a, b):
  return np.sum(np.arctan2(a, b) * np.array([1.0, -2.0, 3.0]))


def arctan2_negative_quadrant(a, b):
  return np.sum(np.arctan2(a, b))


def hypot_f(a, b):
  return np.sum(np.hypot(a, b) * np.array([1.0, -2.0, 3.0]))


def logaddexp_f(a, b):
  return np.sum(np.logaddexp(a, b) * np.array([1.0, -2.0, 3.0]))


def arcsinh_f(a):
  return np.sum(np.arcsinh(a) * np.array([1.0, -2.0, 3.0]))


def arccosh_f(a):
  return np.sum(np.arccosh(a) * np.array([1.0, -2.0, 3.0]))


def arctanh_f(a):
  return np.sum(np.arctanh(a) * np.array([1.0, -2.0, 3.0]))


def exp2_f(a):
  return np.sum(np.exp2(a) * np.array([1.0, -2.0, 3.0]))


def cbrt_f(a):
  return np.sum(np.cbrt(a) * np.array([1.0, -2.0, 3.0]))


def square_f(a):
  return np.sum(np.square(a) * np.array([1.0, -2.0, 3.0]))


def fmax_f(a, b):
  return np.sum(np.fmax(a, b) * np.array([1.0, -2.0, 3.0]))


def fmin_f(a, b):
  return np.sum(np.fmin(a, b) * np.array([1.0, -2.0, 3.0]))


def fmax_broadcast(a, b):
  return np.sum(np.fmax(a, b) * np.array([[0.5, 1.5, -0.5],
                                          [2.0, -1.0, 0.25]]))


def test_arctan2():
  check_both(arctan2_f, (V1, V2), wrt=(0, 1))


def test_arctan2_negative_quadrant():
  a = np.array([-1.0, 2.0, -0.5])
  b = np.array([-2.0, -1.0, 3.0])
  check_both(arctan2_negative_quadrant, (a, b), wrt=(0, 1))


def test_hypot():
  check_both(hypot_f, (V1, V2), wrt=(0, 1))


def test_logaddexp():
  check_both(logaddexp_f, (V1, V2), wrt=(0, 1))


def test_arcsinh():
  check_both(arcsinh_f, (np.array([-1.5, 0.3, 2.0]),))


def test_arccosh():
  check_both(arccosh_f, (np.array([1.5, 2.0, 3.0]),))


def test_arctanh():
  check_both(arctanh_f, (np.array([-0.7, 0.1, 0.6]),))


def test_exp2():
  check_both(exp2_f, (np.array([-1.0, 0.5, 2.0]),))


def test_cbrt():
  # Includes a negative input: cbrt is defined there and so is its gradient.
  check_both(cbrt_f, (np.array([-8.0, 0.5, 2.0]),))


def test_square_forward():
  # The adjoint already existed in numpy_extended; the tangent is new.
  check_both(square_f, (np.array([-1.5, 0.3, 2.0]),))


def test_fmax():
  # Tie-free inputs (finite differences straddle the kink at ties).
  check_both(fmax_f, (np.array([1.0, 3.0, 2.0]),
                      np.array([2.0, 1.0, 5.0])), wrt=(0, 1))


def test_fmin():
  check_both(fmin_f, (np.array([1.0, 3.0, 2.0]),
                      np.array([2.0, 1.0, 5.0])), wrt=(0, 1))


def test_fmax_broadcast():
  check_reverse(fmax_broadcast, (M23, np.array([0.5, 4.0, -1.0])),
                wrt=(0, 1))


def test_maximum_minimum_forward():
  # np.maximum/np.minimum adjoints already existed; the tangents are new.
  check_forward(fmax_like_maximum, (np.array([1.0, 3.0, 2.0]),
                                    np.array([2.0, 1.0, 5.0])), wrt=(0, 1))
  check_forward(fmin_like_minimum, (np.array([1.0, 3.0, 2.0]),
                                    np.array([2.0, 1.0, 5.0])), wrt=(0, 1))


def fmax_like_maximum(a, b):
  return np.sum(np.maximum(a, b) * np.array([1.0, -2.0, 3.0]))


def fmin_like_minimum(a, b):
  return np.sum(np.minimum(a, b) * np.array([1.0, -2.0, 3.0]))


# ---------------------------------------------------------------------------
# 3. Shape and reduction operations
# ---------------------------------------------------------------------------


def cumsum_flat(a, w):
  return np.sum(np.cumsum(a) * w)


def cumsum_flat_matrix(a, w):
  return np.sum(np.cumsum(a) * w)


def cumsum_axis0(a, w):
  return np.sum(np.cumsum(a, axis=0) * w)


def cumsum_axis1(a, w):
  return np.sum(np.cumsum(a, axis=1) * w)


def flip_all(a, w):
  return np.sum(np.flip(a) * w)


def flip_axis0(a, w):
  return np.sum(np.flip(a, 0) * w)


def flip_axis1(a, w):
  return np.sum(np.flip(a, axis=1) * w)


def ravel_f(a, w):
  return np.sum(np.ravel(a) * w)


def swapaxes_f(a, w):
  return np.sum(np.swapaxes(a, 0, 1) * w)


def moveaxis_f(a, w):
  return np.sum(np.moveaxis(a, 0, 2) * w)


def tile_vector(a, w):
  return np.sum(np.tile(a, 3) * w)


def tile_matrix(a, w):
  return np.sum(np.tile(a, (2, 3)) * w)


def tile_promoting(a, w):
  # reps has more dimensions than the input: numpy promotes the input.
  return np.sum(np.tile(a, (2, 2)) * w)


def repeat_flat(a, w):
  return np.sum(np.repeat(a, 2) * w)


def repeat_axis(a, w):
  return np.sum(np.repeat(a, 3, axis=1) * w)


def repeat_array_repeats(a, w):
  return np.sum(np.repeat(a, np.array([1, 2, 3]), axis=0) * w)


def roll_scalar(a, w):
  return np.sum(np.roll(a, 1) * w)


def roll_axis(a, w):
  return np.sum(np.roll(a, 2, axis=1) * w)


def roll_tuple(a, w):
  return np.sum(np.roll(a, (1, 2), axis=(0, 1)) * w)


def test_cumsum_flat():
  check_both(cumsum_flat, (V1, W3))


def test_cumsum_flat_matrix():
  # axis=None flattens; the gradient must be reshaped back to the input.
  check_both(cumsum_flat_matrix, (M23, np.arange(6.0) - 2.5))


def test_cumsum_axis0():
  check_both(cumsum_axis0, (M23, W23))


def test_cumsum_axis1():
  check_both(cumsum_axis1, (M23, W23))


def test_flip_all():
  check_both(flip_all, (M23, W23))


def test_flip_axis0():
  check_both(flip_axis0, (M23, W23))


def test_flip_axis1():
  check_both(flip_axis1, (M23, W23))


def test_ravel():
  check_both(ravel_f, (M23, np.arange(6.0) - 2.5))


def test_swapaxes():
  check_both(swapaxes_f, (M23, W23.T))


def test_moveaxis():
  a = np.arange(24.0).reshape(2, 3, 4)
  w = np.linspace(-1.0, 1.0, 24).reshape(3, 4, 2)
  check_both(moveaxis_f, (a, w))


def test_tile_vector():
  check_both(tile_vector, (V1, np.linspace(-1.0, 1.0, 9)))


def test_tile_matrix():
  check_both(tile_matrix, (M22, np.linspace(-1.0, 1.0, 24).reshape(4, 6)))


def test_tile_promoting():
  check_both(tile_promoting, (V1, np.linspace(-1.0, 1.0, 12).reshape(2, 6)))


def test_repeat_flat():
  check_both(repeat_flat, (M23, np.linspace(-1.0, 1.0, 12)))


def test_repeat_axis():
  check_both(repeat_axis, (M23, np.linspace(-1.0, 1.0, 18).reshape(2, 9)))


def test_repeat_array_repeats():
  check_reverse(repeat_array_repeats,
                (V1.reshape(3, 1), np.linspace(-1.0, 1.0, 6).reshape(6, 1)))


def test_roll_scalar():
  check_both(roll_scalar, (V1, W3))


def test_roll_axis():
  check_both(roll_axis, (M23, W23))


def test_roll_tuple():
  check_both(roll_tuple, (M23, W23))


# ---------------------------------------------------------------------------
# 4. Linear algebra: solve and norm (default 2-norm / Frobenius)
# ---------------------------------------------------------------------------


def solve_vector(a, b, w):
  return np.sum(np.linalg.solve(a, b) * w)


def solve_matrix(a, b, w):
  return np.sum(np.linalg.solve(a, b) * w)


def norm_vector(a):
  return np.linalg.norm(a)


def norm_matrix(a):
  return np.linalg.norm(a)


def norm_axis(a, w):
  return np.sum(np.linalg.norm(a, axis=0) * w)


def norm_keepdims(a, w):
  return np.sum(np.linalg.norm(a, axis=1, keepdims=True) * w)


A_SPD = np.array([[3.0, 1.0], [1.0, 2.0]])
A_GEN = np.array([[2.0, -1.0], [0.5, 3.0]])  # non-symmetric


def test_solve_vector():
  check_both(solve_vector, (A_GEN, np.array([1.0, 2.0]),
                            np.array([1.0, -2.0])), wrt=(0, 1))


def test_solve_vector_spd():
  check_both(solve_vector, (A_SPD, np.array([1.0, 2.0]),
                            np.array([1.0, -2.0])), wrt=(0, 1))


def test_solve_matrix_rhs():
  b = np.array([[1.0, 0.5], [2.0, -1.0]])
  w = np.array([[1.0, -2.0], [0.5, 1.5]])
  check_both(solve_matrix, (A_GEN, b, w), wrt=(0, 1))


def test_norm_vector():
  check_both(norm_vector, (V1,))


def test_norm_matrix_frobenius():
  check_both(norm_matrix, (M23,))


def test_norm_axis():
  check_both(norm_axis, (M23, W3))


def test_norm_keepdims():
  check_both(norm_keepdims, (M23, np.array([[1.0], [-2.0]])))


# ---------------------------------------------------------------------------
# Bookkeeping: the newly implemented ops are out of the UNIMPLEMENTED sets
# and registered in the template dictionaries.
# ---------------------------------------------------------------------------

NEW_OPS = [
    np.add, np.subtract, np.divide, np.true_divide, np.negative, np.power,
    np.float_power, np.arctan2, np.hypot, np.logaddexp, np.arcsinh,
    np.arccosh, np.arctanh, np.exp2, np.cbrt, np.fmax, np.fmin, np.cumsum,
    np.flip, np.ravel, np.swapaxes, np.moveaxis, np.tile, np.repeat, np.roll,
    np.linalg.solve, np.linalg.norm,
]


def test_new_ops_have_adjoints_registered():
  from tangent import grads
  for op in NEW_OPS:
    assert op in grads.adjoints, op
    assert op not in grads.UNIMPLEMENTED_ADJOINTS, op


def test_new_ops_have_tangents_registered():
  from tangent import tangents
  for op in NEW_OPS + [np.square, np.maximum, np.minimum, np.reshape]:
    assert op in tangents.tangents, op
    assert op not in tangents.UNIMPLEMENTED_TANGENTS, op


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
