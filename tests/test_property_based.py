"""Property-based gradient tests: tangent vs. the finite-difference oracle.

For a representative sample of the corpus in tests/functions.py (scalar and
array functions spanning control flow, loops and subscripts), hypothesis
generates random inputs and the gradient returned by `tangent.grad` must match
the finite-difference oracle in tests/utils.py (`numeric_grad`). One test each
also covers forward mode and grad-of-grad on smooth functions.

Two safeguards keep the comparison honest without flakiness:

* Each control-flow function carries the set of its branch-boundary points;
  inputs within 1e-3 of a boundary are discarded (`assume`), because a central
  finite difference straddling a kink averages the two branch slopes while the
  exact gradient follows the taken branch.
* The finite difference is computed at two step sizes and the example is
  discarded when they disagree, filtering any remaining locally-non-smooth
  points.

Runs are deterministic by default (`derandomize=True`); set
TANGENT_HYPOTHESIS_PROFILE=fuzz for a randomized, larger-budget exploration
run.
"""
import functools
import os

import numpy as np
import pytest

pytest.importorskip('hypothesis')
from hypothesis import assume, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import functions
import tangent
import utils

settings.register_profile(
    'deterministic', derandomize=True, max_examples=25, deadline=None)
settings.register_profile(
    'fuzz', derandomize=False, max_examples=200, deadline=None)
settings.load_profile(
    os.environ.get('TANGENT_HYPOTHESIS_PROFILE', 'deterministic'))


# --- Function sample -------------------------------------------------------
# (function, branch-boundary points of the first argument). Boundaries are
# where an `if`/loop guard changes truth value, derived from each function's
# source.

SCALAR_CASES = [
    (functions.tanh, ()),
    (functions.third_pow, ()),
    (functions.nested_if, (0.0, 10.0)),          # a > 0; a < 10
    (functions.serial_if, (0.0,)),               # a > 0; then a < 0
    (functions.devilish_nested_if, (0.0, 10.0 / 3.0)),  # a > 0; 3a < 10
    (functions.multiarg_if, (0.0, 0.25, -0.5)),  # see source: a*b thresholds
    (functions.iterpower_static, ()),            # loop: a ** 8
    (functions.super_iterpower, ()),             # ANF in a loop: a ** 27
    (functions.cond_iterpower1, ()),             # guard a < 20 never binds on [-2, 2]
]

ARRAY_CASES = [
    functions.numpy_sum,
    functions.numpy_mean,
    functions.test_subscript1,          # reads x[0], x[1]
    functions.test_subscript2,          # writes x[0] then sums
    functions.test_subscript3,          # scatter with dependent reads
    functions.test_deep_anf_list,       # list literal + subscripts
    functions.test_implicit_indexing,   # for-loop over the array
]

SMOOTH_SCALAR_FUNCS = [
    functions.tanh,
    functions.third_pow,
    functions.iterpower_static,
]

scalar_inputs = st.floats(min_value=-2.0, max_value=2.0,
                          allow_nan=False, allow_infinity=False)
array_inputs = hnp.arrays(dtype=np.float64,
                          shape=st.integers(min_value=3, max_value=5),
                          elements=scalar_inputs)


def _away_from(value, boundaries, margin=1e-3):
  return all(abs(value - b) > margin for b in boundaries)


def _stable_fd_grad(fn, *args):
  """Finite-difference gradient, or None at a locally-non-smooth point."""
  functions.np = np  # other tests swap the corpus module's np for autograd's
  g_small = utils.numeric_grad(fn, eps=1e-6)(*args)
  g_big = utils.numeric_grad(fn, eps=1e-4)(*args)
  if not np.allclose(g_small, g_big, rtol=1e-2, atol=1e-4):
    return None
  return g_small


@functools.lru_cache(maxsize=None)
def _reverse_grad(fn):
  functions.np = np
  return tangent.grad(fn)


@functools.lru_cache(maxsize=None)
def _forward_diff(fn):
  functions.np = np
  return tangent.autodiff(fn, mode='forward', wrt=(0,),
                          preserve_result=False, optimized=True)


@functools.lru_cache(maxsize=None)
def _gradgrad(fn):
  functions.np = np
  return tangent.grad(_reverse_grad(fn))


def _check_matches_fd(dfn, fn, *args):
  fd = _stable_fd_grad(fn, *args)
  assume(fd is not None)
  got = dfn(*args)
  assert np.allclose(got, fd, rtol=1e-3, atol=1e-5), (
      '%s: tangent gradient %r != finite differences %r at %r'
      % (fn.__name__, got, fd, args))


# --- Properties ------------------------------------------------------------


@pytest.mark.parametrize('fn,boundaries', SCALAR_CASES,
                         ids=[f.__name__ for f, _ in SCALAR_CASES])
@given(a0=scalar_inputs)
def test_scalar_grad_matches_finite_differences(fn, boundaries, a0):
  assume(_away_from(a0, boundaries))
  _check_matches_fd(_reverse_grad(fn), fn, a0)


@pytest.mark.parametrize('fn', ARRAY_CASES,
                         ids=[f.__name__ for f in ARRAY_CASES])
@given(xs=array_inputs)
def test_array_grad_matches_finite_differences(fn, xs):
  _check_matches_fd(_reverse_grad(fn), fn, xs)


@given(a0=scalar_inputs, b0=scalar_inputs, c0=scalar_inputs)
def test_saxpy_grad_matches_finite_differences(a0, b0, c0):
  # Multi-argument function; gradient is taken w.r.t. the first argument.
  _check_matches_fd(_reverse_grad(functions.saxpy), functions.saxpy,
                    a0, b0, c0)


@given(a0=scalar_inputs, b0=scalar_inputs)
def test_multivar_if_grad_matches_finite_differences(a0, b0):
  assume(abs(a0 - b0) > 1e-3)  # branch boundary: a > b
  _check_matches_fd(_reverse_grad(functions.multivar_if),
                    functions.multivar_if, a0, b0)


@pytest.mark.parametrize('fn', SMOOTH_SCALAR_FUNCS,
                         ids=[f.__name__ for f in SMOOTH_SCALAR_FUNCS])
@given(a0=scalar_inputs)
def test_forward_mode_matches_finite_differences(fn, a0):
  fd = _stable_fd_grad(fn, a0)
  assume(fd is not None)
  got = _forward_diff(fn)(a0, 1.0)  # unit seed
  assert np.allclose(got, fd, rtol=1e-3, atol=1e-5), (
      '%s: forward-mode derivative %r != finite differences %r at %r'
      % (fn.__name__, got, fd, a0))


@pytest.mark.parametrize('fn', SMOOTH_SCALAR_FUNCS,
                         ids=[f.__name__ for f in SMOOTH_SCALAR_FUNCS])
@given(a0=scalar_inputs)
def test_grad_of_grad_matches_finite_differences_of_grad(fn, a0):
  # The first-derivative property test validates _reverse_grad(fn) against
  # finite differences; here the second derivative must match finite
  # differences of that (compiled, exact) first derivative.
  df = _reverse_grad(fn)
  fd = _stable_fd_grad(df, a0)
  assume(fd is not None)
  got = _gradgrad(fn)(a0)
  assert np.allclose(got, fd, rtol=1e-3, atol=1e-5), (
      '%s: grad-of-grad %r != finite differences of grad %r at %r'
      % (fn.__name__, got, fd, a0))
