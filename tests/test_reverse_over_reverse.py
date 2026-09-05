# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Tests for reverse-over-reverse automatic differentiation.

Notes
-----
Arguments func, a, b, c, x, and n are automatically filled in.

Pass --short for a quick run.

"""
from autograd import grad as ag_grad
import autograd.numpy as ag_np
import numpy as np
import pytest
import sys
import tangent
import utils


# Documented limitations of second-order (reverse-over-reverse) AD.

# Differentiating low-level tape operations (tangent.push/pop/Stack) twice
# produces correct UNOPTIMIZED second derivatives, but the optimization
# passes' tape-pair elimination still corrupts them. See the
# "Higher-Order Differentiation" section of
# docs/features/PYTHON_FEATURE_SUPPORT.md and the comments on these
# functions in tests/functions.py.
_TAPE_GRADGRAD_LIMITED = frozenset((
    'useless_stack_ops',
    'redefining_var_as_list',
))


def _test_gradgrad_array(func, optimized, *args):
  """Test gradients of functions with NumPy-compatible signatures."""
  if func.__name__ in _TAPE_GRADGRAD_LIMITED and optimized:
    pytest.xfail(
        'Optimized second derivatives through the low-level tape API are '
        'known to be wrong (documented limitation).')

  def tangent_func():
    func.__globals__['np'] = np
    df = tangent.grad(func, optimized=optimized, verbose=True)
    ddf = tangent.grad(df, optimized=optimized, verbose=True)
    return ddf(*args)

  def reference_func():
    func.__globals__['np'] = ag_np
    return ag_grad(ag_grad(func))(*args)

  def backup_reference_func():
    return utils.numeric_grad(utils.numeric_grad(func))(*args)

  utils.assert_result_matches_reference(
      tangent_func, reference_func, backup_reference_func,
      tolerance=1e-2)  # extra loose bounds for 2nd order grad


def test_reverse_over_reverse_unary(func, a, optimized):
  _test_gradgrad_array(func, optimized, a)


def test_reverse_over_reverse_binary(func, a, b, optimized):
  _test_gradgrad_array(func, optimized, a, b)


def test_reverse_over_reverse_ternary(func, optimized, a, b, c):
  _test_gradgrad_array(func, optimized, a, b, c)


def test_third_derivative_polynomial(optimized):
  """Third derivatives of ordinary functions (reverse-over-reverse-over-reverse).

  Requires the adjoints registered for Tangent's own accumulation helpers
  (tangent.unreduce_like etc.) so that differentiating second-order adjoint
  code does not step into their type-dispatch bodies. `optimized` is supplied
  by conftest for both modes.
  """
  def f(x):
    return np.sum(x * x * x)

  x = np.array([1.0, 2.0])
  old_limit = sys.getrecursionlimit()
  sys.setrecursionlimit(max(old_limit, 20000))
  try:
    d3 = tangent.grad(tangent.grad(tangent.grad(f, optimized=optimized),
                                    optimized=optimized), optimized=optimized)
    got = d3(x)
  finally:
    sys.setrecursionlimit(old_limit)
  # d^3/dx^3 sum(x^3) = 6 elementwise.
  assert np.allclose(got, np.array([6.0, 6.0]), atol=1e-2)


def test_third_derivative_exp(optimized):
  def f(x):
    return np.sum(np.exp(x))

  x = np.array([0.5, 1.0])
  old_limit = sys.getrecursionlimit()
  sys.setrecursionlimit(max(old_limit, 20000))
  try:
    d3 = tangent.grad(tangent.grad(tangent.grad(f, optimized=optimized),
                                    optimized=optimized), optimized=optimized)
    got = d3(x)
  finally:
    sys.setrecursionlimit(old_limit)
  assert np.allclose(got, np.exp(x), atol=1e-2)


if __name__ == '__main__':
  assert not pytest.main([__file__, '--short'])
