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
"""Tests for the straight-line coarsening prototype.

Coarsening lifts a straight-line primal segment to a single symbolic (SymPy)
expression, differentiates it once, and emits the resulting vector-Jacobian
product directly.  These tests check the generated adjoint numerically against
central finite differences of the primal, and check that non-straight-line
inputs are rejected (return None) rather than miscompiled.
"""
import math
import unittest

import pytest

# SymPy is an optional dependency; skip this module when it is missing.
pytest.importorskip('sympy')

import gast
from tangent.optimizations.coarsening import (
    StraightLineCoarsener,
    apply_coarsening,
)

# NumPy is a hard dependency of Tangent; the end-to-end check below uses it.
np = pytest.importorskip('numpy')


# Module-level primal for the end-to-end check: Tangent needs real source via
# inspect.getsource, so it cannot be defined inside a test or exec'd string.
def _primal_np(x, y):
  a = x * y
  b = np.sin(a)
  c = b + x
  return c


# Namespace used to execute both the primal and the generated adjoint.  The
# coarsened adjoint references elementwise primitives by bare name, so they
# must be present here.
MATHNS = {
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
    'abs': abs,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
}


def _build(primal_src):
  """Coarsen `primal_src` and return (primal_callable, adjoint_callable)."""
  func = gast.parse(primal_src).body[0]
  adj = apply_coarsening(func)
  assert adj is not None, 'expected %r to be coarsenable' % primal_src
  ns = dict(MATHNS)
  ns['np'] = np  # allow attribute callees such as np.sin in the primal
  exec(primal_src, ns)
  exec(gast.unparse(adj), ns)
  return ns[func.name], ns[adj.name]


def _fd_grad_i(f, point, i, h=1e-6):
  """Central finite-difference partial derivative of f at point w.r.t. arg i."""
  fwd = list(point)
  bwd = list(point)
  fwd[i] += h
  bwd[i] -= h
  return (f(*fwd) - f(*bwd)) / (2.0 * h)


def _check_gradients(primal_src, point, seed=1.0, tol=1e-5):
  """Assert the coarsened adjoint matches finite differences at `point`."""
  f, df = _build(primal_src)
  adjoints = df(*point, seed)
  if not isinstance(adjoints, tuple):
    adjoints = (adjoints,)
  assert len(adjoints) == len(point)
  for i in range(len(point)):
    expected = seed * _fd_grad_i(f, point, i)
    assert math.isclose(adjoints[i], expected, rel_tol=tol, abs_tol=1e-6), (
        'arg %d: adjoint %r != finite-difference %r'
        % (i, adjoints[i], expected))
  return adjoints


class TestCoarseningGradients(unittest.TestCase):
  """Numerical correctness of the coarsened adjoint vs finite differences."""

  def test_single_sin(self):
    _check_gradients('def f(x):\n    return sin(x)', (0.7,))

  def test_product_rule(self):
    # d/dx [exp(x) * x] = exp(x) * (x + 1)
    _check_gradients('def f(x):\n    return exp(x) * x', (0.5,))

  def test_two_inputs(self):
    _check_gradients('def f(x, y):\n    return x * y + sin(x)', (0.7, 1.3))

  def test_log(self):
    # Positive point keeps 1/x on the real branch.
    _check_gradients('def f(x):\n    return log(x)', (2.0,))

  def test_sqrt(self):
    _check_gradients('def f(x):\n    return sqrt(x)', (4.0,))

  def test_tan(self):
    _check_gradients('def f(x):\n    return tan(x)', (0.4,))

  def test_attribute_callee(self):
    # np.sin is an Attribute call target; it must be coarsenable and correct.
    _check_gradients('def f(x):\n    return np.sin(x)', (0.7,))

  def test_multi_statement_inlining(self):
    # Exercises inlining across several intermediate assignments.
    src = ('def f(x, y):\n'
           '    a = x * y\n'
           '    b = sin(a)\n'
           '    c = b + x\n'
           '    return c\n')
    _check_gradients(src, (0.5, 1.5))

  def test_unused_input_is_zero(self):
    # y does not influence the output, so its adjoint must be 0.
    adjoints = _check_gradients('def f(x, y):\n    return sin(x)', (0.7, 2.0))
    assert adjoints[1] == 0.0

  def test_seed_scales_adjoint(self):
    # A seed of 3 must scale every adjoint by 3 (linearity of the VJP).
    src = 'def f(x, y):\n    return x * y + sin(x)'
    point = (0.7, 1.3)
    base = _check_gradients(src, point, seed=1.0)
    scaled = _check_gradients(src, point, seed=3.0)
    for b, s in zip(base, scaled):
      assert math.isclose(s, 3.0 * b, rel_tol=1e-9, abs_tol=1e-9)


class TestCoarseningSimplification(unittest.TestCase):
  """Coarsening should expose cross-operation symbolic simplification."""

  def test_trig_identity_collapses_to_zero(self):
    # sin(x)^2 + cos(x)^2 == 1, so the derivative is exactly 0.  Without
    # symbolic simplification the naive per-op adjoint would carry the
    # 2*sin*cos - 2*cos*sin terms.
    src = 'def f(x):\n    return sin(x) ** 2 + cos(x) ** 2'
    adjoints = _check_gradients(src, (0.9,))
    assert adjoints[0] == 0.0

  def test_log_exp_cancellation(self):
    # log(exp(x)) == x, so the derivative simplifies to exactly 1.
    src = 'def f(x):\n    return log(exp(x))'
    adjoints = _check_gradients(src, (0.3,))
    assert math.isclose(adjoints[0], 1.0, rel_tol=1e-9, abs_tol=1e-9)


class TestCoarseningRejection(unittest.TestCase):
  """Non-straight-line inputs must be rejected (return None), never miscompiled."""

  def _assert_rejected(self, src):
    func = gast.parse(src).body[0]
    self.assertIsNone(apply_coarsening(func))

  def test_control_flow(self):
    self._assert_rejected(
        'def g(x):\n    if x > 0:\n        return x\n    return -x\n')

  def test_loops(self):
    self._assert_rejected(
        'def g(x):\n    s = x\n    for i in range(3):\n        s = s * x\n'
        '    return s\n')

  def test_unsupported_call(self):
    self._assert_rejected('def h(x):\n    return unknown_op(x)\n')

  def test_augmented_assignment(self):
    self._assert_rejected('def h(x):\n    y = x\n    y += x\n    return y\n')

  def test_subscript_target(self):
    self._assert_rejected(
        'def h(x):\n    out = x\n    out[0] = x\n    return out\n')

  def test_attribute_data_access(self):
    # Attribute used as data (not as a call target) must be rejected.
    self._assert_rejected('def h(x):\n    return x.real\n')

  def test_no_return(self):
    self._assert_rejected('def h(x):\n    y = x\n')

  def test_empty_return(self):
    self._assert_rejected('def h(x):\n    y = x\n    return\n')

  def test_varargs(self):
    self._assert_rejected('def h(*xs):\n    return xs\n')


class TestCoarsenerAPI(unittest.TestCase):
  """Exercise the StraightLineCoarsener object and configuration options."""

  def test_diagnostics(self):
    src = 'def f(x, y):\n    return x * y'
    func = gast.parse(src).body[0]
    coarsener = StraightLineCoarsener()
    adj = coarsener.coarsen(func)
    self.assertIsNotNone(adj)
    self.assertEqual(coarsener.inputs, ['x', 'y'])
    self.assertIsNotNone(coarsener.output_expr)

  def test_custom_seed_name(self):
    src = 'def f(x):\n    return sin(x)'
    func = gast.parse(src).body[0]
    adj = apply_coarsening(func, config={'seed_name': 'g_out'})
    param_names = [getattr(a, 'id', None) or getattr(a, 'arg', None)
                   for a in adj.args.args]
    self.assertEqual(param_names, ['x', 'g_out'])

  def test_simplify_toggle(self):
    # With simplification disabled the result must still be a valid adjoint.
    src = 'def f(x):\n    return sin(x) ** 2 + cos(x) ** 2'
    func = gast.parse(src).body[0]
    adj = apply_coarsening(func, config={'simplify': False})
    self.assertIsNotNone(adj)


class TestCoarseningVsTangent(unittest.TestCase):
  """End-to-end: the coarsened adjoint must agree with tangent.grad."""

  def test_matches_tangent_grad(self):
    tangent = pytest.importorskip('tangent')
    import inspect

    gf = tangent.grad(_primal_np, wrt=(0, 1))
    x, y = 0.5, 1.5
    tx, ty = gf(x, y)
    tx, ty = float(np.asarray(tx)), float(np.asarray(ty))

    src = inspect.getsource(_primal_np)
    func = gast.parse(src).body[0]
    adj = apply_coarsening(func)
    self.assertIsNotNone(adj)

    # The primal used np.sin; the lowered adjoint refers to bare sin/cos.
    ns = {'sin': math.sin, 'cos': math.cos}
    exec(gast.unparse(adj), ns)
    cx, cy = ns[adj.name](x, y, 1.0)

    assert math.isclose(tx, cx, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(ty, cy, rel_tol=1e-6, abs_tol=1e-9)


if __name__ == '__main__':
  unittest.main(verbosity=2)
