"""A feature-coverage matrix for the Python language surface Tangent accepts.

Each entry pins down the *current* status of a construct: either it
differentiates to a known-correct gradient, or it is rejected with a clean,
actionable error. The point is to make coverage measurable and to guard against
the worst failure mode - a construct that used to be rejected quietly starting
to compile and return a wrong gradient (or vice versa).

If you change support for a construct, update its row here; an unexpected pass
or failure in this file is a signal that coverage moved.
"""
import numpy as np
import pytest

import tangent
from tangent.errors import GradientNotFoundError
from tangent.errors import TangentParseError


# --- Constructs that differentiate correctly ---------------------------------

def _assigned_lambda(x):
  sq = lambda y: y * y
  return sq(x)


def _make_multiplier(scale):
  def g(x):
    return x * scale
  return g


def _listcomp_constant_range(x):
  vals = [x * i for i in range(4)]
  return np.sum(vals)


def _listcomp_filtered(x):
  vals = [x * i for i in range(4) if i > 1]
  return vals[0]


def _scatter_slice(x):
  a = np.zeros(4)
  a[0:2] = x
  return np.sum(a)


def _scatter_scalar_index(x):
  a = np.zeros(4)
  a[1] = x
  return np.sum(a)


def _chained_comparison(x):
  if 0 < x < 10:
    return x * x
  return x


def _nested_ternary(x):
  return x * x if x > 0 else (x if x > -5 else -x)


def _enumerate_loop(x):
  r = 0.0
  for i, v in enumerate(np.array([1.0, 2.0])):
    r = r + i * v * x
  return r


def _zip_loop(x):
  r = 0.0
  for a, b in zip(np.array([1.0, 2.0]), np.array([3.0, 4.0])):
    r = r + a * b * x
  return r


def _fstring_assert(x):
  assert x > 0, f"x must be positive, got {x}"
  return x * x


def _default_arg(x, p=2.0):
  return x ** p


def _augassign_subscript(x):
  a = np.zeros(4)
  a[0:2] += x
  return np.sum(a)


CORRECT_GRADIENTS = [
    # (function, input, expected gradient)
    (_assigned_lambda, 2.0, 4.0),          # d/dx x^2 = 2x
    (_listcomp_constant_range, 2.0, 6.0),  # x*(0+1+2+3) = 6x
    (_listcomp_filtered, 2.0, 2.0),        # vals[0] = 2x
    (_scatter_slice, 2.0, 2.0),            # sum of 2 scattered x's
    (_scatter_scalar_index, 2.0, 1.0),     # single scatter
    (_chained_comparison, 2.0, 4.0),       # x^2 in range
    (_nested_ternary, 2.0, 4.0),           # x^2 branch
    (_enumerate_loop, 2.0, 2.0),           # (0*1 + 1*2) x = 2x
    (_zip_loop, 2.0, 11.0),                # (1*3 + 2*4) x = 11x
    (_fstring_assert, 2.0, 4.0),           # x^2
    (_default_arg, 2.0, 4.0),              # x^2
    (_augassign_subscript, 2.0, 2.0),      # a[0:2] += x scatters into 2 elts
]


@pytest.mark.parametrize('fn,arg,expected', [
    pytest.param(fn, arg, expected, id=fn.__name__)
    for fn, arg, expected in CORRECT_GRADIENTS
])
def test_correct_gradient(fn, arg, expected):
  assert tangent.grad(fn)(arg) == pytest.approx(expected)


def test_external_closure():
  # A closure created outside the differentiated function is supported.
  g = _make_multiplier(3.0)
  assert tangent.grad(g)(2.0) == pytest.approx(3.0)


# --- Constructs that must be rejected cleanly --------------------------------
#
# Each maps to a clear TangentParseError (language feature) or
# GradientNotFoundError (callable with no adjoint and no retrievable source).

def _walrus(x):
  if (y := x * 2) > 1:
    return y
  return x


def _raise(x):
  if x < 0:
    raise ValueError('negative')
  return x * x


def _try_except(x):
  try:
    return x * x
  except Exception:
    return x


def _nested_def(x):
  def inner(y):
    return y * 2
  return inner(x)


def _closure_in_fn(x):
  scale = 3.0

  def inner(y):
    return y * scale
  return inner(x)


def _recursion(x):
  def rec(n, acc):
    if n == 0:
      return acc
    return rec(n - 1, acc + x)
  return rec(3, 0.0)


def _star_args(*xs):
  return xs[0] * xs[0]


def _varkw(x, **kw):
  return x * x


def _inline_lambda(x):
  return (lambda y: y * y)(x)


def _break_loop(x):
  r = 0.0
  for i in range(10):
    r = r + x
    if r > 5:
      break
  return r


def _continue_loop(x):
  r = 0.0
  for i in range(10):
    if i == 5:
      continue
    r = r + x
  return r


def _class_def(x):
  class M:
    pass
  return x * x


def _yield(x):
  yield x * x


def _genexp(x):
  return sum(x * i for i in range(3))


def _dictcomp_dynamic(x, n):
  d = {i: x * i for i in range(n)}
  return d[1]


def _global_stmt(x):
  global _FEATURE_MATRIX_GLOBAL
  return x * _FEATURE_MATRIX_GLOBAL


def _del_stmt(x):
  y = x * x
  del y
  return x * 3


def _import_inside(x):
  import math
  return x * x


def _builtin_sum(x):
  return sum((x, x * x))


def _int_cast(x):
  return float(int(x)) + x


def _augassign_attribute(x):
  a = np.zeros(3)
  a.real += x
  return np.sum(a)


REJECTIONS = [
    (_walrus, TangentParseError),
    (_raise, TangentParseError),
    (_try_except, TangentParseError),
    (_nested_def, TangentParseError),
    (_closure_in_fn, TangentParseError),
    (_recursion, TangentParseError),
    (_star_args, TangentParseError),
    (_varkw, TangentParseError),
    (_inline_lambda, TangentParseError),
    (_break_loop, TangentParseError),
    (_continue_loop, TangentParseError),
    (_class_def, TangentParseError),
    (_yield, TangentParseError),
    (_genexp, TangentParseError),
    (_dictcomp_dynamic, TangentParseError),
    (_global_stmt, TangentParseError),
    (_del_stmt, TangentParseError),
    (_import_inside, TangentParseError),
    (_builtin_sum, GradientNotFoundError),
    (_int_cast, GradientNotFoundError),
    (_augassign_attribute, TangentParseError),
]


@pytest.mark.parametrize('fn,exc', [
    pytest.param(fn, exc, id=fn.__name__) for fn, exc in REJECTIONS
])
def test_clean_rejection(fn, exc):
  # A rejected construct must raise a clean Tangent error, never crash with an
  # internal ValueError/IndexError/TypeError and never return a wrong gradient.
  with pytest.raises(exc):
    tangent.grad(fn)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
