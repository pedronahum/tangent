"""The frontend passes satisfy their documented IR invariants (tangent/verify.py).

The pass pipeline is a sequence of source-to-source rewrites, each assuming the
previous ones' output shape. `verify` turns those assumptions into checked
contracts; this file is the standing proof that the pipeline actually meets
them - lowering a spread of real functions with verification on must never
raise IRInvariantError - plus unit tests that each invariant genuinely catches
a violation (so the checks are not vacuous).
"""

import gast
import numpy as np
import pytest

import tangent
from tangent import passes, quoting, verify
from tangent.verify import IRInvariantError


# --- Functions spanning the constructs the passes rewrite. ---


def straight_line(x):
    return np.sum(np.tanh(x) ** 2 + 3.0 * x)


def indexed_loop(x):
    s = 0.0
    for i in range(len(x)):
        s = s + x[i] * x[i]
    return s


def dynamic_comprehension(x):
    ys = [v * v for v in x]
    t = 0.0
    for i in range(len(ys)):
        t = t + ys[i]
    return t


def branching(x):
    if np.sum(x) > 0.0:
        y = x * x
    else:
        y = -x
    return np.sum(y)


def uses_assigned_lambda(x):
    sq = lambda v: v * v  # noqa: E731 - exercises lambda_desugar
    return np.sum(sq(x))


def while_loop(a):
    while np.abs(a) > 0.1:
        a = a * 0.5
    return a


CORPUS = [
    straight_line,
    indexed_loop,
    dynamic_comprehension,
    branching,
    uses_assigned_lambda,
    while_loop,
]


class TestPipelineSatisfiesInvariants:
    @pytest.mark.parametrize('fn', CORPUS, ids=[f.__name__ for f in CORPUS])
    @pytest.mark.parametrize('mode', ['reverse', 'forward'])
    def test_lower_with_verification(self, fn, mode):
        import inspect

        node = quoting.parse_function(fn)
        source = inspect.getsource(fn)
        # verify=True checks every registered invariant after each pass; a
        # violation raises IRInvariantError.
        passes.run_passes(node, fn, source, mode, verify=True)

    @pytest.mark.parametrize('fn', CORPUS, ids=[f.__name__ for f in CORPUS])
    def test_grad_runs_under_verification(self, fn, monkeypatch):
        monkeypatch.setenv('TANGENT_VERIFY_IR', '1')
        monkeypatch.setenv('TANGENT_DISK_CACHE', '0')
        tangent.clear_cache()
        x = np.array([0.5, -1.0, 2.0]) if fn is not while_loop else 0.7
        tangent.grad(fn)(x)  # must not raise IRInvariantError


class TestInvariantsAreNotVacuous:
    def test_anf_catches_nested_expression(self):
        node = quoting.parse_string('def f(x):\n    y = (x + x) * x\n    return y\n')
        with pytest.raises(IRInvariantError, match='binop operands are trivial'):
            verify.check_anf(node)

    def test_anf_catches_nested_call_arg(self):
        node = quoting.parse_string('def f(x):\n    y = g(h(x))\n    return y\n')
        with pytest.raises(IRInvariantError, match='call arguments are trivial'):
            verify.check_anf(node)

    def test_resolved_catches_unannotated_call(self):
        node = quoting.parse_string('def f(x):\n    y = g(x)\n    return y\n')
        with pytest.raises(IRInvariantError, match='func annotation'):
            verify.check_calls_resolved(node)

    def test_single_function_catches_nested_def(self):
        node = quoting.parse_string('def f(x):\n    def g(y):\n        return y\n    return g(x)\n')
        with pytest.raises(IRInvariantError, match='nested function'):
            verify.check_single_function(node)

    def test_no_lambdas_catches_lambda(self):
        node = quoting.parse_string('def f(x):\n    return (lambda y: y)(x)\n')
        with pytest.raises(IRInvariantError, match='lambda'):
            verify.check_no_lambdas(node)

    def test_valid_anf_passes(self):
        # An already-ANF body must NOT raise.
        node = quoting.parse_string('def f(x):\n    a = x + x\n    b = a * x\n    return b\n')
        verify.check_anf(node)


def test_disabled_by_default():
    assert not verify.enabled()  # off unless TANGENT_VERIFY_IR is set
