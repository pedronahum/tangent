"""The frontend passes satisfy their documented IR invariants (tangent/verify.py).

The pass pipeline is a sequence of source-to-source rewrites, each assuming the
previous ones' output shape. `verify` turns those assumptions into checked
contracts; this file is the standing proof that the pipeline actually meets
them - lowering a spread of real functions with verification on must never
raise IRInvariantError - plus unit tests that each invariant genuinely catches
a violation (so the checks are not vacuous).
"""

import os

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


def early_return(x):
    if np.sum(x) > 0.0:
        return np.sum(x * x)
    return -np.sum(x)


def return_in_loop(x):
    s = 0.0
    for i in range(len(x)):
        s = s + x[i]
        if s > 100.0:
            return s
    return s


def has_break(x):
    s = 0.0
    for i in range(len(x)):
        if x[i] < 0.0:
            break
        s = s + x[i]
    return s


def has_continue(x):
    s = 0.0
    for i in range(len(x)):
        if x[i] < 0.0:
            continue
        s = s + x[i]
    return s


def chained_assignment(x):
    a = b = x * 2.0
    return np.sum(a + b)


CORPUS = [
    straight_line,
    indexed_loop,
    dynamic_comprehension,
    branching,
    uses_assigned_lambda,
    while_loop,
    early_return,
    return_in_loop,
    has_break,
    has_continue,
    chained_assignment,
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

    def test_single_return_catches_two_returns(self):
        node = quoting.parse_string('def f(x):\n    return x\n    return -x\n')
        with pytest.raises(IRInvariantError, match='exactly one return'):
            verify.check_single_return(node)

    def test_single_return_catches_non_trailing_return(self):
        node = quoting.parse_string(
            'def f(x):\n    if x > 0:\n        return x\n    y = -x\n    return y\n'
        )
        # Two returns here (the early one and the trailing one).
        with pytest.raises(IRInvariantError):
            verify.check_single_return(node)

    def test_no_loop_exits_catches_break(self):
        node = quoting.parse_string(
            'def f(x):\n    for i in range(3):\n        break\n    return x\n'
        )
        with pytest.raises(IRInvariantError, match='break/continue'):
            verify.check_no_loop_exits(node)

    def test_no_loop_exits_catches_continue(self):
        node = quoting.parse_string(
            'def f(x):\n    for i in range(3):\n        continue\n    return x\n'
        )
        with pytest.raises(IRInvariantError, match='break/continue'):
            verify.check_no_loop_exits(node)

    def test_single_target_catches_chained_assignment(self):
        node = quoting.parse_string('def f(x):\n    a = b = x\n    return a\n')
        with pytest.raises(IRInvariantError, match='single assignment target'):
            verify.check_single_target(node)

    def test_valid_single_return_and_targets_pass(self):
        node = quoting.parse_string('def f(x):\n    a = x + x\n    return a\n')
        verify.check_single_return(node)
        verify.check_no_loop_exits(node)
        verify.check_single_target(node)


class TestIRModule:
    """run_passes returns an explicit IRModule marking the lowered frontend IR."""

    def test_run_passes_returns_ir_module(self):
        import inspect

        from tangent import ir

        node = quoting.parse_function(indexed_loop)
        result = passes.run_passes(node, indexed_loop, inspect.getsource(indexed_loop), 'reverse')
        assert isinstance(result, ir.IRModule)
        assert isinstance(result.module, gast.Module)
        assert result.mode == 'reverse'
        # It records the passes that ran (which carry the invariants).
        assert {'anf', 'resolve_calls', 'return_desugar'} <= result.satisfied

    def test_function_property_returns_the_lowered_functiondef(self):
        import inspect

        node = quoting.parse_function(straight_line)
        ir_mod = passes.run_passes(node, straight_line, inspect.getsource(straight_line), 'reverse')
        fn = ir_mod.function
        assert isinstance(fn, gast.FunctionDef)

    def test_verify_re_checks_invariants(self):
        import inspect

        node = quoting.parse_function(branching)
        ir_mod = passes.run_passes(node, branching, inspect.getsource(branching), 'reverse')
        # A well-formed IR re-verifies without raising.
        assert ir_mod.verify() is ir_mod

    def test_verify_detects_a_corrupted_module(self):
        import inspect

        from tangent import ir
        from tangent.verify import IRInvariantError

        node = quoting.parse_function(straight_line)
        ir_mod = passes.run_passes(node, straight_line, inspect.getsource(straight_line), 'reverse')
        # Splice a non-ANF assignment in and re-verify: the anf invariant fires.
        bad = quoting.parse_string('def _(x):\n    y = g(h(x))\n    return y\n').body[0]
        ir_mod.function.body[:0] = bad.body[:1]
        with pytest.raises(IRInvariantError):
            ir_mod.verify()

    def test_rejects_non_module(self):
        from tangent import ir

        with pytest.raises(TypeError):
            ir.IRModule(module='not a module', mode='reverse')


class TestEveryPassHasAContract:
    def test_no_pass_is_unaccounted_for(self):
        # Every frontend pass must have a registered invariant or an explicit
        # exemption; a new pass added without either fails here.
        unchecked = verify.unchecked_passes()
        assert unchecked == set(), (
            'passes with no IR contract (add an invariant to verify.INVARIANTS '
            'or list them in verify.EXEMPT with a reason): %s' % sorted(unchecked)
        )

    def test_invariants_and_exemptions_do_not_overlap(self):
        assert set(verify.INVARIANTS) & set(verify.EXEMPT) == set()


@pytest.mark.skipif(
    os.environ.get('TANGENT_VERIFY_IR', '0') != '0',
    reason='the CI verification gate sets TANGENT_VERIFY_IR=1; this asserts the default',
)
def test_disabled_by_default():
    assert not verify.enabled()  # off unless TANGENT_VERIFY_IR is set
