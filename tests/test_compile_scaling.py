"""Long functions must compile: no recursion limits, terminating fixpoints.

Pins two compile-time fixes:

1. `cfg.Forward.visit` used to recurse once per CFG node, so functions a few
   hundred statements long crashed with RecursionError before they could be
   differentiated. It is now an iterative worklist.
2. The optimizer's fixpoints report changes directly instead of serializing
   the whole AST (`gast.dump`) twice per iteration. A pass must never report
   a change that `transformers.Remove` then refuses to apply (e.g. statements
   containing pri_call/adj_call), or the fixpoint would spin forever - which
   is exercised by differentiating through a called helper function.
"""

import sys
import tempfile
import textwrap

import gast
import pytest

import tangent
from tangent import cfg
from tangent import quoting


def _make_long_function(n):
    """Import a generated module holding an n-statement chain function."""
    lines = ['def long_chain(x):', '    v0 = x * 1.01']
    for k in range(1, n):
        lines.append('    v%d = v%d * 1.01' % (k, k - 1))
    lines.append('    return v%d' % (n - 1))
    src = '\n'.join(lines) + '\n'
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    f.write(src)
    f.close()
    import importlib.util

    spec = importlib.util.spec_from_file_location('long_chain_mod', f.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['long_chain_mod'] = mod
    spec.loader.exec_module(mod)
    return mod.long_chain


def test_cfg_analysis_handles_long_functions():
    # 2000 statements: far beyond what the old recursive CFG walk survived
    # (it hit the default recursion limit around ~300 statements).
    body = '\n'.join('    x = x * 1.0' for _ in range(2000))
    node = quoting.parse_string('def f(x):\n%s\n    return x\n' % body)
    cfg.forward(node.body[0], cfg.Defined())
    assert gast.dump(node)  # analysis completed without RecursionError


def test_long_function_differentiates():
    # End-to-end: 400 chained statements previously crashed with
    # RecursionError inside the reaching-definitions analysis.
    fn = _make_long_function(400)
    df = tangent.grad(fn)
    assert df(1.0) == pytest.approx(1.01**400, rel=1e-9)


def test_optimize_terminates_through_called_functions():
    # Differentiating through a called helper generates pri_call/adj_call
    # statements that `Remove` refuses to delete; DCE must not count them as
    # removal candidates or its fixpoint never terminates.
    df = tangent.grad(_helper_caller)
    assert df(3.0) == pytest.approx(2.0)


def _helper(x):
    y = x * 2.0
    unused = y * 3.0  # noqa: F841 - dead on purpose, DCE bait
    return y


def _helper_caller(x):
    return _helper(x)


def test_constant_folding_reports_changes_exactly():
    from tangent import optimization

    node = quoting.parse_string('y = x * 1.0 + 0.0')
    node, changed = optimization._constant_folding_once(node)
    assert changed
    node, changed = optimization._constant_folding_once(node)
    assert not changed
    assert 'y = x' == quoting.to_source(node).strip()
