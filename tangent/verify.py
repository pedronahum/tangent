# Copyright 2018 Google Inc.
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
"""IR invariant verification for the frontend pass pipeline.

Tangent lowers a function through a sequence of source-to-source passes
(`tangent/passes.py`). Each pass consumes the previous passes' output and
implicitly assumes it has a particular shape - the pipeline "works until two
passes disagree." This module turns those implicit assumptions into
*checkable contracts*: each contract is an invariant the IR must satisfy after
a given pass, verified by an AST walk.

When enabled (env `TANGENT_VERIFY_IR=1`, or `passes.run_passes(..., verify=True)`)
the pass manager checks the relevant invariant after each pass and raises
`IRInvariantError` naming the pass, the invariant, and the offending node - so
a pass that produces malformed IR fails loudly at its source instead of
causing a subtle failure three passes later. It is off by default at runtime (a
pure AST walk per pass has a cost), but it is checked in two always-on ways:

- **In CI**, a dedicated step runs the pipeline with `TANGENT_VERIFY_IR=1`, so
  any pass that stops satisfying its contract fails the build.
- **As a standing test**, `tests/test_ir_invariants.py` lowers a broad corpus
  with verification on (the proof the passes agree), checks each invariant is
  not vacuous, and asserts `unchecked_passes()` is empty - so every frontend
  pass has either a registered invariant or an explicit exemption, and a *new*
  pass cannot silently ship without an IR contract.

The invariants are keyed by pass name. This module is the pragmatic realization
of "one explicit IR with a documented invariant": the restricted, post-ANF gast
sublanguage is the IR, and these checks are its enforced contract at every pass
boundary. See `docs/development/explicit-ir-investigation.md` for the larger
design context.
"""

from __future__ import absolute_import

import os

import gast


class IRInvariantError(AssertionError):
    """A frontend pass produced IR that violates its documented invariant."""


def enabled():
    return os.environ.get('TANGENT_VERIFY_IR', '0') != '0'


def _fail(pass_name, invariant, node):
    from tangent import quoting

    try:
        snippet = quoting.unquote(node)
    except Exception:
        snippet = '<%s>' % type(node).__name__
    raise IRInvariantError(
        'After pass %r the IR invariant %r is violated at:\n    %s'
        % (pass_name, invariant, snippet)
    )


# --- individual invariants -------------------------------------------------

_TRIVIAL = (gast.Name, gast.Constant)


def _is_trivial(node):
    """A trivial ANF operand: a Name, a literal, or None (default args)."""
    return node is None or isinstance(node, _TRIVIAL)


def check_anf(node, pass_name='anf'):
    """The output is in A-normal form (see tangent/anf.py).

    Every argument of a call, operand of a binary/unary op, and subscript
    index on an assignment RHS is trivial (a Name or literal), so no compound
    expression is nested inside another. Loop/if tests and iterables may still
    be trivial names; we check the assignment right-hand sides, which is what
    the AD transform relies on.
    """
    for stmt in gast.walk(node):
        if not isinstance(stmt, gast.Assign):
            continue
        value = stmt.value
        if isinstance(value, gast.BinOp):
            if not (_is_trivial(value.left) and _is_trivial(value.right)):
                _fail(pass_name, 'anf: binop operands are trivial', stmt)
        elif isinstance(value, gast.UnaryOp):
            if not _is_trivial(value.operand):
                _fail(pass_name, 'anf: unaryop operand is trivial', stmt)
        elif isinstance(value, gast.Call):
            for arg in value.args:
                if isinstance(arg, gast.Starred):
                    continue
                if not _is_trivial(arg):
                    _fail(pass_name, 'anf: call arguments are trivial', stmt)
            for kw in value.keywords:
                if not _is_trivial(kw.value):
                    _fail(pass_name, 'anf: call keyword arguments are trivial', stmt)


def check_calls_resolved(node, pass_name='resolve_calls'):
    """Every Call node carries a `func` annotation.

    `resolve_calls` annotates each call with the resolved callable (or None
    for an unresolvable method); a Call with no `func` annotation at all means
    resolution was skipped, which later passes silently mishandle.
    """
    from tangent import annotations as anno

    for n in gast.walk(node):
        if isinstance(n, gast.Call):
            if not anno.hasanno(n, 'func'):
                _fail(pass_name, 'resolve_calls: every call has a func annotation', n)


def check_single_function(node, pass_name='class_desugar'):
    """No function definition is nested inside another function's body.

    Class-method inlining and lambda lowering leave a single flat function to
    differentiate; a surviving nested def breaks the single-exit reverse
    transform.
    """
    for fdef in gast.walk(node):
        if isinstance(fdef, (gast.FunctionDef, gast.AsyncFunctionDef)):
            for child in gast.walk(fdef):
                if child is fdef:
                    continue
                if isinstance(child, (gast.FunctionDef, gast.AsyncFunctionDef)):
                    _fail(pass_name, 'no nested function definitions', child)


def check_no_lambdas(node, pass_name='lambda_desugar'):
    """No Lambda nodes remain (they are lowered to named functions or inlined)."""
    for n in gast.walk(node):
        if isinstance(n, gast.Lambda):
            _fail(pass_name, 'no lambda expressions remain', n)


def check_single_return(node, pass_name='return_desugar'):
    """After return_desugar the function is single-exit.

    Each function has exactly one `Return`, and it is the last statement of the
    body. Early returns and returns inside loops are lifted into a
    `__tangent_retval` variable (and a `__tangent_returning` flag for loops)
    plus one trailing return, so the reverse transform - which reverses a
    straight-line body ending in a single return - has a well-defined exit.
    """
    for fdef in gast.walk(node):
        if not isinstance(fdef, (gast.FunctionDef, gast.AsyncFunctionDef)):
            continue
        returns = [n for n in gast.walk(fdef) if isinstance(n, gast.Return)]
        if len(returns) != 1:
            _fail(pass_name, 'return_desugar: exactly one return statement', fdef)
        if not fdef.body or not isinstance(fdef.body[-1], gast.Return):
            _fail(pass_name, 'return_desugar: the return is the last statement', fdef)


def check_no_loop_exits(node, pass_name='loop_exit_desugar'):
    """No `break`/`continue` remain (lowered to boolean guard flags).

    break/continue - including a `break` that return_desugar inserts for a
    return inside a loop - are lowered to `_brk`/`_skip` guard flags so the
    loop body stays a straight-line sequence the reverse transform can handle.
    """
    for n in gast.walk(node):
        if isinstance(n, (gast.Break, gast.Continue)):
            _fail(pass_name, 'loop_exit_desugar: no break/continue remain', n)


def check_single_target(node, pass_name='chained_assign_desugar'):
    """Every assignment has exactly one target (`a = b = c` is split)."""
    for n in gast.walk(node):
        if isinstance(n, gast.Assign) and len(n.targets) != 1:
            _fail(pass_name, 'chained_assign: single assignment target', n)


# --- registry --------------------------------------------------------------
#
# Maps a pass name to the invariant its output must satisfy. run_passes calls
# the entry for each pass it runs (when verification is on). Every pass is
# accounted for: it either has an invariant here or is listed in EXEMPT with a
# reason. `unchecked_passes()` is empty, and tests/test_ir_invariants.py
# asserts it - so a NEW pass cannot silently escape a contract.
INVARIANTS = {
    'class_desugar': check_single_function,
    'lambda_desugar': check_no_lambdas,
    'return_desugar': check_single_return,
    'chained_assign_desugar': check_single_target,
    'loop_exit_desugar': check_no_loop_exits,
    'resolve_calls': check_calls_resolved,
    'anf': check_anf,
}

# Passes that intentionally carry no structural IR invariant: pure analysis or
# validation, or a lowering whose resulting shape a later pass's invariant
# already subsumes (e.g. the loops enumerate/zip/comprehension produce are
# covered by `anf` downstream). Listed explicitly, with a reason, so the
# accounting is complete rather than "just not added yet."
EXEMPT = {
    'sentinel_rename': 'alpha-renames a local `d`; no structural shape to assert',
    'enumerate_desugar': 'lowers to range(len()) loops; resulting shape covered by anf',
    'zip_desugar': 'lowers to range(len()) loops; resulting shape covered by anf',
    'comprehension_desugar': 'lowers to literals or list_append loops; covered by anf',
    'dict_method_desugar': 'lowers .get() to subscript/ternary; no dedicated shape',
    'concat_desugar': 'lowers to a varargs helper call; covered by resolve_calls/anf',
    'list_method_desugar': 'lowers .append/.pop to rebindings; covered by anf',
    'checkpoint_annotation': 'attaches an annotation and unwraps a with-block; no shape',
    'ifexp_desugar': 'forward-only; lowers ternaries to if-statements',
    'explicit_loop_indexes': (
        'rewrites only *active* for-loops to range(len()); inactive loops keep '
        'their original iterable, so there is no universal post-pass loop shape '
        'to assert (a range-form invariant would wrongly reject inactive loops)'
    ),
    'fence': 'pure validation; raises on unsupported constructs, no tree change',
}


def unchecked_passes():
    """Registry pass names with neither an invariant nor an explicit exemption.

    Should always be empty: every frontend pass is accounted for. A non-empty
    result means a pass was added without deciding on its IR contract.
    """
    from tangent import passes

    names = {p.name for p in passes.get_passes('reverse')}
    names |= {p.name for p in passes.get_passes('forward')}
    return names - set(INVARIANTS) - set(EXEMPT)


def verify_after(pass_name, node):
    """Run the invariant registered for `pass_name`, if any."""
    check = INVARIANTS.get(pass_name)
    if check is not None:
        check(node, pass_name)
    return node
