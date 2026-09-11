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
causing a subtle failure three passes later. It is off by default (a pure AST
walk per pass has a cost) but exercised across the whole corpus by
`tests/test_ir_invariants.py`, which is the standing proof that the passes
agree.

The invariants are keyed by pass name; a pass with no registered invariant is
simply not checked.
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


# --- registry --------------------------------------------------------------
#
# Maps a pass name to the invariant its output must satisfy. run_passes calls
# the entry for each pass it runs (when verification is on). A pass absent here
# is not checked.
INVARIANTS = {
    'lambda_desugar': check_no_lambdas,
    'class_desugar': check_single_function,
    'resolve_calls': check_calls_resolved,
    'anf': check_anf,
}


def verify_after(pass_name, node):
    """Run the invariant registered for `pass_name`, if any."""
    check = INVARIANTS.get(pass_name)
    if check is not None:
        check(node, pass_name)
    return node
