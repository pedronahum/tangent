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
"""Rewrite early returns into single-exit form.

Tangent's core transforms require each function to have exactly one return
statement, at the very end. Ordinary Python code often returns early::

    def f(x):
        if x > 0:
            return x * x
        return x * 3.0

This module lifts such returns into a synthetic result variable and appends a
single trailing return, so the rest of the pipeline sees single-exit form::

    def f(x):
        if x > 0:
            __tangent_retval = x * x
        else:
            __tangent_retval = x * 3.0
        return __tangent_retval

Returns at the tail of ``if``/``else`` branches and a trailing "fall-through"
return are handled (recursively, so arbitrarily nested conditionals work). A
return inside a plain loop is first lowered into an assignment plus a
``__tangent_returning`` flag plus ``break`` (see `_lower_loop_returns`); the
break is lowered by loop_exit_desugar later in the pipeline and the guard
return is lifted like any other branch return. Only loops with an ``else``
clause reject returns, since their semantics depend on how the loop exited.
"""

from __future__ import absolute_import

import gast

from tangent.errors import TangentParseError

RETVAL_NAME = '__tangent_retval'


def _assign(name, value):
    return gast.Assign(targets=[gast.Name(id=name, ctx=gast.Store(), annotation=None)], value=value)


def _load(name):
    return gast.Name(id=name, ctx=gast.Load(), annotation=None)


def _contains_return(stmts):
    return any(isinstance(n, gast.Return) for stmt in stmts for n in gast.walk(stmt))


RETURNING_FLAG = '__tangent_returning'
LOOP_RETVAL = '__tangent_loop_retval'


def _lower_loop_returns(fdef):
    """Rewrite ``return X`` inside a plain loop as assign + flag + ``break``.

    ::

        for ...:                    __tangent_returning = False   (function top)
            if c:                   for ...:
                return R    ->          if c:
        D                                   __tangent_loop_retval = R
                                            __tangent_returning = True
                                            break
                                    if __tangent_returning:
                                        return __tangent_loop_retval
                                    D

    The injected ``break`` is lowered by loop_exit_desugar later in the
    pipeline, and the injected branch-return is lifted by the ordinary
    single-exit transform below - no new differentiation machinery is needed.
    A return inside a *nested* loop breaks out of its own loop and each
    enclosing loop gains ``if __tangent_returning: break`` right after it, so
    the exit propagates outward. Loops with an ``else`` clause are not
    descended into (their exit-dependent semantics are rejected below).

    Returns True if any return was lowered (the caller then initializes the
    flag at the top of the function).
    """

    def flag_assign(value):
        return gast.Assign(
            targets=[gast.Name(id=RETURNING_FLAG, ctx=gast.Store(), annotation=None)],
            value=gast.Constant(value=value, kind=None),
        )

    def process(stmts, in_loop):
        out = []
        converted = False
        for stmt in stmts:
            if in_loop and isinstance(stmt, gast.Return):
                value = (
                    stmt.value if stmt.value is not None else gast.Constant(value=None, kind=None)
                )
                out.append(_assign(LOOP_RETVAL, value))
                out.append(flag_assign(True))
                out.append(gast.Break())
                # Anything after an unconditional return is unreachable; drop it.
                return out, True
            if isinstance(stmt, (gast.For, gast.While)) and not stmt.orelse:
                stmt.body, inner = process(stmt.body, True)
                out.append(stmt)
                if inner:
                    converted = True
                    exit_stmt = gast.Break() if in_loop else gast.Return(value=_load(LOOP_RETVAL))
                    out.append(gast.If(test=_load(RETURNING_FLAG), body=[exit_stmt], orelse=[]))
                continue
            if isinstance(stmt, gast.If):
                stmt.body, c_body = process(stmt.body, in_loop)
                stmt.orelse, c_else = process(stmt.orelse, in_loop)
                converted = converted or c_body or c_else
            out.append(stmt)
        return out, converted

    fdef.body, converted = process(fdef.body, False)
    if converted:
        fdef.body.insert(0, flag_assign(False))
    return converted


def _reject_returns_in_loops(node):
    """Reject returns in the loops the lowering above skipped.

    After `_lower_loop_returns` the only loops still holding a Return are ones
    with an ``else`` clause (whose semantics depend on how the loop exited) or
    async loops.
    """
    for n in gast.walk(node):
        if isinstance(n, (gast.For, gast.While, gast.AsyncFor)):
            if _contains_return(n.body) or _contains_return(n.orelse):
                raise TangentParseError(
                    'return inside this loop is not supported: returns lower into '
                    'an early loop exit, which a loop with an `else` clause (or an '
                    'async loop) cannot model. Restructure without the loop-else, '
                    'or compute the value into a variable and return it after the '
                    'loop.'
                )


def _lift(stmts):
    """Lift returns in a statement list into assignments to RETVAL_NAME.

    Returns a tuple (new_stmts, always_returns) where always_returns indicates
    that control flow always reaches a (lifted) return within these statements.
    """
    new = []
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, gast.Return):
            new.append(_assign(RETVAL_NAME, stmt.value))
            # Anything after an unconditional return is unreachable; drop it.
            return new, True

        if isinstance(stmt, gast.If):
            body, body_ret = _lift(stmt.body)
            orelse, orelse_ret = _lift(stmt.orelse)

            if body_ret or orelse_ret:
                # The statements after this `if` form the fall-through path, reached
                # only when a branch did not return. Move them into the branch(es)
                # that fall through.
                rest, rest_ret = _lift(stmts[idx + 1 :])
                if not body_ret:
                    body = body + rest
                if not orelse_ret:
                    orelse = orelse + rest
                new.append(gast.If(test=stmt.test, body=body or [gast.Pass()], orelse=orelse))
                # After this point everything was folded into the branches.
                always = (body_ret or rest_ret) and (orelse_ret or rest_ret)
                return new, always

            # Neither branch returns: keep the (recursively lifted) if and continue.
            new.append(gast.If(test=stmt.test, body=body or [gast.Pass()], orelse=orelse))
            continue

        new.append(stmt)

    return new, False


class ReturnDesugarer(gast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        _lower_loop_returns(node)

        returns = [n for n in gast.walk(node) if isinstance(n, gast.Return)]
        # Already single-exit (one return, at the end): nothing to do.
        if len(returns) <= 1 and (not node.body or isinstance(node.body[-1], gast.Return)):
            return node

        _reject_returns_in_loops(node)

        lifted, always = _lift(node.body)
        lifted.append(gast.Return(value=_load(RETVAL_NAME)))
        node.body = lifted
        return node


def desugar_returns(node):
    """Rewrite early returns into single-exit form."""
    node = ReturnDesugarer().visit(node)
    gast.fix_missing_locations(node)
    return node
