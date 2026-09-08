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
"""Lower ``break``/``continue`` into guard flags the AD core understands.

Reverse-mode AD records one tape entry per completed loop iteration; a real
``break`` exits mid-iteration, so the backward pass would replay the wrong
number of iterations (this used to *miscompute* gradients, then was rejected).
The lowering removes the early exit instead of trying to differentiate it:

``continue`` becomes a per-iteration skip flag - the remainder of the body is
guarded, and the loop shape is preserved::

    for v in xs:                      for v in xs:
        A                                 _skip0 = False
        if c:                             A
            continue          ->          if c:
        B                                     _skip0 = True
                                          if not _skip0:
                                              B

``break`` additionally needs to stop the loop, so the loop gains a break flag.
A ``while`` folds it into its condition (its own condition may never become
false after the break point, so the loop must genuinely exit early)::

    while c0:                         _brk0 = False
        A                             while c0 and not _brk0:
        if c:             ->              _skip0 = False
            break                         A
        B                                 if c:
                                              _brk0 = True
                                              _skip0 = True
                                          if not _skip0:
                                              B

A ``for`` keeps its shape - preserving every existing For code path in both AD
modes, including the active-sequence rewrite - and instead wraps its whole body
in ``if not _brk0:``. The loop runs to its natural bound with empty iterations
after the break: exact semantics, at the cost of skipped spins for an early
break out of a long range::

    for v in xs:                      _brk0 = False
        A                             for v in xs:
        if c:             ->              if not _brk0:
            break                             _skip0 = False
        B                                     A
                                              if c:
                                                  _brk0 = True
                                                  _skip0 = True
                                              if not _skip0:
                                                  B

Every construct produced (boolean operators, if-branches, flag assignments) is
one the AD core already differentiates in both modes, so gradients through
early-exit loops are exact rather than "replayed wrong". Loops are processed
innermost-first, so a ``break`` always binds to its nearest enclosing loop.
Loops with an ``else`` clause are left untouched (``for``/``else`` semantics
depend on how the loop exited; the fence rejects the construct).

This pass runs after the iterator desugarings (enumerate/zip/comprehensions
produce loops whose bodies may contain user ``break``/``continue``) and before
the fence.
"""

from __future__ import absolute_import

import gast


def _assign_flag(name, value):
    return gast.Assign(
        targets=[gast.Name(id=name, ctx=gast.Store(), annotation=None)],
        value=gast.Constant(value=value, kind=None),
    )


def _load(name):
    return gast.Name(id=name, ctx=gast.Load(), annotation=None)


def _contains_exit(stmts):
    """Whether any statement (recursively) holds a Break or Continue.

    Nested loops have already been lowered when this is asked, so any exit
    found binds to the loop whose body `stmts` is.
    """
    for stmt in stmts:
        for n in gast.walk(stmt):
            if isinstance(n, (gast.Break, gast.Continue)):
                return True
    return False


class LoopExitLowerer(object):
    """Rewrite loops containing break/continue, innermost first."""

    def __init__(self):
        self._counter = 0

    # -- statement-list plumbing ------------------------------------------

    def process_body(self, body):
        out = []
        for stmt in body:
            out.extend(self.process_stmt(stmt))
        return out

    def process_stmt(self, stmt):
        # Recurse first so inner loops consume their own break/continue.
        for field in ('body', 'orelse', 'finalbody'):
            sub = getattr(stmt, field, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], gast.stmt):
                setattr(stmt, field, self.process_body(sub))
        if isinstance(stmt, (gast.For, gast.While)) and not stmt.orelse:
            if _contains_exit(stmt.body):
                return self._lower_loop(stmt)
        return [stmt]

    # -- the lowering ------------------------------------------------------

    def _lower_loop(self, loop):
        n = self._counter
        self._counter += 1
        skip = '_skip%d' % n
        brk = '_brk%d' % n

        has_break = any(
            isinstance(node, gast.Break) for stmt in loop.body for node in gast.walk(stmt)
        )
        body = [_assign_flag(skip, False)]
        body += self._guard(loop.body, skip, brk)

        if not has_break:
            # continue only: nothing stops the loop, so only the per-iteration
            # skip flag is needed. The loop shape (and the For machinery,
            # including the active-sequence rewrite) is preserved.
            loop.body = [gast.copy_location(s, loop) for s in body]
            return [loop]

        if isinstance(loop, gast.While):
            # Fold the break flag into the condition so the loop still exits
            # early - essential for a `while`, whose own condition may never
            # become false after the break point.
            loop.test = gast.BoolOp(
                op=gast.And(),
                values=[loop.test, gast.UnaryOp(op=gast.Not(), operand=_load(brk))],
            )
            loop.body = body
        else:
            # A `for` keeps its shape (and thus every existing For code path,
            # in both AD modes); the whole body is wrapped in `if not _brk:`
            # so iterations after the break do nothing. The loop still runs to
            # its natural bound with empty iterations - exact semantics, at
            # the cost of skipped spins for early breaks out of long ranges.
            loop.body = [
                gast.If(
                    test=gast.UnaryOp(op=gast.Not(), operand=_load(brk)),
                    body=body,
                    orelse=[],
                )
            ]
        stmts = [_assign_flag(brk, False), loop]
        return [gast.copy_location(s, loop) for s in stmts]

    def _guard(self, stmts, skip, brk):
        """Replace break/continue with flag sets; guard everything downstream
        of a statement that may have set a flag with ``if not _skip:``."""
        out = []
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, gast.Break):
                # Anything after a bare break/continue in the same block is
                # unreachable in the original program and is dropped.
                out.append(_assign_flag(brk, True))
                out.append(_assign_flag(skip, True))
                return out
            if isinstance(stmt, gast.Continue):
                out.append(_assign_flag(skip, True))
                return out
            if isinstance(stmt, gast.If) and _contains_exit([stmt]):
                stmt.body = self._guard(stmt.body, skip, brk)
                if stmt.orelse:
                    stmt.orelse = self._guard(stmt.orelse, skip, brk)
                out.append(stmt)
                rest = self._guard(stmts[i + 1 :], skip, brk)
                if rest:
                    out.append(
                        gast.If(
                            test=gast.UnaryOp(op=gast.Not(), operand=_load(skip)),
                            body=rest,
                            orelse=[],
                        )
                    )
                return out
            out.append(stmt)
        return out


def desugar_loop_exits(node):
    """Lower break/continue in all loops of a module's functions."""
    lowerer = LoopExitLowerer()
    for fdef in node.body:
        if isinstance(fdef, gast.FunctionDef):
            fdef.body = lowerer.process_body(fdef.body)
    gast.fix_missing_locations(node)
    return node
