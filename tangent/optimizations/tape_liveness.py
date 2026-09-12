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
"""Tape-liveness: store only the shape of primals the adjoint reads for shape.

Reverse mode saves each reassigned primal on the tape so the adjoint can read
it back. Many of those reads are *shape-only*: the restored value flows into
``tangent.unbroadcast(grad, v)`` as the broadcast target, or into
``tangent.init_grad(v)`` to reset a gradient accumulator - in both cases only
``v``'s shape and dtype are consulted, never its data. Saving the whole array
for that is wasteful; the tape entry can be O(ndim) instead of O(size).

This pass rewrites such a variable's push from ``push(_stack, v, id)`` to
``push(_stack, tangent.taped_shape(v), id)``. Nothing else changes: the pop and
the ``unbroadcast`` / ``init_grad`` calls stay textually identical, and
Tangent's runtime type dispatch routes the restored ``TapedShape`` through the
registered shape-only initializer and the NumPy unbroadcaster (which reads its
``like`` through ``numpy.shape``). The gradient is unchanged - ``unbroadcast``
reduces to the same shape and ``init_grad`` yields the same zero - only the
bytes on the tape shrink.

Soundness rests on reaching-definition analysis, which distinguishes the
*restored* value from the live primal that shares its name. Only loads that a
``v = pop(...)`` definition reaches consume the tape entry; the primal's own
uses of ``v`` (reached by the forward assignment, which precedes every pop) are
irrelevant and left untouched. A variable is converted only when every
pop-reached use of it is one of

  * the second positional argument of ``unbroadcast(...)`` (the shape target),
  * the first positional argument of ``init_grad(...)`` (the reset), or
  * a dead self-copy ``t = v`` whose target ``t`` is never used.

Any other pop-reached use - arithmetic, a reduction's shape query, a container
op - means the data itself is needed, so the store is left intact. The pass is
opt-in (``optimizations={'tape_liveness': True}``) and targets the NumPy path.

Note on `ctx`: mid-pipeline ASTs do not carry reliable `Load`/`Store` context
on Name nodes (Tangent's transforms leave assignment targets with the default
`Load` ctx; `to_source` ignores ctx, so this never shows in output). Use and
definition are therefore determined structurally - by a Name's position in its
parent - never from `node.ctx`.
"""

from __future__ import absolute_import

import gast

from tangent import annotations as anno
from tangent import cfg

_REACHING_LABELS = ('definitions_in', 'definitions_out', 'definitions_gen', 'definitions_kill')


def _is_call_to(node, *names):
    if not isinstance(node, gast.Call):
        return False
    func = node.func
    if isinstance(func, gast.Attribute):
        return func.attr in names
    if isinstance(func, gast.Name):
        return func.id in names
    return False


def _is_pop_assign(stmt):
    return (
        isinstance(stmt, gast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], gast.Name)
        and _is_call_to(stmt.value, 'pop')
    )


def _store_name_ids(func):
    """id() of every Name node that is a pure assignment target (a definition).

    Determined structurally, since mid-pipeline `ctx` is unreliable. AugAssign
    targets are intentionally excluded: `x += y` also *reads* x, so treating
    that Name as a use is the conservative (soundness-preserving) choice.
    """
    ids = set()

    def mark(target):
        if isinstance(target, gast.Name):
            ids.add(id(target))
        elif isinstance(target, (gast.Tuple, gast.List)):
            for elt in target.elts:
                mark(elt)
        elif isinstance(target, gast.Starred):
            mark(target.value)
        # Attribute/Subscript targets read their `.value`, so their inner Names
        # are uses, not definitions - do not mark them.

    for node in gast.walk(func):
        if isinstance(node, gast.Assign):
            for t in node.targets:
                mark(t)
        elif isinstance(node, (gast.For, gast.AsyncFor)):
            mark(node.target)
        elif isinstance(node, gast.comprehension):
            mark(node.target)
        elif isinstance(node, gast.withitem) and node.optional_vars is not None:
            mark(node.optional_vars)
    return ids


def _used_names(func, store_ids):
    return {n.id for n in gast.walk(func) if isinstance(n, gast.Name) and id(n) not in store_ids}


def _dead_targets(func, used):
    dead = set()
    for stmt in gast.walk(func):
        if (
            isinstance(stmt, gast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], gast.Name)
            and stmt.targets[0].id not in used
        ):
            dead.add(stmt.targets[0].id)
    return dead


def _use_is_shape_only(parent, name_node, dead_targets):
    if isinstance(parent, gast.Call):
        if (_is_call_to(parent, 'unbroadcast') and len(parent.args) >= 2
                and parent.args[1] is name_node):
            return True
        if _is_call_to(parent, 'init_grad') and parent.args and parent.args[0] is name_node:
            return True
        return False
    # A dead self-copy `t = v` (t never used) does not need v's value.
    if (isinstance(parent, gast.Assign) and parent.value is name_node
            and len(parent.targets) == 1 and isinstance(parent.targets[0], gast.Name)
            and parent.targets[0].id in dead_targets):
        return True
    return False


def _stmt_uses_shape_only(stmt, name, store_ids, dead_targets):
    """Every use of `name` in `stmt` is shape-only. Returns (all_ok, saw_use)."""
    saw = False
    for parent in gast.walk(stmt):
        for _field, child in gast.iter_fields(parent):
            for item in (child if isinstance(child, list) else [child]):
                if not (isinstance(item, gast.Name) and item.id == name
                        and id(item) not in store_ids):
                    continue
                saw = True
                if not _use_is_shape_only(parent, item, dead_targets):
                    return False, saw
    return True, saw


def _convertible_names(func):
    """Names whose every pop-reached use is shape-only; safe to store as shape."""
    # Clear stale reaching-definition labels (their `def` nodes may point at
    # statements an earlier DCE pass has replaced) so a fresh analysis runs.
    for stmt in gast.walk(func):
        for label in _REACHING_LABELS:
            if anno.hasanno(stmt, label):
                anno.delanno(stmt, label)
    cfg.forward(func, cfg.ReachingDefinitions())

    store_ids = _store_name_ids(func)
    used = _used_names(func, store_ids)
    dead_targets = _dead_targets(func, used)

    popped = {stmt.targets[0].id for stmt in gast.walk(func) if _is_pop_assign(stmt)}
    if not popped:
        return set()

    rejected = set()
    shape_only_seen = set()
    for stmt in gast.walk(func):
        if not anno.hasanno(stmt, 'definitions_in'):
            continue
        reaching = anno.getanno(stmt, 'definitions_in')
        pop_reached = {name for (name, def_stmt) in reaching
                       if name in popped and _is_pop_assign(def_stmt)}
        for name in pop_reached:
            if name in rejected:
                continue
            ok, saw = _stmt_uses_shape_only(stmt, name, store_ids, dead_targets)
            if not ok:
                rejected.add(name)
            elif saw:
                shape_only_seen.add(name)
    return (popped & shape_only_seen) - rejected


def store_shapes_only(module):
    """Convert shape-only tape stores to shape carriers. Returns True if changed.

    Operates on the whole module: a push may live in the primal function and
    its pop in the adjoint (split motion), so both are analyzed together. A
    name is converted only if it is convertible in every function that pops it.
    """
    funcs = [f for f in module.body if isinstance(f, gast.FunctionDef)]
    if not funcs:
        return False

    convertible = None
    for func in funcs:
        names = _convertible_names(func)
        convertible = names if convertible is None else (convertible & names)
    if not convertible:
        return False

    changed = False
    for func in funcs:
        for node in gast.walk(func):
            if (_is_call_to(node, 'push') and len(node.args) >= 2
                    and isinstance(node.args[1], gast.Name) and node.args[1].id in convertible):
                v = node.args[1]
                node.args[1] = gast.Call(
                    func=gast.Attribute(
                        value=gast.Name(id='tangent', ctx=gast.Load(), annotation=None,
                                        type_comment=None),
                        attr='taped_shape', ctx=gast.Load()),
                    args=[gast.Name(id=v.id, ctx=gast.Load(), annotation=None, type_comment=None)],
                    keywords=[],
                )
                changed = True
    return changed
