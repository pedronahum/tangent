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
"""Desugar list mutation methods into differentiable rebindings.

``xs.append(v)`` mutates ``xs`` in place, which the activity analysis and the
tape cannot see: the statement used to be silently treated as non-differentiable
and gradients through the list were dropped (a zero gradient, the worst failure
mode). Rewriting the mutation as a rebinding makes it an ordinary assignment
that every downstream pass already understands::

    xs.append(v)    ->   xs = tangent.list_append(xs, v)
    v = xs.pop()    ->   v = tangent.list_last(xs)
                         xs = tangent.list_init(xs)
    xs.pop()        ->   xs = tangent.list_init(xs)

`tangent.list_append` appends in place and returns the list (O(1)); the tape's
`push` stores lists as slices, so earlier values are still restored correctly.
The three primitives have adjoints and tangents written in terms of each other
(see grads.py / tangents.py), so the rewrite differentiates to any order.

Only calls on a plain name are rewritten - ``obj.attr.append(v)`` or
``xs[i].append(v)`` mutate through an alias the rebinding could not express.
Those, and the other in-place list mutators (``extend``, ``insert``,
``remove``, ``sort``, ``reverse``, ``clear``), are left in place for the fence
to reject with a clear error: before this pass existed they silently dropped
gradients, which is strictly worse than an error. ``.pop(k)`` calls *with*
arguments are left untouched (that form is dict.pop / list.pop(i), neither of
which is supported for differentiation).

This pass must run before ``resolve_calls`` so the injected ``tangent.*`` calls
get resolved against the user's namespace (every caller of ``tangent.grad`` has
``tangent`` imported).
"""

from __future__ import absolute_import

import gast


def _tangent_call(helper, args):
    """Build a ``tangent.<helper>(*args)`` Call node."""
    return gast.Call(
        func=gast.Attribute(
            value=gast.Name(id='tangent', ctx=gast.Load(), annotation=None),
            attr=helper,
            ctx=gast.Load(),
        ),
        args=args,
        keywords=[],
    )


def _match_method_call(node, method, num_args):
    """Return the object Name node if `node` is ``name.method(<num_args>)``."""
    if not isinstance(node, gast.Call):
        return None
    func = node.func
    if not isinstance(func, gast.Attribute) or func.attr != method:
        return None
    if not isinstance(func.value, gast.Name):
        return None
    if len(node.args) != num_args or node.keywords:
        return None
    return func.value


def _load(name):
    return gast.Name(id=name, ctx=gast.Load(), annotation=None)


def _store(name):
    return gast.Name(id=name, ctx=gast.Store(), annotation=None)


class ListMethodDesugarer(gast.NodeTransformer):
    """Rewrite ``xs.append(v)`` / ``xs.pop()`` statements into rebindings."""

    def visit_Expr(self, node):
        self.generic_visit(node)
        obj = _match_method_call(node.value, 'append', 1)
        if obj is not None:
            new = gast.Assign(
                targets=[_store(obj.id)],
                value=_tangent_call('list_append', [_load(obj.id), node.value.args[0]]),
            )
            return gast.copy_location(new, node)
        obj = _match_method_call(node.value, 'pop', 0)
        if obj is not None:
            new = gast.Assign(
                targets=[_store(obj.id)],
                value=_tangent_call('list_init', [_load(obj.id)]),
            )
            return gast.copy_location(new, node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        obj = _match_method_call(node.value, 'pop', 0)
        if obj is None or len(node.targets) != 1:
            return node
        last = gast.Assign(
            targets=node.targets,
            value=_tangent_call('list_last', [_load(obj.id)]),
        )
        trim = gast.Assign(
            targets=[_store(obj.id)],
            value=_tangent_call('list_init', [_load(obj.id)]),
        )
        return [gast.copy_location(last, node), gast.copy_location(trim, node)]


def desugar_list_methods(node):
    """Rewrite list append/pop statements in an AST."""
    node = ListMethodDesugarer().visit(node)
    gast.fix_missing_locations(node)
    return node
