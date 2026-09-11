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
"""Strip `with tangent.checkpoint():` blocks into annotated loops.

The dataflow analyses (activity, defined) treat a `with` statement as one
opaque node, so a loop kept inside one would never be differentiated. This
pass unwraps the block early - replacing the `with` by its body - and marks
the loops it contained with a `force_checkpoint` annotation, which
`reverse_ad._should_checkpoint_loop` honors (no length threshold, runtime
bounds allowed: the user opted in explicitly).

Recognized syntactically as `with tangent.checkpoint():` or, when imported
directly, `with checkpoint():`. `with insert_grad_of(...) as ...:` blocks are
left untouched.
"""

from __future__ import absolute_import

import gast

from tangent import annotations as anno


def _is_checkpoint_call(expr):
    if not isinstance(expr, gast.Call) or expr.args or expr.keywords:
        return False
    func = expr.func
    if isinstance(func, gast.Attribute):
        return (
            func.attr == 'checkpoint'
            and isinstance(func.value, gast.Name)
            and func.value.id == 'tangent'
        )
    return isinstance(func, gast.Name) and func.id == 'checkpoint'


class CheckpointAnnotationStripper(gast.NodeTransformer):
    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1 and node.items[0].optional_vars is None:
            if _is_checkpoint_call(node.items[0].context_expr):
                for stmt in node.body:
                    if isinstance(stmt, (gast.For, gast.While)):
                        anno.setanno(stmt, 'force_checkpoint', True, safe=False)
                return node.body
        return node


def strip_checkpoint_annotations(node):
    node = CheckpointAnnotationStripper().visit(node)
    gast.fix_missing_locations(node)
    return node
