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
"""Desugar chained assignments into single-target assignments.

Tangent's core transforms only handle assignments with a single target, so a
chained assignment previously raised "no support for chained assignment" (and
forward mode would have silently dropped the extra targets). Since ``a = b = e``
binds every target to the same value, it is rewritten into a first assignment
followed by copies from that target::

    a = b = c = expr   ->   a = expr
                            b = a
                            c = a

Only chained assignments whose targets are all plain names are rewritten (the
common case). Anything else is left untouched.
"""
from __future__ import absolute_import

import gast


class ChainedAssignDesugarer(gast.NodeTransformer):
  """Rewrite ``a = b = e`` into ``a = e; b = a``."""

  def visit_Assign(self, node):
    self.generic_visit(node)
    if len(node.targets) <= 1:
      return node
    if not all(isinstance(t, gast.Name) for t in node.targets):
      # Leave chained assignments with non-name targets alone.
      return node

    first = node.targets[0]
    stmts = [gast.Assign(targets=[first], value=node.value)]
    for target in node.targets[1:]:
      stmts.append(gast.Assign(
          targets=[target],
          value=gast.Name(id=first.id, ctx=gast.Load(), annotation=None)))
    for stmt in stmts:
      gast.copy_location(stmt, node)
    return stmts


def desugar_chained_assignments(node):
  """Rewrite chained (multi-target) assignments in an AST."""
  node = ChainedAssignDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
