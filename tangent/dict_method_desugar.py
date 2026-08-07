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
"""Desugar dictionary method calls into supported operations.

Tangent cannot differentiate through Python's dict methods directly, but the
common value-access method ``.get()`` maps cleanly onto subscripting, which is
fully supported:

    d.get(k)           ->  d[k]
    d.get(k, default)  ->  d[k] if k in d else default

The rewrite runs before call resolution and the language fence, so the rest of
the pipeline only ever sees a subscript (and, for the two-argument form, a
membership test and a conditional expression, both of which are supported in
reverse mode).

Only ``.get`` calls with one or two positional arguments (and no keyword or
starred arguments) are rewritten. That is the idiomatic dict-access form; other
uses are left untouched.
"""
from __future__ import absolute_import

import copy

import gast


class DictMethodDesugarer(gast.NodeTransformer):
  """Rewrite ``d.get(...)`` calls into subscripts / conditional expressions."""

  def visit_Call(self, node):
    # Process nested calls first (e.g. outer(d.get(k))).
    self.generic_visit(node)

    func = node.func
    if (isinstance(func, gast.Attribute) and func.attr == 'get' and
        not node.keywords and
        not any(isinstance(a, gast.Starred) for a in node.args) and
        len(node.args) in (1, 2)):
      receiver = func.value
      key = node.args[0]

      # d.get(k) -> d[k]
      subscript = gast.Subscript(
          value=copy.deepcopy(receiver),
          slice=copy.deepcopy(key),
          ctx=gast.Load())

      if len(node.args) == 1:
        return gast.copy_location(subscript, node)

      # d.get(k, default) -> d[k] if k in d else default
      default = node.args[1]
      ifexp = gast.IfExp(
          test=gast.Compare(
              left=copy.deepcopy(key),
              ops=[gast.In()],
              comparators=[copy.deepcopy(receiver)]),
          body=subscript,
          orelse=default)
      return gast.copy_location(ifexp, node)

    return node


def desugar_dict_methods(node):
  """Rewrite supported dictionary method calls in an AST."""
  node = DictMethodDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
