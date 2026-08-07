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

Tangent cannot differentiate through Python's dict methods directly, but two
common patterns map cleanly onto operations that are already supported, so they
are rewritten before call resolution and the language fence:

  * ``.get()`` maps onto subscripting::

        d.get(k)           ->  d[k]
        d.get(k, default)  ->  d[k] if k in d else default

  * ``sum(d.values())`` maps onto a fold of the values, since a dict literal has
    statically known keys::

        sum({'a': p, 'b': q}.values())  ->  p + q
        sum(d.values())                 ->  d['a'] + d['b']   # for d = {'a':..,'b':..}

The ``sum(...)`` fold only fires when the dict's keys are known unambiguously:
either the receiver is a dict literal, or it is a local variable assigned exactly
one dict literal (with constant keys) and never reassigned. Anything else is left
untouched.

Only ``.get`` calls with one or two positional arguments (and no keyword or
starred arguments) are rewritten. Other dict methods are left untouched.
"""
from __future__ import absolute_import

import copy

import gast


def _is_dict_literal_with_const_keys(node):
  return (isinstance(node, gast.Dict) and node.keys and
          all(isinstance(k, gast.Constant) for k in node.keys))


class _StaticDictCollector(gast.NodeVisitor):
  """Find local variables bound to exactly one constant-keyed dict literal.

  A variable qualifies only if *every* assignment to it (there must be exactly
  one, and it must be a plain ``name = {...}``) is such a dict literal. This
  keeps the ``sum(d.values())`` rewrite from firing on ambiguous or reassigned
  dictionaries.
  """

  def __init__(self):
    self.assign_count = {}
    self.keys = {}

  def visit_Assign(self, node):
    self.generic_visit(node)
    if len(node.targets) == 1 and isinstance(node.targets[0], gast.Name):
      name = node.targets[0].id
      self.assign_count[name] = self.assign_count.get(name, 0) + 1
      if _is_dict_literal_with_const_keys(node.value):
        self.keys[name] = list(node.value.keys)
      else:
        # A non-qualifying assignment disqualifies the name.
        self.keys.pop(name, None)

  def static_keys(self):
    return {name: keys for name, keys in self.keys.items()
            if self.assign_count.get(name) == 1}


class DictMethodDesugarer(gast.NodeTransformer):
  """Rewrite supported dict method calls into subscripts / folds."""

  def __init__(self):
    self._static_keys = {}

  def visit_FunctionDef(self, node):
    collector = _StaticDictCollector()
    for stmt in node.body:
      collector.visit(stmt)
    saved = self._static_keys
    self._static_keys = collector.static_keys()
    self.generic_visit(node)
    self._static_keys = saved
    return node

  def _sum_of_values(self, node):
    """Rewrite sum(<dict>.values()) into a fold, or return None if unhandled."""
    if not (isinstance(node.func, gast.Name) and node.func.id == 'sum'):
      return None
    if node.keywords or not (1 <= len(node.args) <= 2):
      return None
    inner = node.args[0]
    if not (isinstance(inner, gast.Call) and
            isinstance(inner.func, gast.Attribute) and
            inner.func.attr == 'values' and
            not inner.args and not inner.keywords):
      return None

    receiver = inner.func.value
    # Determine the value expressions to fold.
    if _is_dict_literal_with_const_keys(receiver):
      value_exprs = [copy.deepcopy(v) for v in receiver.values]
    elif isinstance(receiver, gast.Name) and receiver.id in self._static_keys:
      keys = self._static_keys[receiver.id]
      value_exprs = [
          gast.Subscript(value=copy.deepcopy(receiver),
                         slice=copy.deepcopy(k), ctx=gast.Load())
          for k in keys]
    else:
      return None

    # Optional start value: sum(values, start).
    if len(node.args) == 2:
      fold = node.args[1]
    else:
      fold = value_exprs.pop(0)
    for expr in value_exprs:
      fold = gast.BinOp(left=fold, op=gast.Add(), right=expr)
    return gast.copy_location(fold, node)

  def visit_Call(self, node):
    # Process nested calls first (e.g. outer(d.get(k))).
    self.generic_visit(node)

    # sum(d.values()) -> d[k1] + d[k2] + ...
    folded = self._sum_of_values(node)
    if folded is not None:
      return folded

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
