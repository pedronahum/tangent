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
"""Desugar set and dict comprehensions by unrolling constant ranges.

Building a collection element-by-element cannot be differentiated correctly (the
per-iteration binding is not differentiated, which silently drops gradients).
But when a comprehension ranges over a compile-time-constant iterable - which is
the only loop form Tangent supports anyway - it can be fully unrolled into a
plain literal, substituting the loop variable with each concrete value:

    {i: x ** i for i in range(1, 3)}   ->   {1: x ** 1, 2: x ** 2}
    {x * i for i in range(3)}          ->   {x * 0, x * 1, x * 2}

The resulting dict/set literal is handled by the normal machinery (dict literals
differentiate through their values; set literals are non-differentiable
membership collections).

Only single-generator comprehensions over a constant ``range(...)`` or a
list/tuple literal, with a plain-name target and no ``if`` filters, are
unrolled. Anything else is left as-is and rejected later by the language fence,
so nothing is ever silently mis-differentiated.
"""
from __future__ import absolute_import

import copy

import gast

from tangent import quoting


class _NameSubstituter(gast.NodeTransformer):
  """Replace reads of a single variable name with a given expression."""

  def __init__(self, name, replacement):
    self.name = name
    self.replacement = replacement

  def visit_Name(self, node):
    if isinstance(node.ctx, gast.Load) and node.id == self.name:
      return copy.deepcopy(self.replacement)
    return node


class _FreeNameFinder(gast.NodeVisitor):
  """Detects whether an expression still reads any (unbound) name."""

  def __init__(self):
    self.found = False

  def visit_Name(self, node):
    if isinstance(node.ctx, gast.Load):
      self.found = True
    self.generic_visit(node)


def _constant_filters_keep(ifs, var, value_node):
  """Decide at compile time whether a listcomp element survives its filters.

  Substitutes the loop variable with the concrete value and evaluates each
  `if` clause. Returns True/False when every clause is a closed constant
  expression, or None when a clause still references a name (or fails to
  evaluate) and therefore cannot be decided statically; the caller then leaves
  the comprehension for a later pass instead of guessing.
  """
  for cond in ifs:
    bound = _NameSubstituter(var, value_node).visit(copy.deepcopy(cond))
    finder = _FreeNameFinder()
    finder.visit(bound)
    if finder.found:
      return None
    try:
      keep = bool(eval(quoting.to_source(bound), {'__builtins__': {}}, {}))
    except Exception:
      return None
    if not keep:
      return False
  return True


def _constant_iter_values(iter_node):
  """Return the list of value-nodes a constant iterable yields, or None."""
  # range(...) with constant integer arguments.
  if (isinstance(iter_node, gast.Call) and
      isinstance(iter_node.func, gast.Name) and
      iter_node.func.id == 'range' and not iter_node.keywords):
    args = iter_node.args
    if args and all(isinstance(a, gast.Constant) and isinstance(a.value, int)
                    for a in args):
      return [gast.Constant(value=i, kind=None) for i in range(*[a.value for a in args])]
    return None
  # A list or tuple literal.
  if isinstance(iter_node, (gast.List, gast.Tuple)):
    return [copy.deepcopy(e) for e in iter_node.elts]
  return None


class ComprehensionUnroller(gast.NodeTransformer):
  """Unroll set/dict comprehensions over constant iterables into literals."""

  def _unrollable_generator(self, node):
    if len(node.generators) != 1:
      return None
    gen = node.generators[0]
    if gen.ifs or getattr(gen, 'is_async', 0):
      return None
    if not isinstance(gen.target, gast.Name):
      return None
    values = _constant_iter_values(gen.iter)
    if not values:  # None or empty: leave for the fence to reject.
      return None
    return gen.target.id, values

  def visit_SetComp(self, node):
    self.generic_visit(node)
    info = self._unrollable_generator(node)
    if info is None:
      return node
    var, values = info
    elts = [_NameSubstituter(var, v).visit(copy.deepcopy(node.elt))
            for v in values]
    return gast.copy_location(gast.Set(elts=elts), node)

  def visit_ListComp(self, node):
    """Unroll a list comprehension over a constant iterable into a list
    literal, so that it differentiates like the literal it is.

    Unlike set/dict comprehensions, `if` filters are supported when they can be
    decided at compile time (i.e. once the loop variable is substituted they are
    closed constant expressions). A filter that still references a name cannot
    be decided statically, so the comprehension is left as-is for a later pass.
    """
    self.generic_visit(node)
    if len(node.generators) != 1:
      return node
    gen = node.generators[0]
    if getattr(gen, 'is_async', 0):
      return node
    if not isinstance(gen.target, gast.Name):
      return node
    values = _constant_iter_values(gen.iter)
    if not values:  # None or empty: leave for a later pass to handle/reject.
      return node
    var = gen.target.id
    elts = []
    for v in values:
      if gen.ifs:
        keep = _constant_filters_keep(gen.ifs, var, v)
        if keep is None:
          return node  # filter not statically decidable; don't guess.
        if not keep:
          continue
      elts.append(_NameSubstituter(var, v).visit(copy.deepcopy(node.elt)))
    # A parsed list literal always carries ctx=Load(); forward mode reads it, so
    # the synthesized literal must set it too.
    return gast.copy_location(gast.List(elts=elts, ctx=gast.Load()), node)

  def visit_DictComp(self, node):
    self.generic_visit(node)
    info = self._unrollable_generator(node)
    if info is None:
      return node
    var, values = info
    keys = [_NameSubstituter(var, v).visit(copy.deepcopy(node.key))
            for v in values]
    vals = [_NameSubstituter(var, v).visit(copy.deepcopy(node.value))
            for v in values]
    return gast.copy_location(gast.Dict(keys=keys, values=vals), node)


def desugar_comprehensions(node):
  """Unroll set/dict comprehensions over constant iterables."""
  node = ComprehensionUnroller().visit(node)
  gast.fix_missing_locations(node)
  return node
