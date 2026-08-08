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
"""Desugar ``for i, v in enumerate(seq)`` loops into indexed range loops.

Tangent iterates with a `range(len(...))` index and explicit subscripting; a
tuple-target `enumerate` loop is rewritten into that supported form before
differentiation::

    for i, v in enumerate(seq):
        <body>
    # becomes
    for i in range(len(seq)):
        v = seq[i]
        <body>

A non-name iterable (e.g. a list literal) is hoisted to a temporary first so it
is evaluated once. A start offset (`enumerate(seq, start)`) is supported by
iterating positions and computing the index as `start + position`.

The rewrite runs before call resolution, so the generated `range`/`len` calls
are resolved and annotated like any other.
"""
from __future__ import absolute_import

import copy

import gast


def _name(name, ctx):
  return gast.Name(id=name, ctx=ctx, annotation=None)


def _assign(target, value):
  return gast.Assign(targets=[target], value=value)


def _range_len(iterable):
  return gast.Call(
      func=_name('range', gast.Load()),
      args=[gast.Call(func=_name('len', gast.Load()),
                      args=[copy.deepcopy(iterable)], keywords=[])],
      keywords=[])


class EnumerateDesugarer(gast.NodeTransformer):

  def __init__(self):
    self._counter = 0

  def _fresh(self):
    name = '_enum%d' % self._counter
    self._counter += 1
    return name

  def visit_For(self, node):
    self.generic_visit(node)

    it = node.iter
    if not (isinstance(it, gast.Call) and isinstance(it.func, gast.Name) and
            it.func.id == 'enumerate' and not it.keywords and
            1 <= len(it.args) <= 2 and
            isinstance(node.target, gast.Tuple) and
            len(node.target.elts) == 2):
      return node

    idx_target, val_target = node.target.elts
    if not (isinstance(idx_target, gast.Name) and
            isinstance(val_target, gast.Name)):
      return node

    iterable = it.args[0]
    start = it.args[1] if len(it.args) == 2 else None

    prelude = []
    if isinstance(iterable, gast.Name):
      iter_ref = iterable
    else:
      # Evaluate the iterable once.
      tmp = self._fresh()
      prelude.append(_assign(_name(tmp, gast.Store()), iterable))
      iter_ref = _name(tmp, gast.Load())

    if start is None:
      # for i in range(len(iter)): v = iter[i]; <body>
      node.target = _name(idx_target.id, gast.Store())
      node.iter = _range_len(iter_ref)
      val_assign = _assign(
          _name(val_target.id, gast.Store()),
          gast.Subscript(value=copy.deepcopy(iter_ref),
                         slice=_name(idx_target.id, gast.Load()),
                         ctx=gast.Load()))
      node.body = [val_assign] + node.body
    else:
      # for pos in range(len(iter)): i = start + pos; v = iter[pos]; <body>
      pos = self._fresh()
      node.target = _name(pos, gast.Store())
      node.iter = _range_len(iter_ref)
      idx_assign = _assign(
          _name(idx_target.id, gast.Store()),
          gast.BinOp(left=copy.deepcopy(start), op=gast.Add(),
                     right=_name(pos, gast.Load())))
      val_assign = _assign(
          _name(val_target.id, gast.Store()),
          gast.Subscript(value=copy.deepcopy(iter_ref),
                         slice=_name(pos, gast.Load()), ctx=gast.Load()))
      node.body = [idx_assign, val_assign] + node.body

    if prelude:
      return prelude + [node]
    return node


def desugar_enumerate(node):
  """Rewrite tuple-target enumerate loops into indexed range loops."""
  node = EnumerateDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
