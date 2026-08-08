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
"""Desugar ``for a, b in zip(xs, ys)`` loops into indexed range loops.

Tangent iterates with a ``range(len(...))`` index and explicit subscripting, so
a tuple-target ``zip`` loop is rewritten into that supported form before
differentiation::

    for a, b in zip(xs, ys):    ->    for i in range(len(xs)):
        <body>                            a = xs[i]
                                          b = ys[i]
                                          <body>

The loop length is taken from the first sequence (as in most numeric code, the
zipped sequences have equal length; a shorter later sequence raises an
IndexError rather than silently truncating). Non-name sequences (e.g. list
literals) are hoisted to temporaries so each is evaluated once.

The rewrite runs before call resolution, so the generated ``range``/``len``
calls are resolved and annotated like any other, and each element read
``a = xs[i]`` flows through the existing subscript machinery.
"""
from __future__ import absolute_import

import copy

import gast


def _name(name, ctx):
  return gast.Name(id=name, ctx=ctx, annotation=None)


def _assign(target, value):
  return gast.Assign(targets=[target], value=value)


class ZipDesugarer(gast.NodeTransformer):

  def __init__(self):
    self._counter = 0

  def _fresh(self):
    name = '_zip%d' % self._counter
    self._counter += 1
    return name

  def visit_For(self, node):
    self.generic_visit(node)

    it = node.iter
    if not (isinstance(it, gast.Call) and isinstance(it.func, gast.Name) and
            it.func.id == 'zip' and not it.keywords and it.args and
            not any(isinstance(a, gast.Starred) for a in it.args) and
            isinstance(node.target, gast.Tuple) and
            len(node.target.elts) == len(it.args) and
            all(isinstance(t, gast.Name) for t in node.target.elts)):
      return node

    # Capture the loop targets before rewriting node.target.
    targets = list(node.target.elts)

    # Evaluate each sequence once (a plain name can be referenced directly).
    prelude = []
    seq_refs = []
    for seq in it.args:
      if isinstance(seq, gast.Name):
        seq_refs.append(seq)
      else:
        tmp = self._fresh()
        prelude.append(_assign(_name(tmp, gast.Store()), seq))
        seq_refs.append(_name(tmp, gast.Load()))

    index = self._fresh()
    node.target = _name(index, gast.Store())
    node.iter = gast.Call(
        func=_name('range', gast.Load()),
        args=[gast.Call(func=_name('len', gast.Load()),
                        args=[copy.deepcopy(seq_refs[0])], keywords=[])],
        keywords=[])

    element_reads = [
        _assign(_name(target.id, gast.Store()),
                gast.Subscript(value=copy.deepcopy(ref),
                               slice=_name(index, gast.Load()),
                               ctx=gast.Load()))
        for target, ref in zip(targets, seq_refs)]
    node.body = element_reads + node.body

    if prelude:
      return prelude + [node]
    return node


def desugar_zip(node):
  """Rewrite tuple-target zip loops into indexed range loops."""
  node = ZipDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
