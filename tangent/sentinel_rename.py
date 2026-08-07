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
"""Rename user variables that collide with the gradient-operator sentinel.

Tangent's template system uses the bare name ``d`` as a sentinel: ``d[x]`` means
"the gradient of x". A user variable literally named ``d`` therefore collides -
``d['a']`` or ``d[1]`` get misread as the gradient operator, producing invalid
code (undefined ``_`` placeholders / DictConstructionError). String keys are
handled directly in the template layer, but numeric keys cannot be, because the
gradient operator legitimately uses ``d[<number>]`` for "gradient of a numeric
constant".

The robust fix is to alpha-rename a user variable named ``d`` to a fresh name
before any differentiation runs, eliminating the collision for every key type.
Only a *local* ``d`` (a parameter or an assigned variable) is renamed; a free /
global ``d`` is left alone, since renaming it would break the reference.
"""
from __future__ import absolute_import

import gast

SENTINEL = 'd'


class _Renamer(gast.NodeTransformer):

  def __init__(self, new_name):
    self.new_name = new_name

  def visit_Name(self, node):
    if node.id == SENTINEL:
      node.id = self.new_name
    return node


def _all_names(fn):
  return set(n.id for n in gast.walk(fn) if isinstance(n, gast.Name))


def _param_names(fn):
  args = fn.args
  fields = (list(args.args) + list(getattr(args, 'posonlyargs', []) or []) +
            list(args.kwonlyargs))
  names = set()
  for a in fields:
    n = getattr(a, 'id', getattr(a, 'arg', None))
    if n is not None:
      names.add(n)
  for extra in (getattr(args, 'vararg', None), getattr(args, 'kwarg', None)):
    if extra is not None:
      n = getattr(extra, 'id', getattr(extra, 'arg', None))
      if n is not None:
        names.add(n)
  return names


def _should_rename(fn):
  """Whether ``d`` is a locally-assigned (non-parameter) variable.

  Parameters are not renamed: Tangent ties the generated code to the original
  function's signature, so renaming a parameter would break argument binding.
  A parameter dict named ``d`` with string keys already works via the template
  layer; a free/global ``d`` is likewise left untouched.
  """
  if SENTINEL in _param_names(fn):
    return False
  for n in gast.walk(fn):
    if (isinstance(n, gast.Name) and n.id == SENTINEL and
        isinstance(n.ctx, gast.Store)):
      return True
  return False


def _fresh_name(existing):
  candidate = 'd_'
  i = 0
  while candidate in existing:
    candidate = 'd_%d' % i
    i += 1
  return candidate


def _top_level_functions(node):
  if isinstance(node, gast.Module):
    return [s for s in node.body if isinstance(s, gast.FunctionDef)]
  if isinstance(node, gast.FunctionDef):
    return [node]
  return [s for s in gast.walk(node) if isinstance(s, gast.FunctionDef)]


def rename_sentinel_vars(node):
  """Rename a local variable named ``d`` so it does not shadow the sentinel."""
  for fn in _top_level_functions(node):
    if _should_rename(fn):
      _Renamer(_fresh_name(_all_names(fn))).visit(fn)
  gast.fix_missing_locations(node)
  return node
