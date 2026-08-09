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
"""Desugar list-argument concatenate/stack calls into varargs form.

``jnp.concatenate([a, b, c], axis)`` and ``jnp.stack([a, b, c], axis)`` take a
*list* of arrays, but Tangent's multi-output adjoint machinery distributes
gradients to *varargs* (see the ``broadcast_arrays`` adjoint). A list argument
cannot be differentiated as a container, so these calls are rewritten before
call resolution into a varargs helper that Tangent can differentiate::

    jnp.concatenate([a, b, c], axis)   ->  tangent.concat_seq(a, b, c, axis)
    jnp.stack([a, b, c], axis)         ->  tangent.stack_seq(a, b, c, axis)

The helpers (defined in the backend extensions) call through to the real
concatenate/stack at runtime and carry varargs adjoints.

Only calls whose first argument is a list *literal*, or a variable that is
assigned exactly once to a list/tuple literal and never mutated, are rewritten.
For any other list (built dynamically, reassigned, or modified in place) the
elements cannot be statically identified, so the call is left untouched and will
raise a clear not-implemented error.
"""
from __future__ import absolute_import

import copy

import gast

# Attribute-name pairs (module alias, function name) that we recognize. We
# match on the trailing attribute name and accept any module alias, since the
# concrete binding is resolved later; we only need to know it's concatenate or
# stack being called on a list literal.
_CONCAT_NAMES = ('concatenate',)
_STACK_NAMES = ('stack',)

# Module aliases that indicate JAX (as opposed to NumPy). We only rewrite JAX
# calls; NumPy's concatenate/stack are handled by numpy_extended's adjoints.
# We cannot know the true binding at desugar time (it is resolved later), so
# we match common JAX import spellings: `import jax.numpy as jnp` and
# `jax.numpy.<fn>`. An unusual alias (e.g. `import numpy as jnp`) would be
# misclassified, so we err on the side of not rewriting anything ambiguous.
_JAX_MODULE_ALIASES = ('jnp',)
_JAX_MODULE_CHAIN = ('jax', 'numpy')


def _is_jax_func(func):
  """Return True if the call's func looks like jax.numpy.<fn>."""
  if not isinstance(func, gast.Attribute):
    return False
  # jnp.<fn>
  if (isinstance(func.value, gast.Name) and
      func.value.id in _JAX_MODULE_ALIASES):
    return True
  # jax.numpy.<fn>
  if (isinstance(func.value, gast.Attribute) and
      isinstance(func.value.value, gast.Name) and
      func.value.value.id == _JAX_MODULE_CHAIN[0] and
      func.value.attr == _JAX_MODULE_CHAIN[1]):
    return True
  return False


def _is_attr_call(node, names):
  return (isinstance(node, gast.Call) and
          isinstance(node.func, gast.Attribute) and
          node.func.attr in names)


def _collect_literal_lists(func_ast):
  """Map ``name -> List/Tuple literal`` for safely inlinable list variables.

  A variable is inlinable when it is assigned exactly once in the function,
  that single assignment is a list/tuple literal, and the variable is never
  mutated afterwards (no subscript assignment ``v[i] = ...`` and no augmented
  assignment ``v += ...``). Under those conditions the elements the variable
  holds at any use site are statically known, so a concatenate/stack call that
  receives the variable can be rewritten to receive the literal elements.
  """
  assign_count = {}
  literal_value = {}
  mutated = set()

  for node in gast.walk(func_ast):
    if isinstance(node, gast.Assign):
      for target in node.targets:
        if isinstance(target, gast.Name):
          name = target.id
          assign_count[name] = assign_count.get(name, 0) + 1
          if isinstance(node.value, (gast.List, gast.Tuple)):
            literal_value[name] = node.value
          else:
            literal_value.pop(name, None)
        elif isinstance(target, gast.Subscript):
          base = target.value
          if isinstance(base, gast.Name):
            mutated.add(base.id)
    elif isinstance(node, gast.AugAssign):
      if isinstance(node.target, gast.Name):
        mutated.add(node.target.id)

  safe = {}
  for name, value in literal_value.items():
    if assign_count.get(name) == 1 and name not in mutated:
      safe[name] = value
  return safe


class ConcatDesugarer(gast.NodeTransformer):

  def __init__(self):
    # Safe list variables for the function scope currently being visited.
    self.safe_lists = {}

  def visit_FunctionDef(self, node):
    # Recompute the inlinable-list map for each function scope so that two
    # functions using the same variable name do not interfere.
    saved = self.safe_lists
    self.safe_lists = _collect_literal_lists(node)
    self.generic_visit(node)
    self.safe_lists = saved
    return node

  def visit_Call(self, node):
    self.generic_visit(node)

    if not isinstance(node.func, gast.Attribute) or not _is_jax_func(node.func):
      return node

    if node.func.attr in _CONCAT_NAMES:
      helper = 'concat_seq'
    elif node.func.attr in _STACK_NAMES:
      helper = 'stack_seq'
    else:
      return node

    # concatenate(arrays, axis=...) / stack(arrays, axis=...): first
    # positional arg is the list of arrays; axis is a keyword or 2nd arg.
    if not node.args:
      return node
    arrays_arg = node.args[0]
    if isinstance(arrays_arg, gast.Name):
      # Variable-bound list: inline it only if it is a single, unmutated
      # list/tuple literal assignment. Otherwise leave it untouched so
      # differentiation raises a clear not-implemented error.
      literal = self.safe_lists.get(arrays_arg.id)
      if literal is None:
        return node
      elts = literal.elts
    elif isinstance(arrays_arg, (gast.List, gast.Tuple)):
      elts = arrays_arg.elts
    else:
      return node

    # Determine the axis argument (keyword 'axis' or second positional).
    axis = None
    remaining_keywords = []
    for kw in node.keywords:
      if kw.arg == 'axis':
        axis = kw.value
      else:
        remaining_keywords.append(kw)
    extra_positional = node.args[1:]
    if axis is None and extra_positional:
      axis = extra_positional[0]
      extra_positional = extra_positional[1:]
    if axis is None:
      axis = gast.Constant(value=0, kind=None)

    # Build tangent.<helper>(axis, a, b, c). The axis comes first so it is
    # bound to the helper's leading positional parameter and NOT packed into
    # the varargs (which hold only the arrays).
    new_call = gast.Call(
        func=gast.Attribute(
            value=gast.Name(id='tangent', ctx=gast.Load(), annotation=None),
            attr=helper,
            ctx=gast.Load()),
        args=[copy.deepcopy(axis)] +
             [copy.deepcopy(a) for a in elts] +
             [copy.deepcopy(a) for a in extra_positional],
        keywords=remaining_keywords)
    return new_call


def desugar_concat(node):
  """Rewrite list-literal concatenate/stack calls into varargs helpers."""
  node = ConcatDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
