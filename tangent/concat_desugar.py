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

Only calls whose first argument is a list *literal* are rewritten. If the list
is a variable (built elsewhere), its elements cannot be statically identified,
so the call is left untouched and will raise a clear not-implemented error.
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


class ConcatDesugarer(gast.NodeTransformer):

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
    if not isinstance(arrays_arg, (gast.List, gast.Tuple)):
      # List is a variable; elements are not statically known. Leave as-is so
      # differentiation raises a clear not-implemented error.
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
             [copy.deepcopy(a) for a in arrays_arg.elts] +
             [copy.deepcopy(a) for a in extra_positional],
        keywords=remaining_keywords)
    return new_call


def desugar_concat(node):
  """Rewrite list-literal concatenate/stack calls into varargs helpers."""
  node = ConcatDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
