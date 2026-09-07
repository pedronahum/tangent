# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Ordered registry of the frontend desugaring passes.

Before differentiation, Tangent lowers the input function through a fixed
sequence of source-to-source passes (desugarings, call resolution, validation
fences, and ANF conversion). This module is the single place that sequence is
defined: each step is a `Pass` descriptor whose docstring records why it sits
where it does, and `run_passes` executes the sequence for a given mode.

Backends and users can extend the pipeline without editing this file, e.g.::

    from tangent import passes

    def my_pass(node, ctx):
      ...  # transform and return the gast Module
      return node

    passes.register_pass(
        passes.Pass('my_pass', my_pass), after='concat_desugar')

Each pass callable takes `(node, ctx)` where `node` is the gast Module being
lowered and `ctx` is a `PassContext` with the original function (`ctx.func`),
its source text (`ctx.source`, possibly '' if unavailable), and the AD mode
(`ctx.mode`). It must return the (possibly new) Module.
"""
from __future__ import absolute_import

import collections

from tangent import anf as anf_
from tangent import annotate
from tangent import chained_assign_desugar
from tangent import class_desugar
from tangent import comprehension_desugar
from tangent import concat_desugar
from tangent import desugar
from tangent import dict_method_desugar
from tangent import enumerate_desugar
from tangent import fence
from tangent import ifexp_desugar
from tangent import lambda_desugar
from tangent import listcomp_desugar
from tangent import return_desugar
from tangent import sentinel_rename
from tangent import zip_desugar


ALL_MODES = frozenset(('forward', 'reverse'))

PassContext = collections.namedtuple('PassContext', ('func', 'source', 'mode'))


class Pass(object):
  """Descriptor for one frontend pass.

  Attributes:
    name: Unique string naming the pass; anchors for `register_pass`.
    fn: Callable `(node, ctx) -> node`. See the module docstring.
    modes: Frozenset of AD modes ('forward', 'reverse') the pass runs in.
    doc: Human-readable description, including the ordering rationale. Falls
        back to `fn.__doc__`.
  """

  def __init__(self, name, fn, modes=ALL_MODES, doc=None):
    self.name = name
    self.fn = fn
    self.modes = frozenset(modes)
    self.doc = doc if doc is not None else fn.__doc__

  def __repr__(self):
    return 'Pass(%r, modes=%s)' % (self.name, sorted(self.modes))


def _sentinel_rename(node, ctx):
  """Rename reserved sentinel variable names before any code is generated."""
  return sentinel_rename.rename_sentinel_vars(node)


def _class_desugar(node, ctx):
  """Inline class method calls; needs `ctx.func` for its `__globals__`."""
  return class_desugar.inline_class_methods(node, ctx.func)


def _lambda_desugar(node, ctx):
  """Lower lambdas to named functions."""
  return lambda_desugar.desugar_lambdas(node)


def _return_desugar(node, ctx):
  """Normalize returns; runs early so later passes see a single return form."""
  return return_desugar.desugar_returns(node)


def _chained_assign_desugar(node, ctx):
  """Split chained assignments (a = b = c) into simple assignments."""
  return chained_assign_desugar.desugar_chained_assignments(node)


def _enumerate_desugar(node, ctx):
  """Lower enumerate() loops; runs before the generic comprehension pass."""
  return enumerate_desugar.desugar_enumerate(node)


def _zip_desugar(node, ctx):
  """Lower zip() loops; runs before the generic comprehension pass."""
  return zip_desugar.desugar_zip(node)


def _comprehension_desugar(node, ctx):
  """Lower generator/set/dict comprehensions into explicit loops."""
  return comprehension_desugar.desugar_comprehensions(node)


def _listcomp_desugar(node, ctx):
  """Lower list comprehensions into explicit loops."""
  return listcomp_desugar.desugar_listcomps(node)


def _dict_method_desugar(node, ctx):
  """Lower dict method calls (e.g. d.get(k, default)) into supported forms."""
  return dict_method_desugar.desugar_dict_methods(node)


def _concat_desugar(node, ctx):
  """Lower list/tuple concatenation into differentiable primitives."""
  return concat_desugar.desugar_concat(node)


def _ifexp_desugar(node, ctx):
  """Lower conditional expressions (ternaries) to if-statements.

  Forward mode supports if-statements but not conditional expressions.
  Runs last among the desugarings so it also catches ternaries introduced by
  other passes (e.g. d.get(k, default)). Reverse mode differentiates
  conditional expressions directly, so this pass is forward-only.
  """
  return ifexp_desugar.desugar_ifexps(node)


def _resolve_calls(node, ctx):
  """Annotate calls with resolved function handles.

  Runs after all desugaring so that resolution sees the transformed AST
  (desugarings may introduce or rewrite calls), and before the passes that
  rely on call annotations.
  """
  annotate.ResolveCalls(ctx.func).visit(node)
  return node


def _explicit_loop_indexes(node, ctx):
  """Rewrite for-loops to use explicit loop indexes."""
  return desugar.explicit_loop_indexes(node)


def _fence(node, ctx):
  """Reject unsupported language constructs with a clear error.

  Runs after desugaring (constructs the desugarings remove need no fence)
  and before ANF, so errors point at recognizable user code.
  """
  fence.validate(node, ctx.source)
  return node


def _anf(node, ctx):
  """Convert to A-normal form. Runs last: AD transforms consume ANF."""
  return anf_.anf(node)


# The pipeline, in execution order. `run_passes` walks this list, skipping
# passes whose `modes` do not include the current mode.
_REGISTRY = [
    Pass('sentinel_rename', _sentinel_rename),
    Pass('class_desugar', _class_desugar),
    Pass('lambda_desugar', _lambda_desugar),
    Pass('return_desugar', _return_desugar),
    Pass('chained_assign_desugar', _chained_assign_desugar),
    Pass('enumerate_desugar', _enumerate_desugar),
    Pass('zip_desugar', _zip_desugar),
    Pass('comprehension_desugar', _comprehension_desugar),
    Pass('listcomp_desugar', _listcomp_desugar),
    Pass('dict_method_desugar', _dict_method_desugar),
    Pass('concat_desugar', _concat_desugar),
    Pass('ifexp_desugar', _ifexp_desugar, modes=frozenset(('forward',))),
    Pass('resolve_calls', _resolve_calls),
    Pass('explicit_loop_indexes', _explicit_loop_indexes),
    Pass('fence', _fence),
    Pass('anf', _anf),
]


def _index_of(name):
  for i, p in enumerate(_REGISTRY):
    if p.name == name:
      return i
  raise ValueError('No pass named %r in the registry (have: %s)' %
                   (name, ', '.join(p.name for p in _REGISTRY)))


def register_pass(pass_, before=None, after=None):
  """Insert an additional pass relative to a named existing pass.

  This is the extension hook for backends and users: it splices a new `Pass`
  into the pipeline without editing the built-in sequence.

  Args:
    pass_: A `Pass` descriptor. Its name must not collide with a registered
        pass.
    before: Name of the pass to insert immediately before.
    after: Name of the pass to insert immediately after.
        Exactly one of `before`/`after` must be given.

  Raises:
    ValueError: On a name collision, a missing anchor, or if not exactly one
        of `before`/`after` is given.
  """
  if (before is None) == (after is None):
    raise ValueError('Pass exactly one of before= or after=')
  if any(p.name == pass_.name for p in _REGISTRY):
    raise ValueError('A pass named %r is already registered' % pass_.name)
  if before is not None:
    _REGISTRY.insert(_index_of(before), pass_)
  else:
    _REGISTRY.insert(_index_of(after) + 1, pass_)


def unregister_pass(name):
  """Remove a previously registered pass by name."""
  del _REGISTRY[_index_of(name)]


def get_passes(mode):
  """Return the ordered list of `Pass` descriptors that run for `mode`."""
  return [p for p in _REGISTRY if mode in p.modes]


def run_passes(node, func, source, mode):
  """Run the registered frontend passes for `mode` over a gast Module.

  Args:
    node: The gast Module to lower.
    func: The original Python function being differentiated.
    source: Its source text ('' if unavailable).
    mode: 'forward' or 'reverse'.

  Returns:
    The lowered Module.
  """
  ctx = PassContext(func=func, source=source, mode=mode)
  for p in get_passes(mode):
    node = p.fn(node, ctx)
  return node
