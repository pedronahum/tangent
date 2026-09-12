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
"""The unified backend op catalog.

One place that knows every op whose gradient rule is shared across array
backends, which *family* it belongs to (and therefore how its adjoint/tangent
are generated), and - through a single `register()` entry - how each backend
spells it. Backends call `register(name, ops, ...)` once with a flat dict of
`rule_name -> backend_function`; the catalog routes each rule to the right
template generator so JAX / PyTorch / Keras / tinygrad (and, for the unary
set, TensorFlow) cannot drift on any cataloged op.

This unifies what used to be two separate `register_elementwise` /
`register_binary` calls per backend, and gives the coverage tests a single
list of shared ops to check every backend against (`tests/`).

The rule *bodies* (the math) live in `tangent/elementwise_rules.py`; this
module owns the catalog structure, the family classification, and the unified
registration entry. Op families whose rules carry genuine per-backend
divergence (reductions with axis/keepdims, matmul rank promotion, conv/pool)
stay hand-written in the extension modules until they are migrated here.
"""

from __future__ import absolute_import

from tangent import elementwise_rules as _er

# Rule name -> family. The family selects the generator and documents what
# kind of op it is. Kept in sync with the tables in elementwise_rules.
UNARY_RULES = frozenset(_er.FORMULAS) | frozenset(_er.ZERO_FORMULA_OPS)
BINARY_RULES = frozenset(_er.BINARY_RULES)
REDUCTION_RULES = frozenset(_er.REDUCTION_RULES)

RULE_FAMILY = {}
for _name in UNARY_RULES:
    RULE_FAMILY[_name] = 'unary'
for _name in BINARY_RULES:
    RULE_FAMILY[_name] = 'binary'
for _name in REDUCTION_RULES:
    RULE_FAMILY[_name] = 'reduction'


def rule_names():
    """All rule names the catalog can generate, sorted."""
    return sorted(RULE_FAMILY)


def family_of(rule_name):
    """The family ('unary' or 'binary') a rule belongs to."""
    return RULE_FAMILY[rule_name]


def register(backend, ops, vocab=None, seed='{g}'):
    """Register every cataloged op a backend provides, from one flat dict.

    Args:
      backend: Short backend name (e.g. 'torch'), for generated names.
      ops: Dict mapping a catalog rule name to the backend's function object
        (or a tuple of aliases). Unknown rule names raise, so a typo can't
        silently skip an op. `None` values are skipped (a backend may lack an
        op). Unary and binary rules may be mixed freely.
      vocab: Vocabulary callable for unary elementwise rules (see
        `elementwise_rules.prefix_vocab` / `method_vocab`). Required if `ops`
        contains any unary rule; ignored otherwise.
      seed: Reverse-mode seed wrapper over `{g}` (e.g.
        `'tangent.torch_seed({g}, x)'`), applied to both families.

    Raises:
      KeyError: if `ops` names a rule the catalog does not define.
      ValueError: if a unary rule is given without a `vocab`.
    """
    unary_ops = {}
    binary_ops = {}
    for rule_name, funcs in ops.items():
        family = RULE_FAMILY.get(rule_name)
        if family is None:
            raise KeyError(
                'Unknown catalog op %r for backend %r; known ops: %s'
                % (rule_name, backend, ', '.join(rule_names()))
            )
        if family == 'unary':
            unary_ops[rule_name] = funcs
        elif family == 'binary':
            binary_ops[rule_name] = funcs
        else:
            raise ValueError(
                'Reduction rule %r must be registered with register_reductions(), '
                'not register() - it needs an explicit forward-mode body.' % rule_name
            )

    if unary_ops:
        if vocab is None:
            raise ValueError('vocab is required to register unary rules: %s' % sorted(unary_ops))
        _er.register_elementwise(backend, unary_ops, vocab, seed=seed)
    if binary_ops:
        _er.register_binary(backend, binary_ops, seed=seed)


def register_reductions(backend, ops, forward, **kwargs):
    """Register a backend's sum / mean reductions through the catalog.

    Thin passthrough to ``elementwise_rules.register_reductions`` that keeps
    reductions inside the one catalog surface. The reduction adjoint is
    generated from a shared template; ``forward`` carries the backend-specific
    forward-mode bodies (see that function for the full argument list).
    """
    _er.register_reductions(backend, ops, forward, **kwargs)
