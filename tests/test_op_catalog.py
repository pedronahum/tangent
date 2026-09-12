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
"""The unified op catalog: classification, routing, and the single entry point.

Every array backend (JAX / PyTorch / Keras / tinygrad, and TF for the unary
set) registers its shared elementwise ops through the one `op_catalog.register`
call. These tests pin the catalog's own machinery: the family classification
stays in sync with the underlying rule tables, a typo'd rule name fails loudly
instead of silently skipping an op, and the unified entry point routes each
rule to the correct generator so no backend can drift on a cataloged op.
"""

import pytest

from tangent import op_catalog
from tangent import elementwise_rules as er


class TestClassification:
    def test_every_rule_has_exactly_one_family(self):
        # RULE_FAMILY partitions the union of the underlying tables; nothing is
        # missing and nothing is double-classified.
        assert set(op_catalog.RULE_FAMILY) == op_catalog.UNARY_RULES | op_catalog.BINARY_RULES
        assert op_catalog.UNARY_RULES.isdisjoint(op_catalog.BINARY_RULES)

    def test_family_tables_match_elementwise_rules(self):
        # The catalog is a view over the elementwise_rules tables; if a rule is
        # added there it must appear here in the right family.
        assert op_catalog.UNARY_RULES == frozenset(er.FORMULAS) | frozenset(er.ZERO_FORMULA_OPS)
        assert op_catalog.BINARY_RULES == frozenset(er.BINARY_RULES)

    def test_rule_names_is_sorted_and_complete(self):
        assert op_catalog.rule_names() == sorted(op_catalog.RULE_FAMILY)

    def test_family_of(self):
        assert op_catalog.family_of('exp') == 'unary'
        assert op_catalog.family_of('multiply') == 'binary'

    def test_family_of_unknown_raises(self):
        with pytest.raises(KeyError):
            op_catalog.family_of('definitely_not_a_rule')


class TestRegisterRouting:
    def test_unknown_rule_raises_with_helpful_message(self):
        # A typo can't silently skip an op: it names the offending rule and
        # lists what the catalog knows.
        with pytest.raises(KeyError) as exc:
            op_catalog.register('fakebackend', {'expp': object()}, vocab=lambda fn, a: a)
        msg = str(exc.value)
        assert 'expp' in msg
        assert 'exp' in msg  # the known-ops list

    def test_unary_rule_without_vocab_raises(self):
        with pytest.raises(ValueError, match='vocab'):
            op_catalog.register('fakebackend', {'exp': object()})

    def test_binary_only_needs_no_vocab(self):
        # A backend registering only binary ops must not be forced to pass a
        # vocab (vocab is a unary-elementwise concept).
        recorded = {}

        def fake_binary(backend, ops, seed='{g}'):
            recorded['binary'] = (backend, dict(ops), seed)

        def fake_unary(*a, **k):
            raise AssertionError('unary generator must not be called')

        orig_b, orig_u = er.register_binary, er.register_elementwise
        er.register_binary, er.register_elementwise = fake_binary, fake_unary
        try:
            sentinel = object()
            op_catalog.register('fakebackend', {'add': sentinel}, seed='wrap({g})')
        finally:
            er.register_binary, er.register_elementwise = orig_b, orig_u

        backend, ops, seed = recorded['binary']
        assert backend == 'fakebackend'
        assert ops == {'add': sentinel}
        assert seed == 'wrap({g})'

    def test_mixed_dict_is_split_by_family(self):
        # One flat dict of mixed unary/binary rules is routed to both
        # generators, partitioned by family, with the shared seed forwarded.
        seen = {}

        def fake_unary(backend, ops, vocab, seed='{g}'):
            seen['unary'] = set(ops)

        def fake_binary(backend, ops, seed='{g}'):
            seen['binary'] = set(ops)

        orig_u, orig_b = er.register_elementwise, er.register_binary
        er.register_elementwise, er.register_binary = fake_unary, fake_binary
        try:
            op_catalog.register(
                'fakebackend',
                {'exp': object(), 'log': object(), 'add': object(), 'multiply': object()},
                vocab=lambda fn, a: a,
            )
        finally:
            er.register_elementwise, er.register_binary = orig_u, orig_b

        assert seen['unary'] == {'exp', 'log'}
        assert seen['binary'] == {'add', 'multiply'}


class TestBackendsUseTheCatalog:
    """The installed backends register through the catalog, so their shared
    elementwise ops carry both a reverse- and forward-mode rule."""

    @pytest.mark.parametrize(
        'backend,build_ops',
        [
            ('torch', lambda: _torch_ops()),
            ('jax', lambda: _jax_ops()),
            ('tinygrad', lambda: _tinygrad_ops()),
        ],
    )
    def test_shared_binary_ops_are_registered(self, backend, build_ops):
        ops = build_ops()
        if ops is None:
            pytest.skip('%s not installed' % backend)
        from tangent import grads
        from tangent import tangents as tangents_module

        for rule, op in ops.items():
            assert op in grads.adjoints, '%s.%s missing adjoint' % (backend, rule)
            assert op in tangents_module.tangents, '%s.%s missing tangent' % (backend, rule)


def _torch_ops():
    try:
        import torch
    except ImportError:
        return None
    return {'add': torch.add, 'multiply': torch.mul, 'exp': torch.exp, 'tanh': torch.tanh}


def _jax_ops():
    try:
        import jax.numpy as jnp
    except ImportError:
        return None
    return {'add': jnp.add, 'multiply': jnp.multiply, 'exp': jnp.exp, 'tanh': jnp.tanh}


def _tinygrad_ops():
    try:
        from tinygrad import Tensor
    except ImportError:
        return None
    return {'add': Tensor.add, 'multiply': Tensor.mul, 'exp': Tensor.exp, 'tanh': Tensor.tanh}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
