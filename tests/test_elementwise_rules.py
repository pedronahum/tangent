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
"""Tests for the backend-neutral elementwise rule table.

The numerical behaviour of the generated rules is validated per backend by
tests/test_backend_coverage.py (finite-difference oracles); this module
checks the generation machinery itself: every generated template has
retrievable source (Tangent splices template source into derivative code)
and every table-registered op carries BOTH a reverse-mode and a
forward-mode rule - the parity property the table exists to guarantee.
"""
import inspect

import pytest

import tangent  # noqa: F401  (imports register the backend extensions)
from tangent import grads
from tangent import tangents


def _installed(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _sample_ops():
    """A sample of (backend, op object) pairs for installed backends."""
    ops = []
    if _installed('torch'):
        import torch
        ops += [('torch', f) for f in
                (torch.exp, torch.tanh, torch.arcsin, torch.sign,
                 torch.rsqrt, torch.expm1)]
    if _installed('jax'):
        import jax.numpy as jnp
        ops += [('jax', f) for f in
                (jnp.exp, jnp.tanh, jnp.arcsin, jnp.sign, jnp.reciprocal,
                 jnp.log1p)]
    if _installed('tinygrad'):
        from tinygrad import Tensor
        ops += [('tinygrad', f) for f in
                (Tensor.exp, Tensor.tanh, Tensor.asin, Tensor.sign,
                 Tensor.rsqrt)]
    if _installed('keras'):
        import keras.ops as kops
        ops += [('keras', f) for f in
                (kops.exp, kops.tanh, kops.arcsin, kops.sign)]
    return ops


@pytest.mark.parametrize('backend,op', _sample_ops(),
                         ids=lambda v: getattr(v, '__name__', v))
def test_both_directions_registered(backend, op):
    assert op in grads.adjoints, '%s: missing adjoint' % backend
    assert op in tangents.tangents, '%s: missing tangent' % backend
    assert op not in grads.UNIMPLEMENTED_ADJOINTS
    assert op not in tangents.UNIMPLEMENTED_TANGENTS


@pytest.mark.parametrize('backend,op', _sample_ops(),
                         ids=lambda v: getattr(v, '__name__', v))
def test_generated_templates_have_source(backend, op):
    # Tangent parses template source with inspect.getsource; the generated
    # templates register their synthetic files in linecache.
    adj_src = inspect.getsource(grads.adjoints[op])
    tan_src = inspect.getsource(tangents.tangents[op])
    assert adj_src.startswith('def adjoint_')
    assert 'd[x] = ' in adj_src
    assert tan_src.startswith('def tangent_')
    assert 'd[y] = ' in tan_src


def test_formulas_use_g_placeholder_once():
    """Both modes are linear in the incoming derivative: {g} appears once."""
    from tangent import elementwise_rules

    def vocab(fn, arg):
        return 'M.%s(%s)' % (fn, arg)

    for name, formula in elementwise_rules.FORMULAS.items():
        expr = formula(vocab)
        assert expr.count('{g}') == 1, name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
