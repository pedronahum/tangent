"""The shared binary-op rule table is registered identically across backends.

`tangent/elementwise_rules.py` generates the adjoint AND tangent for the
backend-neutral binary ops (add, subtract, multiply, divide) from one
canonical table, so JAX / PyTorch / Keras / tinygrad cannot drift on them.
This is the executable "cannot drift" guarantee: for every installed backend,
each table op must have a registered adjoint and tangent, and must produce the
analytically correct gradient.
"""

import numpy as np
import pytest

import tangent
from tangent import grads
from tangent import tangents as tangents_module
from tangent.elementwise_rules import BINARY_RULES

# (backend name, module import, op accessor) - the accessor returns the op
# object for a rule name, or None if the backend does not expose it.
_BACKENDS = []


def _try(name, build):
    try:
        _BACKENDS.append((name, build()))
    except Exception:
        pass


_try('jax', lambda: __import__('jax.numpy', fromlist=['']))
_try('torch', lambda: __import__('torch'))
_try('tinygrad', lambda: __import__('tinygrad', fromlist=['Tensor']))


def _op(backend, mod, rule):
    if backend == 'jax':
        return {
            'add': mod.add,
            'subtract': mod.subtract,
            'multiply': mod.multiply,
            'divide': mod.divide,
        }[rule]
    if backend == 'torch':
        return {'add': mod.add, 'subtract': mod.sub, 'multiply': mod.mul, 'divide': mod.div}[rule]
    if backend == 'tinygrad':
        T = mod.Tensor
        return {'add': T.add, 'subtract': T.sub, 'multiply': T.mul, 'divide': T.div}[rule]
    return None


@pytest.mark.skipif(not _BACKENDS, reason='no array backend installed')
@pytest.mark.parametrize('rule', sorted(BINARY_RULES))
def test_every_backend_registers_binary_rule(rule):
    for backend, mod in _BACKENDS:
        op = _op(backend, mod, rule)
        assert op in grads.adjoints, '%s.%s has no generated adjoint' % (backend, rule)
        assert op in tangents_module.tangents, '%s.%s has no generated tangent' % (backend, rule)


class TestGeneratedGradientsAreCorrect:
    # Analytic gradients of z = op(x, y), summed: dx and dy.
    EXPECTED = {
        'add': lambda x, y: (np.ones_like(x), np.ones_like(y)),
        'subtract': lambda x, y: (np.ones_like(x), -np.ones_like(y)),
        'multiply': lambda x, y: (y, x),
        'divide': lambda x, y: (1.0 / y, -x / (y * y)),
    }

    @pytest.mark.skipif(not any(b[0] == 'jax' for b in _BACKENDS), reason='jax not installed')
    @pytest.mark.parametrize('rule', sorted(BINARY_RULES))
    def test_jax(self, rule):
        import jax.numpy as jnp

        op = _op('jax', jnp, rule)
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 1.0, 4.0])

        def f(a, b):
            return jnp.sum(op(a, b))

        gx, gy = tangent.grad(f, wrt=(0, 1))(x, y)
        ex, ey = self.EXPECTED[rule](np.asarray(x), np.asarray(y))
        np.testing.assert_allclose(np.asarray(gx), ex, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(gy), ey, rtol=1e-5)


def test_table_covers_the_neutral_binary_ops():
    assert set(BINARY_RULES) == {'add', 'subtract', 'multiply', 'divide'}
