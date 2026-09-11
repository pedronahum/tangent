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
"""Backend-neutral elementwise gradient rules.

One canonical mathematical definition per unary elementwise op, from which
the per-backend adjoint *and* tangent templates are generated. This replaces
the rule bodies that were previously copy-pasted (with only the module
prefix and seed helper changed) across the JAX, TensorFlow, PyTorch, Keras
and tinygrad extension modules, and guarantees every generated op has both a
reverse- and a forward-mode rule.

Mechanics
---------
Tangent's adjoints/tangents are *source templates*: their source is parsed
with inspect.getsource and spliced into the generated derivative code. The
generator therefore renders per-backend template source text, compiles it
under a synthetic filename registered in ``linecache`` (which is where
``inspect.getsource`` finds it), and registers the resulting function object
with the ordinary ``@adjoint`` / ``@tangent_`` registries.

For a unary elementwise op ``y = f(x)`` both derivative directions multiply
an incoming derivative ``g`` by the same factor ``f'(x)``::

    reverse (adjoint):  d[x] = seed(d[y]) * f'(x)
    forward (tangent):  d[y] = d[x] * f'(x)

so a single ``FORMULAS`` entry serves both modes. A formula is a function of
the backend *vocabulary* returning an expression string over:

    ``{g}`` - the incoming derivative (wrapped in the backend's seed helper
              in reverse mode, for backends whose tensors do not mix with
              plain Python float seeds);
    ``x``   - the primal input;
    ``y``   - the primal output.

Backend-specific calls (``cos(x)``, ``sqrt(1 - x*x)``, the relu mask) are
spelled through the vocabulary, so one formula renders as ``jnp.cos(x)``
for JAX, ``tf.cos(x)`` for TensorFlow and ``(x).cos()`` for tinygrad's
method-style API.

Ops whose rules genuinely differ per backend (convolutions, pooling, matmul
variants, reductions, wrapped activation stacks like ``jax.nn.relu``) stay
hand-written in the extension modules.
"""

from __future__ import absolute_import

import linecache

from tangent import grads
from tangent import tangents as tangents_module


# ---------------------------------------------------------------------------
# Canonical formulas: rule name -> fn(vocab) -> expression in {g}, x, y.
#
# Every formula must use {g} exactly once (both modes are linear in the
# incoming derivative). Constants are spelled as float literals so the
# expression stays backend-neutral (ln 2 = 0.6931..., ln 10 = 2.3025...).
# ---------------------------------------------------------------------------

FORMULAS = {
    'exp': lambda v: '{g} * y',
    'expm1': lambda v: '{g} * ' + v('exp', 'x'),
    'exp2': lambda v: '{g} * y * 0.6931471805599453',
    'log': lambda v: '{g} / x',
    'log2': lambda v: '{g} / (x * 0.6931471805599453)',
    'log10': lambda v: '{g} / (x * 2.302585092994046)',
    'log1p': lambda v: '{g} / (1.0 + x)',
    'sqrt': lambda v: '{g} / (2.0 * y)',
    'rsqrt': lambda v: '-{g} * y / (2.0 * x)',
    'square': lambda v: '{g} * 2.0 * x',
    'reciprocal': lambda v: '-{g} / (x * x)',
    'negative': lambda v: '-{g}',
    'abs': lambda v: '{g} * ' + v('sign', 'x'),
    'sin': lambda v: '{g} * ' + v('cos', 'x'),
    'cos': lambda v: '-{g} * ' + v('sin', 'x'),
    'tan': lambda v: '{g} * (1.0 + y * y)',
    'arcsin': lambda v: '{g} / ' + v('sqrt', '1.0 - x * x'),
    'arccos': lambda v: '-{g} / ' + v('sqrt', '1.0 - x * x'),
    'arctan': lambda v: '{g} / (1.0 + x * x)',
    'sinh': lambda v: '{g} * ' + v('cosh', 'x'),
    'cosh': lambda v: '{g} * ' + v('sinh', 'x'),
    'tanh': lambda v: '{g} * (1.0 - y * y)',
    'sigmoid': lambda v: '{g} * y * (1.0 - y)',
    'relu': lambda v: '{g} * ' + v('mask_pos', 'x'),
}

# Piecewise-constant ops: the derivative is zero everywhere it exists.
# tangent.init_grad dispatches on the runtime tensor type, so the same body
# serves every backend.
ZERO_FORMULA_OPS = ('floor', 'ceil', 'round', 'sign')


# ---------------------------------------------------------------------------
# Vocabularies: how a backend spells calls used inside formulas.
# ---------------------------------------------------------------------------


def prefix_vocab(prefix, **overrides):
    """Vocabulary for function-style backends (``<prefix>.<fn>(arg)``).

    Overrides map a vocabulary name to a format string over ``{arg}``
    (e.g. ``mask_pos='tf.cast(({arg}) > 0, x.dtype)'``).
    """

    def vocab(fn, arg):
        spelled = overrides.get(fn)
        if spelled is not None:
            return spelled.format(arg=arg)
        return '%s.%s(%s)' % (prefix, fn, arg)

    return vocab


def method_vocab(**overrides):
    """Vocabulary for method-style backends (``(arg).<fn>()``)."""

    def vocab(fn, arg):
        spelled = overrides.get(fn)
        if spelled is not None:
            return spelled.format(arg=arg)
        return '(%s).%s()' % (arg, fn)

    return vocab


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def _compile_template(name, body, filename):
    """Compile one template function from source, retrievably for inspect."""
    src = 'def %s(y, x):\n    %s\n' % (name, body)
    code = compile(src, filename, 'exec')
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    fn = namespace[name]
    # inspect.getsource consults linecache; register the synthetic file so
    # Tangent's quoting.parse_function can recover the template source.
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    return fn


def register_elementwise(backend, ops, vocab, seed='{g}'):
    """Generate and register adjoint + tangent templates for elementwise ops.

    Args:
      backend: Short backend name, used in generated function and file names
        (e.g. 'torch').
      ops: Dict mapping a rule name (a key of FORMULAS, or one of
        ZERO_FORMULA_OPS) to the backend function object to register the
        rules against - or to a tuple of alias objects (e.g. ``torch.neg``
        and ``torch.negative``). ``None`` values are skipped, so callers can
        build the dict with ``getattr(mod, 'name', None)`` guards.
      vocab: A vocabulary callable ``(fn, arg_expr) -> source string``; see
        prefix_vocab / method_vocab.
      seed: Format string over ``{g}`` wrapping the incoming derivative in
        reverse mode (e.g. ``'tangent.torch_seed({g}, x)'``). Backends whose
        tensors mix with plain float seeds keep the default passthrough.
    """
    for rule_name, funcs in ops.items():
        if funcs is None:
            continue
        if not isinstance(funcs, tuple):
            funcs = (funcs,)

        if rule_name in ZERO_FORMULA_OPS:
            adjoint_expr = 'tangent.init_grad(x)'
            tangent_expr = 'tangent.init_grad(x)'
        else:
            expr = FORMULAS[rule_name](vocab)
            adjoint_expr = expr.format(g=seed.format(g='d[y]'))
            tangent_expr = expr.format(g='d[x]')

        adjoint_fn = _compile_template(
            'adjoint_%s_%s' % (backend, rule_name),
            'd[x] = %s' % adjoint_expr,
            '<tangent-elementwise>/%s_%s_adjoint.py' % (backend, rule_name),
        )
        tangent_fn = _compile_template(
            'tangent_%s_%s' % (backend, rule_name),
            'd[y] = %s' % tangent_expr,
            '<tangent-elementwise>/%s_%s_tangent.py' % (backend, rule_name),
        )

        for func in funcs:
            if func is None:
                continue
            grads.adjoint(func)(adjoint_fn)
            tangents_module.tangent_(func)(tangent_fn)
            grads.UNIMPLEMENTED_ADJOINTS.discard(func)
            tangents_module.UNIMPLEMENTED_TANGENTS.discard(func)


# ---------------------------------------------------------------------------
# Binary elementwise ops: z = op(x, y)
#
# These are fully backend-neutral - their bodies use only tangent.unbroadcast
# (backend-dispatched on the primal's type) and arithmetic operators, so ONE
# canonical template per op registers against every backend's spelling of the
# op (jnp.add, torch.add, tf.add, kops.add, Tensor.add, ...). They were
# previously copy-pasted across all five extension modules; generating them
# here makes drift impossible.
# ---------------------------------------------------------------------------

# rule name -> (adjoint body over d[z], x, y ; tangent body over d[x], d[y]).
BINARY_RULES = {
    # Adjoint bodies use `dz` - the (optionally seed-wrapped) incoming
    # derivative d[z], bound once by the generator.
    'add': (
        'd[x] = tangent.unbroadcast(dz, x); d[y] = tangent.unbroadcast(dz, y)',
        'd[z] = d[x] + d[y]',
    ),
    'subtract': (
        'd[x] = tangent.unbroadcast(dz, x); d[y] = tangent.unbroadcast(-dz, y)',
        'd[z] = d[x] - d[y]',
    ),
    'multiply': (
        'd[x] = tangent.unbroadcast(dz * y, x); d[y] = tangent.unbroadcast(dz * x, y)',
        'd[z] = d[x] * y + x * d[y]',
    ),
    'divide': (
        'd[x] = tangent.unbroadcast(dz / y, x); d[y] = tangent.unbroadcast(-dz * x / (y * y), y)',
        'd[z] = (d[x] * y - x * d[y]) / (y * y)',
    ),
}


def _compile_binary_template(name, targets_body, params, filename):
    src = 'def %s(%s):\n    %s\n' % (name, params, targets_body)
    code = compile(src, filename, 'exec')
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    fn = namespace[name]
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    return fn


def register_binary(backend, ops, seed='{g}'):
    """Generate and register adjoint + tangent templates for binary ops.

    Args:
      backend: Short backend name, for generated function/file names.
      ops: Dict mapping a BINARY_RULES key to the backend's function object
        (or a tuple of aliases, e.g. torch.divide and torch.true_divide).
        None values are skipped.
      seed: Format string over ``{g}`` wrapping the incoming derivative d[z]
        in reverse mode (e.g. ``'tangent.torch_seed({g}, x)'``); the default
        passes it through for backends whose tensors mix with float seeds.
    """
    for rule_name, funcs in ops.items():
        if funcs is None:
            continue
        if not isinstance(funcs, tuple):
            funcs = (funcs,)
        adjoint_body, tangent_body = BINARY_RULES[rule_name]
        adjoint_fn = _compile_binary_template(
            'adjoint_%s_%s' % (backend, rule_name),
            'dz = %s; %s' % (seed.format(g='d[z]'), adjoint_body),
            'z, x, y',
            '<tangent-binary>/%s_%s_adjoint.py' % (backend, rule_name),
        )
        tangent_fn = _compile_binary_template(
            'tangent_%s_%s' % (backend, rule_name),
            tangent_body,
            'z, x, y',
            '<tangent-binary>/%s_%s_tangent.py' % (backend, rule_name),
        )
        for func in funcs:
            if func is None:
                continue
            grads.adjoint(func)(adjoint_fn)
            tangents_module.tangent_(func)(tangent_fn)
            grads.UNIMPLEMENTED_ADJOINTS.discard(func)
            tangents_module.UNIMPLEMENTED_TANGENTS.discard(func)
