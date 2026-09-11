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
"""Custom gradients as a public, JAX-shaped API.

``@tangent.custom_vjp`` marks a function as differentiated by a user-supplied
rule instead of by transforming its body - for numerically better gradients,
for calling out to code Tangent cannot transform, or for registering a rule
computed by Tangent itself in another backend::

    @tangent.custom_vjp
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))

    @gelu.defvjp
    def gelu_vjp(g, ans, x):
        # cotangent, primal output, then the primal arguments
        cdf = 0.5 * (1.0 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))
        pdf = np.exp(-0.5 * x * x) * 0.3989422804014327
        return g * (cdf + x * pdf)

The bwd rule receives ``(g, ans, *primal_args)`` and returns one gradient per
argument (a single value for unary functions, a tuple otherwise). Tangent
re-provides the primal inputs and output, so there is no residual plumbing.

``@gelu.defjvp`` optionally registers the forward-mode rule, receiving
``(ans, *primal_args, *arg_tangents)`` and returning the output tangent.

``tangent.stop_gradient(x)`` is the identity with a zero derivative in both
modes - the standard way to freeze part of a computation.
"""

from __future__ import absolute_import

# Registry: stable key -> user bwd/jvp callables, consulted by the generated
# code at run time via tangent.custom_bwd_call / custom_jvp_call.
_custom_bwds = {}
_custom_jvps = {}


def _key(func):
    return '%s.%s' % (getattr(func, '__module__', '?'), getattr(func, '__qualname__', func))


def custom_bwd_call(key, g, ans, *args):
    """Runtime hook the generated adjoint calls for a custom_vjp function."""
    return _custom_bwds[key](g, ans, *args)


def custom_jvp_call(key, ans, *args_and_tangents):
    """Runtime hook the generated tangent code calls for a custom_jvp rule."""
    return _custom_jvps[key](ans, *args_and_tangents)


def _make_template(source, name):
    """Compile template source into a function whose source is retrievable.

    Gradient templates are consumed by `template.replace`, which reads the
    function's source - so a plain exec() won't do; `compile_file` writes a
    real file first.
    """
    from tangent import compile as compile_

    module = compile_.compile_file(source, {})
    return getattr(module, name)


def _param_names(func):
    import inspect

    params = []
    for p in inspect.signature(func).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ValueError(
                'custom_vjp functions must have a fixed positional signature '
                '(no *args/**kwargs): %s' % func.__qualname__
            )
        params.append(p.name)
    return params


def custom_vjp(func):
    """Mark `func` as differentiated by a user-registered VJP rule.

    Returns `func` with two decorator attributes: `func.defvjp` registers the
    reverse-mode rule and `func.defjvp` the (optional) forward-mode rule.
    Until `defvjp` is called, differentiating a caller of `func` transforms
    its body as usual.
    """
    params = _param_names(func)
    key = _key(func)

    def defvjp(bwd):
        from tangent import grads

        _custom_bwds[key] = bwd
        args = ', '.join(params)
        lines = ['def _adjoint_template(ans, %s):' % args]
        lines.append(
            "    _tng_customgrads = tangent.custom_bwd_call('%s', d[ans], ans, %s)" % (key, args)
        )
        if len(params) == 1:
            lines.append('    d[%s] = _tng_customgrads' % params[0])
        else:
            for i, p in enumerate(params):
                lines.append('    d[%s] = _tng_customgrads[%d]' % (p, i))
        template_fn = _make_template('\n'.join(lines) + '\n', '_adjoint_template')
        grads.adjoints[func] = template_fn
        grads.UNIMPLEMENTED_ADJOINTS.discard(func)
        return bwd

    def defjvp(jvp):
        from tangent import tangents

        _custom_jvps[key] = jvp
        args = ', '.join(params)
        dargs = ', '.join('d[%s]' % p for p in params)
        source = (
            'def _tangent_template(ans, %s):\n'
            "    d[ans] = tangent.custom_jvp_call('%s', ans, %s, %s)\n" % (args, key, args, dargs)
        )
        template_fn = _make_template(source, '_tangent_template')
        tangents.tangents[func] = template_fn
        tangents.UNIMPLEMENTED_TANGENTS.discard(func)
        return jvp

    func.defvjp = defvjp
    func.defjvp = defjvp
    return func


def stop_gradient(x):
    """Identity in the primal; zero derivative in both modes.

    Values flowing through `stop_gradient` are treated as constants by the
    differentiator - the standard way to freeze part of a computation.
    """
    return x
