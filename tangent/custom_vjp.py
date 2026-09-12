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

For a library op you do not own (rather than your own function), the low-level
``tangent.register_adjoint`` / ``tangent.register_tangent`` decorators attach a
gradient rule to an existing callable using Tangent's template DSL - see their
docstrings.
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


# ---------------------------------------------------------------------------
# Low-level rule registration.
#
# custom_vjp above is the ergonomic path for a function you own or a black box
# you call: you write a plain-Python bwd/jvp over (g, ans, *args). The two
# decorators below are the low-level path for adding a rule to a callable you
# do NOT own - a library op such as numpy.hypot, or a backend function - using
# Tangent's template DSL directly (the same mechanism the built-in backend
# rules use). Prefer custom_vjp unless you specifically need to attach a rule
# to an existing function object rather than wrap it.
# ---------------------------------------------------------------------------


def register_adjoint(func):
    """Register a reverse-mode (adjoint) template for an existing callable.

    Use this to teach Tangent the gradient of a function it cannot transform -
    typically a library or backend op you do not own. For your own Python
    functions, prefer `custom_vjp`, whose rule is plain Python.

    The decorated function is an adjoint *template* written in Tangent's DSL:
    its first parameter is the primal output, the rest are the primal inputs
    (matched by position to the differentiated call), and it assigns each
    input's gradient into `d[<param>]` from the output gradient `d[<output>]`::

        @tangent.register_adjoint(numpy.hypot)
        def hypot_adjoint(z, x, y):
            d[x] = d[z] * x / z
            d[y] = d[z] * y / z

    Registering a rule shadows any built-in rule for `func`. The template must
    have retrievable source (define it in a module, not the REPL), like any
    function Tangent differentiates.

    Returns:
      A decorator that registers its argument as the adjoint template and
      returns it unchanged.
    """
    from tangent import grads

    def decorator(template):
        grads.adjoints[func] = template
        grads.UNIMPLEMENTED_ADJOINTS.discard(func)
        return template

    return decorator


def register_tangent(func):
    """Register a forward-mode (tangent) template for an existing callable.

    The forward-mode counterpart of `register_adjoint`. The decorated template
    receives the primal output followed by the primal inputs, and assigns the
    output's tangent into `d[<output>]` from the input tangents `d[<input>]`::

        @tangent.register_tangent(numpy.hypot)
        def hypot_tangent(z, x, y):
            d[z] = (x * d[x] + y * d[y]) / z

    Returns:
      A decorator that registers its argument as the tangent template and
      returns it unchanged.
    """
    from tangent import tangents

    def decorator(template):
        tangents.tangents[func] = template
        tangents.UNIMPLEMENTED_TANGENTS.discard(func)
        return template

    return decorator
