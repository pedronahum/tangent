# Custom Gradients

Sometimes Tangent cannot, or should not, derive a gradient by transforming a
function's body: the function calls into a C extension or external solver, you
want a numerically better rule than the one differentiation would produce, or
you need to attach a gradient to a library op Tangent does not already know.

Tangent gives you two documented ways to supply the rule yourself. Pick by
**who owns the function**:

| You want to…                                              | Use |
|-----------------------------------------------------------|-----|
| Give *your own* function a custom rule, or wrap a black box you call | [`custom_vjp`](#custom_vjp-your-own-functions-and-black-boxes) |
| Attach a rule to a *library op you do not own* (`numpy.hypot`, a backend function) | [`register_adjoint` / `register_tangent`](#register_adjoint-library-ops-you-dont-own) |

## `custom_vjp`: your own functions and black boxes

Decorate the function with `@tangent.custom_vjp`, then register its
reverse-mode rule with `.defvjp`. The rule is **plain Python** — no template
DSL — and receives the incoming cotangent `g`, the primal output `ans`, and
then the primal arguments; it returns one gradient per argument:

```python
import numpy as np
import tangent

@tangent.custom_vjp
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

@gelu.defvjp
def gelu_vjp(g, ans, x):            # (cotangent, primal output, *primal args)
    cdf = 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    pdf = np.exp(-0.5 * x * x) * 0.3989422804
    return g * (cdf + x * pdf)      # one gradient (unary function)

grad_gelu = tangent.grad(gelu)
```

For a function of several arguments, return a **tuple** of gradients, one per
argument, in order.

### Wrapping a black box

Because Tangent never transforms the body of a `custom_vjp` function — it calls
it as-is in the primal and defers to your rule in reverse — the body may do
anything, including call code Tangent cannot see:

```python
import scipy.optimize
import tangent

@tangent.custom_vjp
def solve(a, b):
    # opaque to Tangent: a C/Fortran routine with no Python source to transform
    return scipy.optimize.brentq(lambda t: a * t * t - b, 0.0, 10.0)

@solve.defvjp
def solve_vjp(g, ans, a, b):
    # implicit-function theorem: d(root)/d(params) from the residual
    dr_dt = 2.0 * a * ans
    return (g * (-ans * ans / dr_dt), g * (1.0 / dr_dt))
```

### Forward mode

Register the optional forward-mode rule with `.defjvp`. It receives the primal
output `ans`, the primal arguments, then the argument tangents, and returns the
output tangent:

```python
@gelu.defjvp
def gelu_jvp(ans, x, dx):           # (primal output, *primal args, *arg tangents)
    cdf = 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    pdf = np.exp(-0.5 * x * x) * 0.3989422804
    return dx * (cdf + x * pdf)
```

`custom_vjp` functions must have a fixed positional signature (no
`*args`/`**kwargs`).

## `register_adjoint`: library ops you don't own

When the function is one you cannot decorate — `numpy.hypot`, a backend op —
attach a rule to the existing callable with `tangent.register_adjoint` (reverse
mode) and, optionally, `tangent.register_tangent` (forward mode). These use
Tangent's **template DSL**: the decorated function is parsed, not executed, so
its body is written in terms of `d[...]` derivative markers.

```python
import numpy as np
import tangent

@tangent.register_adjoint(np.hypot)
def hypot_adjoint(z, x, y):         # first param: primal output; rest: inputs
    d[x] = d[z] * x / z             # assign each input's gradient from d[output]
    d[y] = d[z] * y / z

@tangent.register_tangent(np.hypot)
def hypot_tangent(z, x, y):
    d[z] = (x * d[x] + y * d[y]) / z

def norm(x, y):
    return np.hypot(x, y)

tangent.grad(norm, wrt=(0, 1))(3.0, 4.0)   # -> (0.6, 0.8)
```

Inside a template, `d[v]` is the derivative associated with variable `v`, and
the parameter names bind to the differentiated call's output and arguments in
order. A registered rule shadows any built-in rule for that op. Like any
function Tangent differentiates, a template must have **retrievable source** —
define it in a module, not the REPL.

These are the same low-level decorators (`tangent.grads.adjoint` /
`tangent.tangents.tangent_`) that Tangent's own backend extensions use to
register the built-in NumPy/JAX/PyTorch rules; `register_adjoint` and
`register_tangent` are their public, documented spelling.

## Freezing part of a computation

`tangent.stop_gradient(x)` is the identity in the primal and has a zero
derivative in both modes — the standard way to treat part of a computation as a
constant:

```python
def loss(x):
    # x flows to the output, but no gradient flows back through the frozen term
    return x + tangent.stop_gradient(x) * 5.0
```
