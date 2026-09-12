# API Reference

The public surface is small: two entry points cover almost all use, plus
registration hooks for extending Tangent with new gradients and tensor types.

## Differentiation

::: tangent.grad_util.grad

::: tangent.grad_util.autodiff

::: tangent.grad_util.vjp

::: tangent.grad_util.jvp

## Differentiable ODEs

::: tangent.ode.odeint

## Custom gradients

See the [Custom Gradients guide](custom_gradients.md) for a walkthrough of when
to use each of these.

::: tangent.custom_vjp.custom_vjp

::: tangent.custom_vjp.register_adjoint

::: tangent.custom_vjp.register_tangent

::: tangent.custom_vjp.stop_gradient

## Working in notebooks and the REPL

::: tangent.capture.function

## Shape checking

::: tangent.shape_check.check_shapes

## Debugging

::: tangent.explain.explain

::: tangent.explain.source_map

## Extending Tangent

### Registering gradient templates

To teach Tangent the gradient of a library op it does not already know, use the
public `tangent.register_adjoint` (reverse mode) and `tangent.register_tangent`
(forward mode) decorators:

```python
import numpy as np
import tangent

@tangent.register_adjoint(np.sinc)   # reverse rule: fills d[x] from d[y]
def asinc(y, x):
    d[x] = d[y] * (np.cos(np.pi * x) / x - y / x)

@tangent.register_tangent(np.sinc)   # forward rule: fills d[y] from d[x]
def tsinc(y, x):
    d[y] = d[x] * (np.cos(np.pi * x) / x - y / x)
```

Inside a template, `d[v]` denotes the derivative associated with variable
`v`; the template's parameter names bind to the call site's result and
arguments in order. (These wrap the lower-level `tangent.grads.adjoint` /
`tangent.tangents.tangent_` decorators that Tangent's own backend extensions
use.) For your own Python functions, prefer
[`custom_vjp`](custom_gradients.md), whose rule is plain Python.

### Runtime type registries

New tensor types participate in gradient bookkeeping through the dispatch
registries in `tangent.utils`:

::: tangent.utils.register_init_grad

::: tangent.utils.register_add_grad

::: tangent.utils.register_unbroadcast

::: tangent.utils.register_unreduce

::: tangent.utils.register_matmul_grad

### Frontend passes

::: tangent.passes.register_pass

## Utilities

::: tangent.backend_status

::: tangent.utils.checkpoint

`tangent.grads.UNIMPLEMENTED_ADJOINTS` / `tangent.tangents.UNIMPLEMENTED_TANGENTS`
hold the ops known to lack a derivative in each mode; calls to them raise the
clean `ReverseNotImplementedError` / `ForwardNotImplementedError`.
