# API Reference

The public surface is small: two entry points cover almost all use, plus
registration hooks for extending Tangent with new gradients and tensor types.

## Differentiation

::: tangent.grad_util.grad

::: tangent.grad_util.autodiff

::: tangent.grad_util.vjp

::: tangent.grad_util.jvp

## Extending Tangent

### Registering gradient templates

Adjoints (reverse mode) and tangents (forward mode) are registered with
decorators from `tangent.grads` and `tangent.tangents`:

```python
from tangent.grads import adjoint
from tangent.tangents import tangent_
import numpy as np

@adjoint(np.sinc)                 # reverse rule: fills d[x] from d[y]
def asinc(y, x):
    d[x] = d[y] * (np.cos(np.pi * x) / x - y / x)

@tangent_(np.sinc)                # forward rule: fills d[y] from d[x]
def tsinc(y, x):
    d[y] = d[x] * (np.cos(np.pi * x) / x - y / x)
```

Inside a template, `d[v]` denotes the derivative associated with variable
`v`; the template's parameter names bind to the call site's result and
arguments in order.

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

`tangent.grads.UNIMPLEMENTED_ADJOINTS` / `tangent.tangents.UNIMPLEMENTED_TANGENTS`
hold the ops known to lack a derivative in each mode; calls to them raise the
clean `ReverseNotImplementedError` / `ForwardNotImplementedError`.
