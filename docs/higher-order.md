# Higher-Order Derivatives

Because Tangent's output is ordinary Python, higher-order differentiation is
compositional: differentiate the generated function again.

## Second and third derivatives

```python
import tangent

def f(x):
    return x ** 4

df   = tangent.grad(f)                       # 4x³
ddf  = tangent.grad(tangent.grad(f))         # 12x²
dddf = tangent.grad(tangent.grad(tangent.grad(f)))   # 24x

ddf(2.0)    # 48.0
```

Second and third derivatives are fully supported and tested (including
through loops, `break`/`continue`, list building, and container arguments).
Fourth order and beyond is not yet reliable.

## Forward over reverse: Hessian-vector products

The standard trick for `H·v` without materializing the Hessian — reverse mode
for the gradient, forward mode through it:

```python
import numpy as np

def loss(x):
    return np.sum(np.tanh(x) ** 2)

grad_f = tangent.grad(loss)                       # reverse
hvp    = tangent.autodiff(grad_f, mode='forward') # forward over reverse

x = np.array([0.5, -1.0, 2.0])
v = np.array([1.0, 0.0, 0.0])
hvp(x, v)    # H(x) @ v
```

## Forward mode directly

```python
df = tangent.autodiff(f, mode='forward')
df(2.0, 1.0)          # JVP: derivative in direction of the seed (here 1.0)
```

Forward mode pairs each primal statement with its tangent statement — the
generated code interleaves them, and `verbose=1` shows both.

## Choosing a composition

| Quantity | Composition |
|---|---|
| Gradient of a scalar loss | `grad(f)` |
| Jacobian-vector product | `autodiff(f, mode='forward')` |
| Hessian-vector product | `autodiff(grad(f), mode='forward')` |
| Full second derivative (scalar) | `grad(grad(f))` |

!!! note "Checkpointed gradients are first-order"

    `grad(f, checkpoint=True)` produces a gradient function that is exact
    but not itself differentiable — use plain `grad` for derivatives you
    intend to differentiate again.
