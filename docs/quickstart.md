# Quickstart

## Your first gradient

```python
import tangent

def f(x):
    return x ** 3 - 2 * x ** 2 + 3 * x - 1

df = tangent.grad(f)
df(2.0)   # 7.0  (= 3·4 - 4·2 + 3)
```

`tangent.grad(f)` parses `f`'s source, applies reverse-mode automatic
differentiation to the AST, and compiles a brand-new Python function that
returns `df/dx`.

## Read the generated code

Pass `verbose=1` and Tangent prints the gradient function it built:

```python
df = tangent.grad(f, verbose=1)
```

This is the feature everything else is built around: the backward pass is
ordinary Python. If a gradient surprises you, *read it*.

## Arrays and backends

Write the function against your favorite array library; `tangent.grad` is the
same call:

```python
import numpy as np

def loss(x):
    return np.sum(np.tanh(x) ** 2)

dloss = tangent.grad(loss)
dloss(np.array([0.5, -1.0, 2.0]))
```

The same works for `jax.numpy`, TensorFlow eager, PyTorch, `keras.ops`, and
tinygrad tensors — see [Backends](backends.md).

## Real Python control flow

Loops, conditionals, early exits, and list building all differentiate:

```python
def pieces(x):
    total = 0.0
    for v in x:
        if v < 0.0:
            break            # early exit: gradient counts executed iterations
        total = total + v * v
    return total

def build(x):
    ys = [v * v for v in x]  # dynamic list comprehension
    s = 0.0
    for i in range(len(ys)):
        s = s + ys[i]
    return s
```

The full construct-by-construct matrix lives in
[Python Language Support](features/PYTHON_FEATURE_SUPPORT.md).

## Multiple arguments

`wrt` selects which arguments to differentiate with respect to:

```python
def g(a, b):
    return a * a * b

dg_da        = tangent.grad(g)                 # d/da (default: wrt=(0,))
dg_dab       = tangent.grad(g, wrt=(0, 1))     # returns (d/da, d/db)
dg_da(2.0, 3.0)      # 12.0
dg_dab(2.0, 3.0)     # (12.0, 4.0)
```

## Forward mode and more

```python
# Forward mode (JVP): efficient for few inputs / many outputs
df = tangent.autodiff(f, mode='forward')
df(2.0, 1.0)                       # directional derivative with seed 1.0

# Second derivatives: differentiate the gradient
ddf = tangent.grad(tangent.grad(f))
ddf(2.0)                           # 8.0

# Keep the function value alongside the gradient
df = tangent.grad(f, preserve_result=True)
grad_val, val = df(2.0)
```

See [Higher-Order Derivatives](higher-order.md) for Hessian-vector products
and forward-over-reverse.

## Memory-bounded gradients

For long loops carrying large state, segment checkpointing trades
recomputation for memory:

```python
df = tangent.grad(simulate, checkpoint=True)   # O(√n) peak tape memory
```

See the [Gradient Checkpointing](checkpointing_user_guide.md) guide.


## Notebooks, the REPL, and dynamic code

Tangent reads your function's source. In a Jupyter notebook this works
out of the box. In the plain Python REPL, in `exec`-generated code, or
if the defining file may be moved or deleted, capture the source at
definition time:

```python
@tangent.function        # grabs the source now
def f(x):
    return x * x

tangent.grad(f)(3.0)
```

For code with no retrievable source at all, pass it explicitly:

```python
tangent.grad(f, source="def f(x):\n    return x * x")
```

`python -m tangent doctor` diagnoses source-retrieval problems (and the
old-package collision).
