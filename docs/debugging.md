# Debugging Gradients

Tangent's core advantage: when a gradient looks wrong, you can *read* it.

## 1. Print the generated code

```python
df = tangent.grad(f, verbose=1)
```

The generated function is annotated with comments tying each adjoint block to
the primal statement it differentiates:

```python
# Grad of: y = x * x
_bx = tangent.unbroadcast(by * x, x)
_bx2 = tangent.unbroadcast(by * x, x)
bx = tangent.add_grad(bx, _bx)
bx = tangent.add_grad(bx, _bx2)
```

`optimized=False` disables the cleanup passes if you want to see the naive,
fully-explicit derivative.

## 2. Step through it

The gradient is a plain function — `pdb`, IDE breakpoints, and `print`
all work inside the backward pass:

```python
import pdb

df = tangent.grad(f)
pdb.runcall(df, 2.0)     # step into the adjoint statements
```

## 3. Check against finite differences

The pattern the test suite uses everywhere:

```python
def numeric_grad(f, x, eps=1e-6):
    return (f(x + eps / 2) - f(x - eps / 2)) / eps

assert abs(tangent.grad(f)(2.0) - numeric_grad(f, 2.0)) < 1e-5
```

## 4. Inject debugging into the backward pass

`insert_grad_of` lets you run arbitrary code *at the corresponding point of
the backward pass* — print an intermediate gradient, clip it, or assert on
it:

```python
from tangent import insert_grad_of

def f(x):
    y = x * x
    with insert_grad_of(y) as dy:
        print('gradient flowing into y:', dy)
    return y * 3.0
```

## 5. Visualize gradient flow

With the `viz` extra installed:

```python
tangent.visualize_gradient_flow(f, x)   # matplotlib figure of the flow
```

## Understanding errors

Tangent fails loudly and specifically:

| Error | Meaning |
|---|---|
| `TangentParseError` | The function uses a construct Tangent rejects (the message includes a suggested rewrite) |
| `ReverseNotImplementedError` / `ForwardNotImplementedError` | A called op has no registered derivative in that mode — register one or rewrite |
| `GradientNotFoundError` | A called function has neither a derivative nor retrievable source |

The philosophy is *reject-or-be-correct*: if your function compiles, its
gradient is trustworthy; if it does not, the error tells you why and what to
do instead.
