# Tangent

**Source-to-source automatic differentiation for Python.**

Tangent takes your Python function and generates a *new Python function* that
computes its gradient — code you can read, print, step through in a debugger,
and reason about. No graphs, no tapes hidden inside a C++ runtime, no black
boxes.

```python
import tangent

def f(x):
    return x ** 3 - 2 * x ** 2 + 3 * x - 1

df = tangent.grad(f, verbose=1)   # prints the generated gradient code
print(df(2.0))                    # f'(2) = 7.0
```

```python
# What verbose=1 prints - the gradient is just Python:
def dfdx(x, bf=1.0):
    ...
    bx = 3 * x ** 2 * bf - 4 * x * bf + 3 * bf
    return bx
```

---

## Why Tangent?

<div class="grid cards" markdown>

- :material-file-code:{ .lg .middle } **Readable gradients**

    ---

    The derivative of your function is ordinary Python source. Inspect it,
    profile it, or paste it into a code review.

- :material-bug:{ .lg .middle } **Debuggable**

    ---

    Set a breakpoint *inside* the backward pass. Print an intermediate
    adjoint. No other autodiff system lets you do this naturally.

- :material-layers-triple:{ .lg .middle } **One API, six backends**

    ---

    The same `tangent.grad` differentiates code written against NumPy, JAX,
    TensorFlow, PyTorch, Keras 3, or tinygrad.

- :material-language-python:{ .lg .middle } **Real Python**

    ---

    Loops (with `break`/`continue`/early `return`), conditionals, closures,
    classes, list building, comprehensions, tuple unpacking, containers
    in and out.

- :material-sigma:{ .lg .middle } **Higher-order**

    ---

    Forward and reverse mode, second and third derivatives,
    Hessian-vector products, forward-over-reverse.

- :material-memory:{ .lg .middle } **√n checkpointing**

    ---

    `grad(f, checkpoint=True)` trades recomputation for memory:
    96.8% peak-tape reduction measured, with bit-identical gradients.

</div>

---

## Reject-or-be-correct

Tangent's guiding principle: a construct either **differentiates correctly in
every mode** — verified against finite differences by a 79,000+ case test
suite — or it is **rejected with a clear, actionable error**. Never a silently
wrong gradient.

## At a glance

| | |
|---|---|
| **Install** | `pip install tangent-ad` — see [Installation](installation.md) |
| **Import** | `import tangent` (the import name is unchanged) |
| **Python** | 3.9 – 3.13 |
| **License** | Apache 2.0 |
| **Lineage** | Created by [Google Research](https://github.com/google/tangent) (2017); revived, completed, and maintained by [@pedronahum](https://github.com/pedronahum) |

[Get started :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }
[Language support matrix](features/PYTHON_FEATURE_SUPPORT.md){ .md-button }
