# Compile-Time Shape Checking

`tangent.check_shapes` catches the class of bug that finite differences only
reveal at run time - rank, broadcast, and matmul-dimension mistakes - *before*
you run anything, and points at the offending line.

```python
import numpy as np
import tangent

def layer(x, w, b):
    return np.tanh(x @ w) + b     # b has the wrong shape

tangent.check_shapes(layer, np.zeros((8, 4)), np.zeros((4, 16)), np.zeros((8,)))
# tangent.errors.ShapeError: Incompatible shapes for add: cannot broadcast
#   (8, 16) and (8,)
#     at layer.py:5: return np.tanh(x @ w) + b
```

## How it works

The function runs on **abstract arrays** - `ShapedArray` objects that carry a
shape and dtype but no data. NumPy's dispatch protocols
(`__array_ufunc__`/`__array_function__`) route real operations through the
checker, which applies NumPy's actual broadcasting and linear-algebra rules.
Because there is no data, it is cheap even for huge shapes, and the error names
the source line via the traceback.

## No false positives

An operation whose shape rule is not modeled yields an *unknown* shape
(dimensions become `None`, which is compatible with anything) rather than a
guessed one. The checker only ever reports a mismatch it is certain about, so a
clean `check_shapes` never masks a real bug and a modeled op never invents one.

## Inputs

Pass real arrays (only their `.shape`/`.dtype` are read), `tangent.ShapedArray`
instances, or scalars:

```python
from tangent import ShapedArray
tangent.check_shapes(layer, ShapedArray((8, 4)), ShapedArray((4, 16)), ShapedArray((16,)))
```

## Scope

Covers the forward (primal) function: elementwise ops and broadcasting,
`matmul`/`@`/`dot`, `reshape`, `transpose`/`.T`, reductions with `axis`,
`concatenate`/`stack`/`where`, indexing, `len()`, and iteration. Reductions and
ufuncs cover most NumPy code; unmodeled functions degrade to unknown shapes.
