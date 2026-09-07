# List Comprehension Support in Tangent

**Status**: ✅ Supported over compile-time-constant iterables (unrolled); dynamic iterables are rejected with a clear error

## Summary

A comprehension over a compile-time-constant iterable — a constant
`range(...)` call or a list/tuple literal — is **unrolled into a plain list
literal** before differentiation, so it differentiates exactly like the
literal it denotes:

```python
[x * i for i in range(4)]            ->  [x * 0, x * 1, x * 2, x * 3]
[x * c for c in [1.0, 2.0, 3.0]]     ->  [x * 1.0, x * 2.0, x * 3.0]
[x * i for i in range(4) if i > 1]   ->  [x * 2, x * 3]   # filters decided at compile time
```

`if` filters are supported when they can be decided at compile time (i.e. once
the loop variable is substituted, the clause is a closed constant expression).

The unrolling lives in
[`tangent/comprehension_desugar.py`](../../tangent/comprehension_desugar.py)
(which also handles set/dict comprehensions) and is exercised by
[`tests/test_comprehensions.py`](../../tests/test_comprehensions.py).

## Dynamic iterables are rejected

A comprehension over a runtime value cannot be unrolled:

```python
def f(x):
    vals = [v * 3.0 for v in x]   # x is a runtime array
    return np.sum(vals)

tangent.grad(f)   # TangentParseError: List comprehensions over dynamic
                  # iterables are not supported
```

The language fence ([`tangent/fence.py`](../../tangent/fence.py)) raises a
`TangentParseError` pointing at the comprehension, with workarounds in the
message.

### Why not lower to an explicit loop?

An earlier implementation desugared these comprehensions into an
`.append()` loop:

```python
vals = []
for v in x:
    _tmp = v * 3.0
    vals.append(_tmp)
```

That lowering was removed: the `.append()` call is opaque to the adjoint
machinery, so the loop's per-iteration binding was never differentiated. In
assignment position the gradient came back silently as all zeros, and in
return position (`return np.sum([...])`) the transform crashed in naming with
an opaque `AttributeError`. Rejecting up front is the only honest behavior
until list mutation is differentiable.

## Workarounds

```python
# ❌ Rejected: dynamic iterable
ys = [v * 3.0 for v in xs]

# ✅ Vectorized NumPy operation
ys = xs * 3.0

# ✅ Explicit accumulation loop (differentiates correctly)
total = 0.0
for v in xs:
    total = total + v * 3.0
```

## Related

- [PYTHON_FEATURE_SUPPORT.md](PYTHON_FEATURE_SUPPORT.md#comprehensions) — full feature matrix
- [FOR_LOOP_SUPPORT.md](FOR_LOOP_SUPPORT.md) — explicit loop support
