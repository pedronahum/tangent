# Straight-Line Coarsening

**Status:** prototype, opt-in via `optimizations={'coarsening': True}`

Coarsening is an alternative reverse-mode strategy inspired by *Integrating
symbolic and algorithmic automatic differentiation* (Shen et al., OOPSLA
2021). Instead of differentiating a function one primitive operation at a
time, it treats a whole straight-line segment as a single symbolic function,
differentiates that function once, and emits the resulting vector-Jacobian
product (VJP) directly.

## Why

The default per-op reverse mode emits, for every primitive op, an adjoint
statement plus the tape `push`/`pop` bookkeeping needed to save and replay
intermediate values. For a long straight-line stretch of arithmetic this is
the dominant cost, and it also hides simplification opportunities that only
become visible across several operations (e.g. `sin(x)**2 + cos(x)**2 -> 1`).

Coarsening avoids both:

- **Fewer statements** — one symbolic VJP replaces the per-op adjoints.
- **Less tape traffic** — intermediates inside the segment are inlined away,
  so they do not need to be pushed and popped.
- **Cross-op simplification** — the segment is handed to SymPy, which folds
  identities before the adjoint is lowered back to code.

## How it works

`tangent/optimizations/coarsening.py` builds on the SymPy round-trip from
`tangent/optimizations/algebraic_simplification.py`:

1. **Validate** the function is a straight-line segment: single-target `Name`
   assignments followed by one `return`, no control flow, and only the
   elementwise primitives the SymPy converters understand. Attribute callees
   such as `np.sin` are accepted; attribute *data* access is rejected.
2. **Inline** every intermediate into the return expression, yielding one
   SymPy expression over the input symbols.
3. **Differentiate** that expression once with respect to each input
   (`sp.diff`), optionally `sp.simplify`-ing the result.
4. **Lower** each `seed * d(output)/d(input)` back to an AST and assemble the
   adjoint function `d<name>(inputs..., seed) -> (adjoints...)`.

## Using it through `tangent.grad`

```python
df = tangent.grad(f, optimizations={'coarsening': True})
```

When enabled for a reverse-mode gradient, Tangent first checks whether `f` is
a coarsenable straight-line segment of NumPy elementwise arithmetic. If so it
emits the single symbolic VJP; otherwise it transparently falls back to the
standard per-op pipeline. Anything that is not coarsenable — control flow,
reductions such as `np.sum`, non-NumPy backends (`jnp`/`torch`/`tf`/`kops`),
varargs, multi-output configurations (`output_index`/`output_weights`), or
`preserve_result` — takes the fallback path, so enabling the option never
changes correctness.

Because the lowered adjoint references elementwise primitives by bare name
(`cos`, `sin`, ...), the compile namespace is extended with their NumPy
implementations, and the gradient cache is bypassed when coarsening is
requested (the cache key does not encode the `optimizations` dict).

## Direct use

The prototype can also be applied to a function AST without going through
`tangent.grad`:

```python
import gast
from tangent.optimizations.coarsening import apply_coarsening

func_ast = gast.parse(source).body[0]
adj_ast = apply_coarsening(func_ast)   # None if not coarsenable
```

## Limitations and future work

- **NumPy only.** The lowered adjoint uses bare elementwise names bound to
  NumPy; other backends fall back to the standard pipeline. Making the
  lowering backend-aware (JAX/PyTorch/TensorFlow/Keras) is the main follow-up.
- **No reductions.** `np.sum`, `np.mean`, etc. are not part of the elementwise
  subset, so functions that reduce fall back.
- **Scalar symbolic model.** SymPy models each value as a scalar symbol, which
  is exactly the elementwise regime; it is not a shape/broadcasting analysis.
- **Not yet wired into the default pipeline.** It is opt-in; deciding when it
  is profitable by default (e.g. segment length heuristics) is open work.

See `tests/test_coarsening.py` for numerical checks against finite
differences, `tangent.grad`, and the fallback behaviour.
