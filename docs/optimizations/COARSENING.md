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
a coarsenable straight-line segment of elementwise arithmetic. If so it emits
the single symbolic VJP; otherwise it transparently falls back to the standard
per-op pipeline. Anything that is not coarsenable — control flow, reductions
such as `np.sum`, TensorFlow/Keras primals, varargs, multi-output
configurations (`output_index`/`output_weights`), or `preserve_result` — takes
the fallback path, so enabling the option never changes correctness.

## Backend kernel handoff

The lowered adjoint references elementwise primitives by bare name (`cos`,
`sin`, ...), and the compile namespace binds those names to the backend the
**primal** is written against:

- **NumPy** primals bind to `numpy.*` and run as plain NumPy.
- **JAX** (`jnp.*`) and **PyTorch** (`torch.*`) primals bind to that backend's
  ops. Because the coarsened VJP is a *single expression* per input in the
  backend's own ops, pairing it with `compile='jax'` (or `torch.compile`) hands
  the whole segment to the backend as one fused kernel — the point of
  coarsening for array backends:

  ```python
  import jax.numpy as jnp
  import tangent

  def f(x):
      a = jnp.sin(x)
      return jnp.sqrt(jnp.exp(a))

  df = tangent.grad(f, optimizations={'coarsening': True}, compile='jax')
  ```

- **TensorFlow / Keras** primals are not coarsened (their elementwise VJP
  semantics are not validated here) and fall back to the standard pipeline.

The gradient cache is bypassed when coarsening is requested (the cache key does
not encode the `optimizations` dict).

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

- **JAX / PyTorch / NumPy.** The lowered adjoint binds its bare elementwise
  names to the primal's backend, so JAX and PyTorch primals get a single fused
  VJP kernel under `compile=`. TensorFlow / Keras are not yet coarsened
  (their elementwise VJP semantics are unvalidated) and fall back.
- **Narrow elementwise op set.** Coarsened ops are `sin`, `cos`, `tan`, `exp`,
  `log`, `sqrt`, `arcsin`, `arccos`, `arctan`. Others (`abs`, `sinh`, `cosh`,
  `tanh`) fall back because their derivatives reintroduce a function the
  SymPy-to-AST lowering cannot emit (e.g. `tanh' = 1 - tanh^2`, `sinh' = cosh`,
  and SymPy's derivative of `Abs` over a complex symbol is not lowerable).
  `tests/test_coarsening.py::test_elementwise_support_set_is_exact` pins the
  set. (Enabling the inverse-trig ops also surfaced and fixed a latent bug in
  the lowering: `isinstance(expr, sp.sqrt)` raised `TypeError` because
  `sp.sqrt` is a function, not a type, which silently broke every function
  check after it - including `tan`.)
- **No reductions.** `np.sum`, `np.mean`, etc. are not part of the elementwise
  subset, so functions that reduce fall back.
- **Scalar symbolic model.** SymPy models each value as a scalar symbol; it is
  not a shape/broadcasting analysis. Consequently a coarsened gradient is the
  scalar symbolic derivative: for array inputs it yields a scalar rather than
  an element-wise array, so coarsening is best applied to scalar-valued
  kernels (use the standard pipeline for array-shaped gradients).
- **Not yet wired into the default pipeline.** It is opt-in; deciding when it
  is profitable by default (e.g. segment length heuristics) is open work.

See `tests/test_coarsening.py` for numerical checks against finite
differences, `tangent.grad`, and the fallback behaviour.
