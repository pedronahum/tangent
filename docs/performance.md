# Performance & Compilation

The generated adjoint is ordinary Python — that's the product — but ordinary
Python is also a performance ceiling. Tangent's answer has three parts:
compile the adjoint, cache the compilation, and measure honestly.

## Lower the adjoint: `compile=`

```python
df = tangent.grad(f)                      # readable Python (default)
df = tangent.grad(f, compile='jax')       # same adjoint, jax.jit-fused
df = tangent.grad(f, compile='torch')     # torch.compile
df = tangent.grad(f, compile='tinygrad')  # TinyJit
```

You inspect Python, you run compiled. The readable source stays attached
(`df.__tangent_source__`, `verbose=1`, [`tangent.explain`](debugging.md)),
and the cache stores the plain-Python function so every `compile=` choice
shares one differentiation.

Caveats: `compile='jax'` requires the primal to be written against
`jax.numpy` and traceable (fixed loops unroll; data-dependent Python control
flow does not trace). `compile='torch'` permits graph breaks, so tape
operations degrade gracefully. Numba is deliberately not offered — the
generated code calls Python-level tape helpers its nopython mode cannot
compile.

## The compile bill, and the caches that pay it

Source transformation always pays a compile cost. Two caches amortize it:

- **In-memory LRU** — repeated `grad(f)` calls in one process are ~µs.
- **Persistent disk cache** — the *generated source* is stored under
  `~/.cache/tangent-ad/` keyed by function source, options, and Tangent
  version, so new processes skip parsing, differentiation, and optimization
  entirely (measured ~13× faster cold start). Only source is stored, never
  pickles; on load, every global the compiled code references is verified to
  resolve, and any problem falls back to full recompilation. Disable with
  `TANGENT_DISK_CACHE=0`; relocate with `TANGENT_CACHE_DIR`.

## Measured against the frameworks users would switch from

`benchmarks/vs_frameworks.py`, CPU (aarch64), best-of-50 eval; compile/first
call reported separately:

**3-layer MLP loss (128×128, batch 64) — tensor-heavy:**

| | compile + first call | eval |
|---|---:|---:|
| tangent (Python adjoint) | 32 ms | 0.181 ms |
| tangent `compile='jax'` | 60 ms | **0.075 ms** |
| `jax.grad` + `jit` | 29 ms | 0.077 ms |
| `jax.grad` (no jit) | 252 ms | 1.044 ms |
| `torch.autograd` | 63 ms | 0.238 ms |

The lowered Tangent adjoint **matches `jax.grad`+`jit`** — the readable
source costs nothing at run time once compiled. Even the plain-Python adjoint
beats eager PyTorch and unjitted JAX here.

**Scalar recurrence, 1000 data-dependent steps — control-flow-heavy:**

| | compile + first call | eval |
|---|---:|---:|
| tangent (Python adjoint) | 61 ms | **3.5 ms** |
| `torch.autograd` (eager) | 34 ms | 22.2 ms |
| `jax.grad` + `jit` (unrolled) | **6,802 ms** | 0.017 ms |

This is the workload source transformation exists for: 6× faster than eager
autograd per evaluation, at 1% of JAX's unroll-compile cost. (If you evaluate
the same shapes thousands of times, JAX's unrolled kernel eventually wins —
the table shows exactly where the crossover economics live.)

## What the optimizer does

The generated code runs through tape-aware dead-code elimination, constant
folding, and assignment propagation by default (`optimized=False` to see the
naive adjoint), with optional SymPy-based algebraic simplification and CSE
(`optimizations={'cse': True, 'algebraic': True}`). DCE is tape-aware: a
push is only removed together with the pop that consumes it, so optimization
never corrupts higher-order derivatives.

## Shrinking the tape: `tape_liveness`

Reverse mode saves each reassigned primal on the tape so the adjoint can read
it back. Many of those reads are **shape-only**: the value is only ever the
`like` of `unbroadcast` or the argument of `init_grad`, which consult its shape
and dtype but never its data. `optimizations={'tape_liveness': True}` stores a
lightweight shape carrier for exactly those entries instead of the whole array,
shrinking that tape slot from O(size) to O(ndim):

```python
def loop(x):
    s = x
    for i in range(50):
        a = s + 1.0            # `a` is read in reverse only for its shape
        s = a * 0.5 + s * 0.5
    return np.sum(s * s)

df = tangent.grad(loop, optimizations={'tape_liveness': True})
```

On a 20,000-element array over 50 iterations this cuts peak gradient memory by
~90%. The rewrite is proven safe by reaching-definition analysis - it converts
a variable only when *every* pop-reached use is shape-only, so any value the
adjoint reads arithmetically stays fully taped - and the gradient it produces
is identical to the default. It is opt-in (off by default) and targets the
NumPy execution path; it composes with `√n` checkpointing and `compile=`.

## Fusing a straight-line segment: `coarsening`

For a straight-line stretch of elementwise arithmetic, `optimizations={'coarsening': True}`
differentiates the whole segment once symbolically (SymPy) and emits a single
vector-Jacobian-product expression per input instead of one adjoint statement
per primitive. Bound to the primal's backend, that one expression is handed to
JAX or PyTorch as a **single fused kernel** under `compile=`:

```python
def f(x):
    a = jnp.sin(x)
    return jnp.sqrt(jnp.exp(a))

df = tangent.grad(f, optimizations={'coarsening': True}, compile='jax')
```

It is opt-in and applies only to coarsenable segments (elementwise straight-line
code on NumPy/JAX/PyTorch); anything else - control flow, reductions,
TensorFlow/Keras - falls back to the standard pipeline unchanged. See the
[coarsening deep-dive](https://github.com/pedronahum/tangent/blob/master/docs/optimizations/COARSENING.md)
for the supported op set.
