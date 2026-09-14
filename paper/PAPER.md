# Tangent 2: Multi-Backend Source-to-Source Automatic Differentiation in Modern Python

*A reproducible writeup. Every code block below runs — open it in Colab, or run
`python paper/reproduce.py`.*

## Abstract

Tangent is an automatic differentiation (AD) library that works by
**source-to-source transformation**: it reads a Python function and emits a new
Python function that computes its gradient. Unlike tracing frameworks (JAX,
PyTorch, TensorFlow), which build an opaque graph or tape at run time, the
gradient Tangent produces is ordinary Python you can read, debug, step through,
and even edit. This writeup describes *Tangent 2*, a modernized and extended
fork of Google's 2017 Tangent: one differentiation API spanning six array
backends (NumPy, JAX, TensorFlow, PyTorch, Keras 3, tinygrad), optional lowering
of the generated adjoint to a fused backend kernel, memory-bounded
differentiation of long simulations (adjoint ODEs and √n / online
checkpointing), an opt-in tape-liveness pass that stores only the shapes of
values the reverse sweep reads for shape, a symbolic straight-line coarsening
pass, a statically typed public API, and a debuggability toolkit (`explain`,
`source_map`, `insert_grad_of`). Its niche is not replacing tracing AD for large
tensor programs; it is making the derivatives of gnarly, loop- and
branch-heavy scientific and quantitative code **legible**.

## 1. Introduction

Automatic differentiation is the engine under modern machine learning and,
increasingly, scientific computing. The dominant libraries trace a program's
execution into a graph (TensorFlow) or a tape (PyTorch, JAX) and differentiate
that intermediate representation. This is fast and general, but the gradient is
a black box: when a derivative is wrong, or a simulator's sensitivity is
surprising, there is no readable adjoint to inspect.

Source-to-source AD takes the other path: transform the *source code*. The
output is a function you can open in an editor. Tangent, introduced by Google
Research in 2017, pioneered this for Python. Development stopped, and the
package name `tangent` on PyPI still points at the unmaintained 2017 release.

*Tangent 2* (distributed as `tangent-ad`, imported as `tangent`) revives and
extends the idea for the current Python and array-library ecosystem. This
writeup is itself reproducible: install the package and run the cells.

```python
# The one dependency for everything below. (JAX and PyTorch are optional and
# auto-detected; in Colab they are already present.)
import numpy as np
import tangent
print("tangent", tangent.__version__)
```

## 2. The core idea: gradients are source code

`tangent.grad(f)` returns a Python function. Its source is right there.

```python
def cubic(x):
    return x**3 - 2 * x**2 + 3 * x - 1

df = tangent.grad(cubic)
print("f'(2) =", df(2.0))          # 7.0
print()
print(df.__tangent_source__)        # the generated gradient, as readable Python
```

Higher-order derivatives are just differentiation applied again — the second
and third derivatives are themselves generated Python:

```python
def cube(x):
    return x**3

ddf = tangent.grad(tangent.grad(cube))   # d2/dx2 x^3 = 6x
dddf = tangent.grad(ddf)                 # d3/dx3 x^3 = 6
print("f''(2) =", ddf(2.0), "   f'''(2) =", dddf(2.0))
```

## 3. What Tangent 2 adds

### 3.1 One API across six backends

The same `tangent.grad` differentiates code written against NumPy, JAX,
TensorFlow, PyTorch, Keras 3, or tinygrad. A single op catalog registers each
backend's spelling of the shared elementwise, binary and reduction rules, so no
backend can silently drift. Here is the identical transform on three backends
(others auto-skip if not installed):

```python
analytic = lambda x: 2 * np.tanh(x) * (1 - np.tanh(x) ** 2)
x = np.array([0.5, -1.0, 2.0])

def np_loss(x):
    return np.sum(np.tanh(x) ** 2)
print("NumPy :", np.round(tangent.grad(np_loss)(x), 5), " (matches analytic:",
      np.allclose(tangent.grad(np_loss)(x), analytic(x)), ")")

try:
    import jax.numpy as jnp
    def jax_loss(x):
        return jnp.sum(jnp.tanh(x) ** 2)
    print("JAX   :", np.round(np.asarray(tangent.grad(jax_loss)(jnp.asarray(x))), 5))
except ImportError:
    print("JAX   : (not installed)")

try:
    import torch
    def torch_loss(x):
        return torch.sum(torch.tanh(x) ** 2)
    print("Torch :", np.round(tangent.grad(torch_loss)(torch.as_tensor(x)).numpy(), 5))
except ImportError:
    print("Torch : (not installed)")
```

### 3.2 Inspect Python, run compiled

Generated NumPy pays per-op interpreter overhead. `compile='jax'` (or
`torch.compile`, or tinygrad's `TinyJit`) lowers the *same* readable adjoint
through the backend's JIT — you inspect Python and run at `jax.grad`+`jit`
speed. A persistent disk cache amortizes the source transform across processes.

```python
try:
    import jax.numpy as jnp
    def jax_loss(x):
        return jnp.sum(jnp.tanh(x) ** 2)
    xj = jnp.asarray(x)
    plain = np.asarray(tangent.grad(jax_loss)(xj))
    jitted = np.asarray(tangent.grad(jax_loss, compile='jax')(xj))
    print("compile='jax' matches the plain adjoint:", np.allclose(plain, jitted, atol=1e-6))
except ImportError:
    print("(JAX not installed)")
```

### 3.3 Bounded memory for long simulations

Reverse mode records intermediates on a tape whose size grows with the number
of steps. Tangent 2 offers three ways to bound that:

- **`tangent.odeint`** differentiates an ODE solution by the continuous adjoint
  method — memory constant in the number of steps.
- **Checkpointing** (`tangent.grad(f, checkpoint=True)`) recomputes instead of
  storing: √n segment checkpointing for constant-bound loops, online
  (Stumm–Walther) checkpointing for data-dependent loops.
- **Tape-liveness** (opt-in) stores only the *shape* of values the reverse
  sweep reads for shape (the `like` of `unbroadcast`, the argument of
  `init_grad`) instead of the whole array. It is proven safe by
  reaching-definition analysis and leaves the gradient bit-identical:

```python
import tracemalloc

def array_loop(x):
    s = x
    for i in range(50):
        a = s + 1.0
        s = a * 0.5 + s * 0.5
    return np.sum(s * s)

xs = np.arange(20000.0)
off = tangent.grad(array_loop)
on = tangent.grad(array_loop, optimizations={'tape_liveness': True})

def peak(df):
    df(xs)
    tracemalloc.start(); df(xs); p = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    return p

p_off, p_on = peak(off), peak(on)
print("identical gradient:", np.allclose(off(xs), on(xs)))
print("peak memory: %.1f MB -> %.1f MB  (%.0f%% less)"
      % (p_off/1e6, p_on/1e6, 100*(p_off-p_on)/p_off))
```

### 3.4 Symbolic coarsening, fused into a backend kernel

For a straight-line stretch of elementwise math, Tangent can differentiate the
whole segment once symbolically (via SymPy) and emit a single vector-Jacobian
product instead of one adjoint statement per primitive. For a JAX or PyTorch
primal that VJP is emitted in the backend's own ops, so `compile=` fuses it
into one kernel. The coarsened gradient equals the per-op gradient:

```python
def kernel(a, b, c):
    return (np.exp(np.sin(a * b)) + c) * a

std = tangent.grad(kernel, wrt=(0, 1, 2))(0.7, 1.1, 0.3)
coa = tangent.grad(kernel, wrt=(0, 1, 2), optimizations={'coarsening': True})(0.7, 1.1, 0.3)
print("standard  :", np.round(std, 6))
print("coarsened :", np.round(coa, 6), " match:", np.allclose(std, coa))
```

### 3.5 Debuggability as a toolkit

`tangent.explain(f, x)` is the debuggability pitch in one call: it prints the
primal, the generated adjoint, a finite-difference check, the primal's
**data-flow graph**, and flags inputs that provably never affect the output.

```python
def model(x, unused_bias):
    a = x * x
    b = a * 3.0
    return np.sum(b)              # unused_bias never reaches the output

report = tangent.explain(model, np.array([1.0, 2.0]), 5.0, wrt=(0, 1))
print("\\nstatically dead arguments:", report['dead_inputs'])
```

Two more debuggability tools:

- **`tangent.source_map(df, f)`** links every generated adjoint line back to
  the primal statement it differentiates.
- **`tangent.insert_grad_of`** does *gradient surgery* — splice code into the
  backward pass (scale, clip, log, guard a gradient) that survives the
  optimizer. This is the demo tracing AD cannot copy:

```python
def f_clipped(x):
    y = x * x
    with tangent.insert_grad_of(y) as dy:
        dy = np.clip(dy, -5.0, 5.0)     # edit the gradient flowing into y
    return np.sum(y * y * y)

print("d/dx with clipped intermediate gradient:",
      tangent.grad(f_clipped)(np.array([2.0])))
```

### 3.6 A typed public API

Tangent 2 ships a `py.typed` marker (PEP 561), so `df = tangent.grad(f)` is a
typed callable to mypy/pyright rather than `Any`; for a single-argument function
the gradient even carries the input's type
(`grad(f: Callable[[T], Any]) -> Callable[..., T]`). And `python -m tangent
doctor` diagnoses an install — including detecting the dead 2017 `tangent`
package shadowing `tangent-ad`.

## 4. Reproducible case study: greeks of a Monte-Carlo rate model

Interest-rate desks price derivatives by simulating forward rates and need
*greeks* — sensitivities to every input. The traditional method is
bump-and-revalue: `2K` re-simulations for `K` inputs, each noisy. Below is a
stylized single-factor LIBOR Market Model caplet, written as a plain NumPy
Monte-Carlo loop whose spot-measure drift couples the whole curve each step
(one matmul against a triangular indicator). Tangent differentiates it *as
written* to give all deltas and vegas in **one reverse pass**, matching
bump-and-revalue.

```python
def caplet_price(F0, sigma, Z, K, tau, dt, L, e0, M, N, n_steps):
    F = F0 * np.ones((M, N))
    sqrt_dt = dt ** 0.5
    for step in range(n_steps):
        g = tau * sigma * F / (1.0 + tau * F)
        drift = sigma * (g @ L) - 0.5 * sigma * sigma      # state-dependent drift
        F = F * np.exp(drift * dt + sigma * (sqrt_dt * Z[step]))
    F_reset = F @ e0
    intrinsic = F_reset - K
    return np.sum(tau * (intrinsic * (intrinsic > 0.0)) / (1.0 + tau * F_reset)) / M

N, M, n_steps = 5, 4000, 8
tau, K, dt = 0.5, 0.03, 1.0 / 8
F0 = np.array([0.03, 0.032, 0.034, 0.035, 0.036])
sigma = np.array([0.20, 0.22, 0.24, 0.23, 0.21])
Z = np.random.RandomState(0).standard_normal((n_steps, M, 1))
L = np.tril(np.ones((N, N))); e0 = np.eye(N)[0]
C = (Z, K, tau, dt, L, e0, M, N, n_steps)

delta, vega = tangent.grad(caplet_price, wrt=(0, 1))(F0, sigma, *C)
h = 1e-6
fd_vega = np.array([(caplet_price(F0, sigma + h*np.eye(N)[i], *C)
                     - caplet_price(F0, sigma - h*np.eye(N)[i], *C)) / (2*h) for i in range(N)])
print("AD vega :", np.round(vega, 6))
print("FD vega :", np.round(fd_vega, 6))
print("max|AD - FD| =", np.max(np.abs(vega - fd_vega)))
```

The full notebooks — this LMM study, a projectile-with-drag ODE, and an SIR
epidemic model calibrated *through a data-dependent lockdown branch* — are in
the [notebook gallery](https://pedronahum.github.io/tangent/gallery/).

## 5. Evaluation

**Correctness.** The test suite runs ~80,000 parameterized cases with zero
failures, checking a shared op catalog across every installed backend against
both analytic gradients and an independent finite-difference oracle.

**Speed (honest).** On a 3-layer MLP, generated NumPy is within ~2× of
`jax.jit` and beats eager PyTorch and un-jitted JAX; `compile='jax'` matches
`jax.jit` (it lowers the same adjoint). On control-flow-heavy scalar code
(a 1000-step data-dependent recurrence) source transformation wins outright —
several-fold over eager PyTorch/autograd, while `jax.jit` only wins the eval
after a multi-second unroll. On the building-thermal simulation the optimized
gradient is faster than eager PyTorch (1.28×) and TensorFlow (1.58×), with
tape-aware DCE contributing ~2.6×. Reproduce these with the scripts in
`benchmarks/` (see the
[framework benchmarks](https://pedronahum.github.io/tangent/benchmarks/FRAMEWORK_BENCHMARKS/)
and [building simulation](https://pedronahum.github.io/tangent/benchmarks/BUILDING_SIMULATION_BENCHMARK/)
pages).

Tangent's advantage is not raw tensor throughput; it is readable, editable
derivatives of ordinary Python, and competitive speed on the control-flow-heavy
code tracing frameworks handle least gracefully.

## 6. Related work and positioning

Tracing frameworks — **JAX** (`jax.grad`), **PyTorch** (`torch.autograd`,
`torch.func`), **TensorFlow** — dominate large tensor AD and offer GPU/TPU
execution Tangent does not target. **HIPS autograd** pioneered Python
operator-overloading AD. The sibling project **kooderive** implements
LMM-style Monte Carlo in JAX. Tangent 2's contribution is orthogonal: a
*readable, editable* adjoint for gnarly loop- and branch-heavy code, one API
across six backends, with optional compilation when speed matters.

## 7. Reproducibility

This document is executable. Two ways to reproduce every result:

- **In your browser:** open `paper/Tangent2.ipynb` in Colab (badge in
  `paper/README.md`) — the first cell installs `tangent-ad`.
- **From a shell:** `pip install "tangent-ad[symbolic]>=0.4.0" && python paper/reproduce.py`
  runs and asserts every quantitative claim in this writeup in a few seconds.

## 8. Conclusion

Source-to-source AD makes derivatives legible, and Tangent 2 makes it practical
in modern Python: multi-backend, optionally compiled, memory-bounded for long
simulations, statically typed, and — above all — debuggable. For code where you
need to *understand* the gradient, not just compute it, that is the point.

## References

1. Merriënboer, Moldovan, Wiltschko. *Tangent: automatic differentiation using
   source-code transformation.* NeurIPS 2018.
2. Griewank, Walther. *Evaluating Derivatives.* SIAM, 2008.
3. Chen et al. *Neural Ordinary Differential Equations.* NeurIPS 2018.
4. Stumm, Walther. *Multistage approaches for optimal offline checkpointing.*
   SIAM J. Sci. Comput., 2009.
5. Maclaurin, Duvenaud, Adams. *Autograd.* 2015.
