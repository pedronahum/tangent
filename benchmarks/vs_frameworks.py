"""Gradient benchmarks: Tangent vs jax.grad vs torch.autograd.

Two workloads chosen to show both sides honestly:

- **MLP** (tensor-heavy): a 3-layer MLP loss. Tracing frameworks fuse this
  well; Tangent's generated Python pays interpreter overhead per op - unless
  the adjoint is lowered with `compile='jax'`, which is the point of that
  feature.
- **Scalar recurrence** (loop-heavy, control-flow-heavy): 1000 sequential
  data-dependent steps. Tracing ADs must unroll (compile cost) or fall back
  to eager per-op dispatch (eval cost); source transformation shines here.

Compile/first-call time is reported separately from steady-state eval time -
source transformation always pays a compile bill, and the disk cache
amortizes it across processes.

    .venv/bin/python benchmarks/vs_frameworks.py
"""

import time

import numpy as np


def time_eval(fn, *args, repeats=50):
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def bench(name, make_grad, call, *args):
    t0 = time.perf_counter()
    df = make_grad()
    call(df, *args)  # first call: includes trace/jit for lazy compilers
    compile_time = time.perf_counter() - t0
    eval_time = time_eval(lambda *a: call(df, *a), *args)
    print('  %-28s compile+first %8.1f ms   eval %10.3f ms' % (name, compile_time * 1e3, eval_time * 1e3))
    return eval_time


# ---------------------------------------------------------------------------
# Workload 1: 3-layer MLP loss (tensor-heavy)
# ---------------------------------------------------------------------------

DIM = 128
BATCH = 64
rng = np.random.RandomState(0)
W1 = rng.randn(DIM, DIM) * 0.1
W2 = rng.randn(DIM, DIM) * 0.1
W3 = rng.randn(DIM, 1) * 0.1
X = rng.randn(BATCH, DIM)


def mlp_np(w1):
    h1 = np.tanh(X @ w1)
    h2 = np.tanh(h1 @ W2)
    out = h2 @ W3
    return np.sum(out * out)


def run_mlp():
    import tangent

    print('MLP loss (%dx%d, batch %d), d/dW1:' % (DIM, DIM, BATCH))
    t_eval = bench('tangent (numpy)', lambda: tangent.grad(mlp_np), lambda df, w: df(w), W1)

    try:
        import jax
        import jax.numpy as jnp

        Xj = jnp.asarray(X)
        W2j, W3j = jnp.asarray(W2), jnp.asarray(W3)

        def mlp_jax(w1):
            h1 = jnp.tanh(Xj @ w1)
            h2 = jnp.tanh(h1 @ W2j)
            out = h2 @ W3j
            return jnp.sum(out * out)

        W1j = jnp.asarray(W1)
        block = lambda r: jax.block_until_ready(r)
        bench('jax.grad (no jit)', lambda: jax.grad(mlp_jax), lambda df, w: block(df(w)), W1j)
        bench('jax.grad + jit', lambda: jax.jit(jax.grad(mlp_jax)), lambda df, w: block(df(w)), W1j)
        j_eval = bench(
            "tangent compile='jax'",
            lambda: tangent.grad(mlp_jax, compile='jax'),
            lambda df, w: block(df(w)),
            W1j,
        )
        print('  -> tangent-jitted vs tangent-python eval: %.1fx' % (t_eval / j_eval))
    except ImportError:
        print('  (jax not installed)')

    try:
        import torch

        Xt = torch.tensor(X)
        W2t, W3t = torch.tensor(W2), torch.tensor(W3)

        def torch_grad(w):
            w = w.detach().requires_grad_(True)
            h1 = torch.tanh(Xt @ w)
            h2 = torch.tanh(h1 @ W2t)
            out = h2 @ W3t
            loss = torch.sum(out * out)
            loss.backward()
            return w.grad

        W1t = torch.tensor(W1)
        bench('torch.autograd', lambda: torch_grad, lambda df, w: df(w), W1t)
    except ImportError:
        print('  (torch not installed)')


# ---------------------------------------------------------------------------
# Workload 2: scalar recurrence (loop-heavy)
# ---------------------------------------------------------------------------

N_STEPS = 1000


def recurrence_np(x):
    s = 0.0
    v = x
    for i in range(N_STEPS):
        v = v * 0.999 + 0.1 * s
        if v > 2.0:
            v = v * 0.5
        s = s + v * v * 1e-4
    return s


def run_recurrence():
    import tangent

    print('Scalar recurrence (%d data-dependent steps), d/dx:' % N_STEPS)
    bench('tangent (python)', lambda: tangent.grad(recurrence_np), lambda df, x: df(x), 1.5)

    try:
        import torch

        def rec_torch(x):
            s = torch.zeros(())
            v = x
            for i in range(N_STEPS):
                v = v * 0.999 + 0.1 * s
                if v > 2.0:
                    v = v * 0.5
                s = s + v * v * 1e-4
            return s

        def torch_grad(x):
            x = x.detach().requires_grad_(True)
            rec_torch(x).backward()
            return x.grad

        bench('torch.autograd (eager)', lambda: torch_grad, lambda df, x: df(x), torch.tensor(1.5))
    except ImportError:
        print('  (torch not installed)')

    try:
        import jax
        import jax.numpy as jnp

        def rec_jax(x):
            s = jnp.zeros(())
            v = x
            for i in range(N_STEPS):
                v = v * 0.999 + 0.1 * s
                v = jnp.where(v > 2.0, v * 0.5, v)
                s = s + v * v * 1e-4
            return s

        block = lambda r: jax.block_until_ready(r)
        bench(
            'jax.grad + jit (unrolled)',
            lambda: jax.jit(jax.grad(rec_jax)),
            lambda df, x: block(df(x)),
            jnp.asarray(1.5),
        )
    except ImportError:
        print('  (jax not installed)')


if __name__ == '__main__':
    run_mlp()
    print()
    run_recurrence()
