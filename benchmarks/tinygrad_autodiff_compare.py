"""Benchmark: tinygrad's built-in autodiff vs Tangent-generated gradients.

Both arms compute the same thing end to end: given materialized inputs on the
device, produce the gradients of a scalar loss with respect to every input and
materialize them (forcing full realization of the lazy graph).

  - native : `f(*ts).gradient(*ts)`  — tinygrad's tape-based runtime autodiff.
  - tangent: `df(*ts)` where `df = tangent.grad(f, wrt=all)` — the source-to-
    source generated gradient function (compiled once; compile time is
    reported separately, like a JIT).

Methodology: warmup iterations (tinygrad compiles device kernels on first
use), then timed iterations; the median is reported. Every timed iteration
includes `.numpy()` on each gradient so both arms pay realization + device to
host copy. Gradients are cross-checked for agreement before timing.

Usage:
    .venv-tinygrad/bin/python benchmarks/tinygrad_autodiff_compare.py
    DEV=CPU .venv-tinygrad/bin/python benchmarks/tinygrad_autodiff_compare.py
"""
import statistics
import sys
import time

import numpy as np

from tinygrad import Device, Tensor
import tangent


# ----------------------------------------------------------------------------
# Workloads (module-level functions: Tangent reads their source)
# ----------------------------------------------------------------------------

def wl_elementwise(x):
    return ((x * x * 0.5 + 1.0).exp() + x.tanh() * 3.0).sum()


def wl_mlp(x, w1, b1, w2, b2):
    h = x.matmul(w1).add(b1).relu()
    return h.matmul(w2).add(b2).relu().sum()


def wl_matmul_big(x, w):
    return x.matmul(w).sum()


def wl_convnet(x, w1, b1, w2, b2):
    h = x.conv2d(w1, b1, padding=1).relu().max_pool2d()
    return h.conv2d(w2, b2, padding=1).relu().sum()


def wl_softmax_ce(logits, targets):
    return -(targets * logits.softmax().log()).sum()


def wl_layernorm_mlp(x, w, b):
    return x.layernorm().matmul(w).add(b).relu().sum()


def _rs(seed):
    return np.random.RandomState(seed)


WORKLOADS = [
    # (name, func, [(size_label, input builder, iterations)])
    ('elementwise', wl_elementwise, [
        ('1e4', lambda: [_rs(0).randn(10_000).astype(np.float32)], 100),
        ('1e6', lambda: [_rs(0).randn(1_000_000).astype(np.float32)], 50),
        ('1e7', lambda: [_rs(0).randn(10_000_000).astype(np.float32)], 20),
    ]),
    ('mlp(2 layers)', wl_mlp, [
        ('B32 H128', lambda: [
            _rs(1).randn(32, 128).astype(np.float32),
            _rs(2).randn(128, 128).astype(np.float32) * 0.1,
            _rs(3).randn(128).astype(np.float32),
            _rs(4).randn(128, 128).astype(np.float32) * 0.1,
            _rs(5).randn(128).astype(np.float32)], 100),
        ('B256 H512', lambda: [
            _rs(1).randn(256, 512).astype(np.float32),
            _rs(2).randn(512, 512).astype(np.float32) * 0.05,
            _rs(3).randn(512).astype(np.float32),
            _rs(4).randn(512, 512).astype(np.float32) * 0.05,
            _rs(5).randn(512).astype(np.float32)], 50),
        ('B1024 H1024', lambda: [
            _rs(1).randn(1024, 1024).astype(np.float32),
            _rs(2).randn(1024, 1024).astype(np.float32) * 0.03,
            _rs(3).randn(1024).astype(np.float32),
            _rs(4).randn(1024, 1024).astype(np.float32) * 0.03,
            _rs(5).randn(1024).astype(np.float32)], 20),
    ]),
    ('matmul', wl_matmul_big, [
        ('1024^3', lambda: [
            _rs(6).randn(1024, 1024).astype(np.float32),
            _rs(7).randn(1024, 1024).astype(np.float32)], 20),
        ('2048^3', lambda: [
            _rs(6).randn(2048, 2048).astype(np.float32),
            _rs(7).randn(2048, 2048).astype(np.float32)], 10),
    ]),
    ('convnet', wl_convnet, [
        ('N8 3>16>32 56px', lambda: [
            _rs(8).randn(8, 3, 56, 56).astype(np.float32),
            _rs(9).randn(16, 3, 3, 3).astype(np.float32) * 0.2,
            _rs(10).randn(16).astype(np.float32),
            _rs(11).randn(32, 16, 3, 3).astype(np.float32) * 0.1,
            _rs(12).randn(32).astype(np.float32)], 30),
        ('N32 3>32>64 112px', lambda: [
            _rs(8).randn(32, 3, 112, 112).astype(np.float32),
            _rs(9).randn(32, 3, 3, 3).astype(np.float32) * 0.2,
            _rs(10).randn(32).astype(np.float32),
            _rs(11).randn(64, 32, 3, 3).astype(np.float32) * 0.1,
            _rs(12).randn(64).astype(np.float32)], 10),
    ]),
    ('softmax x-ent', wl_softmax_ce, [
        ('B256 C100', lambda: [
            _rs(13).randn(256, 100).astype(np.float32),
            (_rs(14).rand(256, 100) > 0.97).astype(np.float32)], 100),
        ('B4096 C1000', lambda: [
            _rs(13).randn(4096, 1000).astype(np.float32),
            (_rs(14).rand(4096, 1000) > 0.997).astype(np.float32)], 20),
    ]),
    ('layernorm mlp', wl_layernorm_mlp, [
        ('B256 H512', lambda: [
            _rs(15).randn(256, 512).astype(np.float32),
            _rs(16).randn(512, 512).astype(np.float32) * 0.05,
            _rs(17).randn(512).astype(np.float32)], 50),
    ]),
]


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------

def _as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def native_arm(f, ts):
    grads = f(*ts).gradient(*ts)
    return [g.numpy() for g in grads]


def make_tangent_arm(f, ts):
    t0 = time.perf_counter()
    df = tangent.grad(f, wrt=tuple(range(len(ts))))
    compile_s = time.perf_counter() - t0

    def arm():
        return [g.numpy() for g in _as_tuple(df(*ts))]

    return arm, compile_s


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    print('device:', Device.DEFAULT)
    print('tinygrad native `Tensor.gradient` vs `tangent.grad` (steady state,')
    print('median of timed iterations incl. gradient materialization)')
    print()
    header = (f"{'workload':<16} {'size':<16} {'native ms':>10} "
              f"{'tangent ms':>11} {'speedup':>8} {'tg compile s':>13} {'match':>6}")
    print(header)
    print('-' * len(header))

    failures = []
    for name, func, sizes in WORKLOADS:
        for size_label, build, iters in sizes:
            ts = [Tensor(a) for a in build()]
            ref = native_arm(func, ts)
            arm, compile_s = make_tangent_arm(func, ts)
            got = arm()
            match = all(np.allclose(g, r, rtol=2e-3, atol=2e-3)
                        for g, r in zip(got, ref))
            if not match:
                failures.append((name, size_label))
            warmup = min(5, iters)
            t_native = bench(lambda: native_arm(func, ts), warmup, iters)
            t_tangent = bench(arm, warmup, iters)
            speedup = t_native / t_tangent
            print(f'{name:<16} {size_label:<16} {t_native * 1e3:>10.2f} '
                  f'{t_tangent * 1e3:>11.2f} {speedup:>7.2f}x '
                  f'{compile_s:>13.3f} {str(match):>6}')
            sys.stdout.flush()

    print()
    if failures:
        print('CORRECTNESS FAILURES:', failures)
        return 1
    print('speedup = native / tangent (>1 means tangent faster)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
