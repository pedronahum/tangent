"""Peak-memory benchmark for segment (sqrt-n) checkpointing.

Runs a state-carrying loop (the workload checkpointing exists for) with and
without `checkpoint=True` and reports peak traced memory of the gradient call.

    .venv/bin/python benchmarks/checkpointing_memory.py
"""

import tracemalloc

import numpy as np

import tangent


def long_loop(x):
    s = np.zeros(1000)
    for i in range(2000):
        s = s * 0.999 + x * x
    return np.sum(s)


def peak_bytes(df, x):
    tracemalloc.start()
    df(x)  # absorb one-time allocations
    tracemalloc.reset_peak()
    df(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    x = np.full(1000, 0.5)
    df_plain = tangent.grad(long_loop)
    df_ckpt = tangent.grad(long_loop, checkpoint=True)

    g1, g2 = df_plain(x), df_ckpt(x)
    assert np.array_equal(g1, g2), 'checkpointed gradient diverged'

    p_plain = peak_bytes(df_plain, x)
    p_ckpt = peak_bytes(df_ckpt, x)

    print('workload: 2000 iterations x 1000-float loop state')
    print('full tape      peak: %6.1f MB' % (p_plain / 1e6))
    print('checkpointed   peak: %6.1f MB' % (p_ckpt / 1e6))
    print('reduction:           %6.1f%%' % (100.0 * (1.0 - p_ckpt / p_plain)))
    print('gradients identical: True')


if __name__ == '__main__':
    main()
