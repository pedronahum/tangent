# Copyright 2026 Tangent contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproduce every quantitative claim in the Tangent 2 writeup.

Run it in ~a few seconds:

    pip install "tangent-ad[symbolic]"      # jax / torch are optional (auto-skipped)
    python paper/reproduce.py

Each check prints a line and asserts its result, so a non-zero exit means a
claim in the paper no longer holds. Optional-backend checks (JAX, PyTorch) skip
cleanly when the backend is not installed.
"""

import sys
import tracemalloc

import numpy as np

import tangent

CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append(ok)
    print('[%s] %s%s' % ('PASS' if ok else 'FAIL', name, ('  — ' + detail) if detail else ''))


# --- 1. One API, readable gradient, correct value ---------------------------
def cubic(x):
    return x**3 - 2 * x**2 + 3 * x - 1


def s1_readable_gradient():
    df = tangent.grad(cubic)
    src = df.__tangent_source__
    check('readable gradient is Python source', 'def d' in src and 'return' in src)
    check('gradient value f\'(2) == 7', abs(df(2.0) - 7.0) < 1e-9, 'got %.6f' % df(2.0))


# --- 2. Higher-order derivatives --------------------------------------------
def cube(x):
    return x**3


def s2_higher_order():
    ddf = tangent.grad(tangent.grad(cube))
    dddf = tangent.grad(ddf)
    check('2nd derivative d2/dx2 x^3 == 6x (=12 @2)', abs(ddf(2.0) - 12.0) < 1e-9)
    check('3rd derivative d3/dx3 x^3 == 6', abs(dddf(2.0) - 6.0) < 1e-9)


# --- 3. Multi-backend: the same transform across array libraries -------------
def np_loss(x):
    return np.sum(np.tanh(x) ** 2)


def s3_multibackend():
    x = np.array([0.5, -1.0, 2.0])
    g_np = tangent.grad(np_loss)(x)
    analytic = 2 * np.tanh(x) * (1 - np.tanh(x) ** 2)
    check('NumPy gradient matches analytic', np.allclose(g_np, analytic))

    try:
        import jax.numpy as jnp

        def jax_loss(x):
            return jnp.sum(jnp.tanh(x) ** 2)

        xj = jnp.asarray(x)
        g_plain = np.asarray(tangent.grad(jax_loss)(xj))
        g_jit = np.asarray(tangent.grad(jax_loss, compile='jax')(xj))
        check('JAX gradient matches analytic', np.allclose(g_plain, analytic, atol=1e-5))
        check("compile='jax' matches the plain adjoint", np.allclose(g_plain, g_jit, atol=1e-6))
    except ImportError:
        print('[SKIP] JAX not installed')

    try:
        import torch

        def torch_loss(x):
            return torch.sum(torch.tanh(x) ** 2)

        g_t = tangent.grad(torch_loss)(torch.as_tensor(x)).numpy()
        check('PyTorch gradient matches analytic', np.allclose(g_t, analytic, atol=1e-6))
    except ImportError:
        print('[SKIP] PyTorch not installed')


# --- 4. Tape-liveness: less memory, identical gradient -----------------------
def array_loop(x):
    s = x
    for i in range(50):
        a = s + 1.0
        s = a * 0.5 + s * 0.5
    return np.sum(s * s)


def s4_tape_liveness():
    x = np.arange(20000.0)
    off = tangent.grad(array_loop)
    on = tangent.grad(array_loop, optimizations={'tape_liveness': True})
    check('tape-liveness gradient is identical', np.allclose(off(x), on(x)))

    def peak(df):
        df(x)  # warm
        tracemalloc.start()
        df(x)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    p_off, p_on = peak(off), peak(on)
    check(
        'tape-liveness cuts peak memory > 2x',
        p_on < 0.5 * p_off,
        '%.1f MB -> %.1f MB (%.0f%% less)'
        % (p_off / 1e6, p_on / 1e6, 100 * (p_off - p_on) / p_off),
    )


# --- 5. Straight-line coarsening: same gradient, one symbolic VJP ------------
def kernel(a, b, c):
    return (np.exp(np.sin(a * b)) + c) * a


def s5_coarsening():
    std = tangent.grad(kernel, wrt=(0, 1, 2))(0.7, 1.1, 0.3)
    coa = tangent.grad(kernel, wrt=(0, 1, 2), optimizations={'coarsening': True})(0.7, 1.1, 0.3)
    check('coarsened gradient matches the per-op gradient', np.allclose(std, coa))


# --- 6. Differentiate a messy simulator: LMM caplet greeks vs FD -------------
def caplet_price(F0, sigma, Z, K, tau, dt, L, e0, M, N, n_steps):
    F = F0 * np.ones((M, N))
    sqrt_dt = dt**0.5
    for step in range(n_steps):
        g = tau * sigma * F / (1.0 + tau * F)
        drift = sigma * (g @ L) - 0.5 * sigma * sigma
        F = F * np.exp(drift * dt + sigma * (sqrt_dt * Z[step]))
    F_reset = F @ e0
    intrinsic = F_reset - K
    payoff = tau * (intrinsic * (intrinsic > 0.0)) / (1.0 + tau * F_reset)
    return np.sum(payoff) / M


def s6_lmm_greeks():
    N, M, n_steps = 5, 4000, 8
    tau, K = 0.5, 0.03
    dt = 1.0 / n_steps
    F0 = np.array([0.03, 0.032, 0.034, 0.035, 0.036])
    sigma = np.array([0.20, 0.22, 0.24, 0.23, 0.21])
    Z = np.random.RandomState(0).standard_normal((n_steps, M, 1))
    L = np.tril(np.ones((N, N)))
    e0 = np.eye(N)[0]
    C = (Z, K, tau, dt, L, e0, M, N, n_steps)

    delta, vega = tangent.grad(caplet_price, wrt=(0, 1))(F0, sigma, *C)
    h = 1e-6
    fd_delta = np.array(
        [
            (
                caplet_price(F0 + h * np.eye(N)[i], sigma, *C)
                - caplet_price(F0 - h * np.eye(N)[i], sigma, *C)
            )
            / (2 * h)
            for i in range(N)
        ]
    )
    fd_vega = np.array(
        [
            (
                caplet_price(F0, sigma + h * np.eye(N)[i], *C)
                - caplet_price(F0, sigma - h * np.eye(N)[i], *C)
            )
            / (2 * h)
            for i in range(N)
        ]
    )
    check('LMM vega matches finite differences', np.max(np.abs(vega - fd_vega)) < 1e-6)
    check(
        'LMM delta matches finite differences (ATM kink noise)',
        np.max(np.abs(delta - fd_delta)) < 1e-3,
        'max|AD-FD| = %.2e' % np.max(np.abs(delta - fd_delta)),
    )


def main():
    print('Reproducing the Tangent 2 writeup — tangent version %s\n' % tangent.__version__)
    s1_readable_gradient()
    s2_higher_order()
    s3_multibackend()
    s4_tape_liveness()
    s5_coarsening()
    s6_lmm_greeks()
    passed = sum(CHECKS)
    print('\n%d/%d checks passed.' % (passed, len(CHECKS)))
    return 0 if all(CHECKS) else 1


if __name__ == '__main__':
    sys.exit(main())
