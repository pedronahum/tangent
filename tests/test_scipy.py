"""Gradients for scipy.special and scipy.linalg (tangent/scipy_extensions.py).

Every gradient is checked against the central finite-difference oracle. Skipped
cleanly when SciPy is not installed.
"""

import numpy as np
import pytest

import tangent

from utils import numeric_grad

scipy = pytest.importorskip('scipy', reason='scipy not installed')
from scipy import linalg  # noqa: E402
from scipy import special as sp  # noqa: E402


def f_erf(x):
    return np.sum(sp.erf(x))


def f_erfc(x):
    return np.sum(sp.erfc(x))


def f_gammaln(x):
    return np.sum(sp.gammaln(x))


def f_gamma(x):
    return np.sum(sp.gamma(x))


def f_psi(x):
    return np.sum(sp.psi(x))


def f_expit(x):
    return np.sum(sp.expit(x))


def f_logit(x):
    return np.sum(sp.logit(x))


def f_xlogy(x):
    return np.sum(sp.xlogy(x, x + 2.0))


def f_logsumexp(x):
    return sp.logsumexp(x)


SPECIAL = [
    (f_erf, np.array([0.4, 1.2, -0.8])),
    (f_erfc, np.array([0.4, 1.2, -0.8])),
    (f_gammaln, np.array([1.4, 2.2, 0.8])),
    (f_gamma, np.array([1.4, 2.2, 0.8])),
    (f_psi, np.array([1.4, 2.2, 0.8])),
    (f_expit, np.array([0.4, 1.2, -0.8])),
    (f_logit, np.array([0.3, 0.6, 0.9])),
    (f_xlogy, np.array([0.4, 1.2, 0.8])),
    (f_logsumexp, np.array([0.4, 1.2, -0.8])),
]


class TestSpecial:
    @pytest.mark.parametrize('fn,xin', SPECIAL, ids=[f.__name__ for f, _ in SPECIAL])
    def test_reverse_matches_fd(self, fn, xin):
        np.testing.assert_allclose(
            tangent.grad(fn)(xin), numeric_grad(fn)(xin), rtol=1e-4, atol=1e-6
        )

    @pytest.mark.parametrize(
        'fn,xin',
        [
            (f_erf, np.array([0.4, 1.2])),
            (f_expit, np.array([0.4, 1.2])),
            (f_xlogy, np.array([0.5, 1.5])),
        ],
        ids=['erf', 'expit', 'xlogy'],
    )
    def test_forward_matches_fd(self, fn, xin):
        got = tangent.autodiff(fn, mode='forward')(xin, np.ones_like(xin))
        assert got == pytest.approx(float(np.sum(numeric_grad(fn)(xin))), rel=1e-4)

    def test_second_order_erf(self):
        # d^2/dx^2 erf(x) = -4x/sqrt(pi) exp(-x^2)
        x = np.array([0.5, -1.0])
        dd = tangent.grad(tangent.grad(f_erf))(x)
        expected = -4.0 * x / np.sqrt(np.pi) * np.exp(-(x**2))
        np.testing.assert_allclose(dd, expected, rtol=1e-4)


class TestLinalg:
    def test_solve_vector(self):
        def f(b):
            A = np.array([[3.0, 1.0], [1.0, 2.0]])
            return np.sum(linalg.solve(A, b))

        b = np.array([1.0, 2.0])
        np.testing.assert_allclose(tangent.grad(f)(b), numeric_grad(f)(b), rtol=1e-4)

    def test_solve_matrix(self):
        def f(bmat):
            A = np.array([[3.0, 1.0], [1.0, 2.0]])
            return np.sum(linalg.solve(A, bmat) * np.array([[1.0, 2.0], [3.0, 4.0]]))

        bm = np.array([[1.0, 0.5], [2.0, 1.5]])
        np.testing.assert_allclose(tangent.grad(f)(bm), numeric_grad(f)(bm), rtol=1e-4)

    def test_inv(self):
        def f(x):
            A = np.reshape(x, (2, 2))
            return np.sum(linalg.inv(A))

        x = np.array([2.0, 1.0, 1.0, 3.0])
        np.testing.assert_allclose(tangent.grad(f)(x), numeric_grad(f)(x), rtol=1e-4)


def test_backend_status_reports_scipy():
    assert tangent.backend_status().get('scipy') == 'available'
