"""Op-tail derivative coverage: sort, cumprod, pad, take, einsum, linalg.

Every gradient is verified against the central finite-difference oracle;
forward-mode rules are checked where registered. Unsupported einsum forms
raise clean NotImplementedError instead of computing something wrong.
"""

import numpy as np
import pytest

import tangent

from utils import numeric_grad

X4 = np.array([0.3, 1.7, 0.9, 2.4])


def f_sort(x):
    return np.sum(np.sort(x) * np.array([1.0, 2.0, 3.0, 4.0]))


def f_sort_axis(x):
    m = np.reshape(x, (2, 2))
    return np.sum(np.sort(m, 0) * np.array([[1.0, 2.0], [3.0, 4.0]]))


def f_argsort_take(x):
    order = np.argsort(x)
    return np.sum(np.take(x, order) * np.array([1.0, 2.0, 3.0, 4.0]))


def f_cumprod(x):
    return np.sum(np.cumprod(x))


def f_pad(x):
    return np.sum(np.pad(x, 2) * np.arange(8.0))


def f_pad_tuple(x):
    return np.sum(np.pad(x, (1, 3)) * np.arange(8.0))


def f_take_flat(x):
    # Repeated index: gradients must accumulate, not overwrite.
    return np.sum(np.take(x, np.array([0, 2, 2])))


def f_take_axis(x):
    m = np.reshape(x, (2, 2))
    return np.sum(np.take(m, np.array([1, 0]), 1) * np.array([[1.0, 2.0], [3.0, 4.0]]))


def f_einsum_matmul(x):
    a = np.reshape(x, (2, 2))
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    return np.sum(np.einsum('ij,jk->ik', a, b))


def f_einsum_inner(x):
    b = np.array([1.0, 2.0, 3.0, 4.0])
    return np.einsum('i,i->', x, b) * 2.0


def f_einsum_summed_index(x):
    # j appears only in the first operand: its gradient broadcasts back.
    a = np.reshape(x, (2, 2))
    b = np.array([1.0, 2.0])
    return np.sum(np.einsum('ij,i->i', a, b))


def f_einsum_batched(x):
    a = np.reshape(x, (1, 2, 2))
    b = np.full((1, 2, 2), 0.5)
    return np.sum(np.einsum('bij,bjk->bik', a, b))


def f_cholesky(x):
    a = np.reshape(x, (2, 2))
    spd = a @ np.transpose(a) + 4.0 * np.eye(2)
    L = np.linalg.cholesky(spd)
    return np.sum(L * np.array([[1.0, 0.0], [2.0, 3.0]]))


def f_eigvalsh(x):
    a = np.reshape(x, (2, 2))
    sym = a + np.transpose(a)
    w = np.linalg.eigvalsh(sym)
    return np.sum(w * np.array([1.0, 3.0]))


REVERSE_CASES = [
    f_sort,
    f_sort_axis,
    f_argsort_take,
    f_cumprod,
    f_pad,
    f_pad_tuple,
    f_take_flat,
    f_take_axis,
    f_einsum_matmul,
    f_einsum_inner,
    f_einsum_summed_index,
    f_einsum_batched,
    f_cholesky,
    f_eigvalsh,
]


@pytest.mark.parametrize('fn', REVERSE_CASES, ids=lambda f: f.__name__)
def test_reverse_matches_fd(fn):
    g = tangent.grad(fn)(X4)
    np.testing.assert_allclose(g, numeric_grad(fn)(X4), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    'fn', [f_sort, f_cumprod, f_pad, f_take_flat, f_einsum_matmul], ids=lambda f: f.__name__
)
def test_forward_matches_fd(fn):
    got = tangent.autodiff(fn, mode='forward')(X4, np.ones_like(X4))
    assert got == pytest.approx(float(np.sum(numeric_grad(fn)(X4))), rel=1e-4)


class TestEinsumRejections:
    def test_implicit_output_rejected(self):
        def f(x):
            b = np.array([1.0, 2.0, 3.0, 4.0])
            return np.einsum('i,i', x, b)

        with pytest.raises(NotImplementedError, match='explicit output'):
            tangent.grad(f)(X4)

    def test_ellipsis_rejected(self):
        def f(x):
            a = np.reshape(x, (2, 2))
            return np.sum(np.einsum('...j,jk->...k', a, np.eye(2)))

        with pytest.raises(NotImplementedError, match='ellipsis'):
            tangent.grad(f)(X4)

    def test_repeated_index_rejected(self):
        def f(x):
            a = np.reshape(x, (2, 2))
            return np.einsum('ii,i->', a, np.array([1.0, 2.0]))

        with pytest.raises(NotImplementedError, match='repeated index'):
            tangent.grad(f)(X4)


def test_second_order_sort():
    def f(x):
        s = np.sort(x)
        return np.sum(s * s)

    np.testing.assert_allclose(tangent.grad(tangent.grad(f))(X4), 2.0 * np.ones_like(X4), rtol=1e-6)
