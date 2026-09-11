"""Features from the 2026 roadmap review: disk cache, compile=, explain,
custom_vjp/stop_gradient, dynamic concat/stack, checkpoint annotation, doctor.
"""

import importlib.util
import os
import sys
import tracemalloc

import numpy as np
import pytest

import tangent
from tangent import disk_cache

from utils import numeric_grad


# --- persistent disk cache -------------------------------------------------


def cached_fn(x):
    s = 0.0
    for i in range(5):
        s = s + x * x * 0.1
    return s


class TestDiskCache:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv('TANGENT_CACHE_DIR', str(tmp_path))
        monkeypatch.setenv('TANGENT_DISK_CACHE', '1')
        disk_cache._package_fingerprint_memo = None
        tangent.clear_cache()
        df = tangent.grad(cached_fn)
        assert df(2.0) == pytest.approx(2.0)
        entries = [f for f in os.listdir(tmp_path) if f.endswith('.json')]
        assert len(entries) == 1

        # Clear the in-memory cache: the next grad() must reconstruct from
        # disk (observable through cache stats: a miss, then no new entry).
        tangent.clear_cache()
        df2 = tangent.grad(cached_fn)
        assert df2(2.0) == pytest.approx(2.0)
        assert getattr(df2, '__tangent_source__', None)
        assert [f for f in os.listdir(tmp_path) if f.endswith('.json')] == entries

    def test_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv('TANGENT_CACHE_DIR', str(tmp_path))
        monkeypatch.setenv('TANGENT_DISK_CACHE', '0')
        tangent.clear_cache()
        tangent.grad(cached_fn)(2.0)
        assert not [f for f in os.listdir(tmp_path) if f.endswith('.json')]


# --- compile= lowering ------------------------------------------------------


def test_source_attached():
    df = tangent.grad(cached_fn)
    assert 'def d' in df.__tangent_source__
    assert df.__tangent_entries__


def test_compile_python_is_default():
    assert tangent.grad(cached_fn, compile='python')(2.0) == tangent.grad(cached_fn)(2.0)


def test_compile_numba_rejected_with_reason():
    with pytest.raises(ValueError, match='numba'):
        tangent.grad(cached_fn, compile='numba')


def test_compile_unknown_rejected():
    with pytest.raises(ValueError, match='Unknown compile backend'):
        tangent.grad(cached_fn, compile='cuda')


jax = pytest.importorskip('jax', reason='jax not installed')
import jax.numpy as jnp  # noqa: E402


def jax_loss(x):
    return jnp.sum(jnp.tanh(x) ** 2)


def test_compile_jax_matches_python():
    x = jnp.array([0.5, -1.0, 2.0])
    plain = tangent.grad(jax_loss)(x)
    jitted = tangent.grad(jax_loss, compile='jax')(x)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(jitted), rtol=1e-6)


def test_compile_jax_matmul_under_trace():
    w = jnp.eye(3) * 0.5
    b = jnp.ones((3, 3))

    def f(w):
        return jnp.sum((w @ b) ** 2)

    plain = tangent.grad(f)(w)
    jitted = tangent.grad(f, compile='jax')(w)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(jitted), rtol=1e-6)


# --- explain / source map ---------------------------------------------------


def explained(x, unused):
    y = x * x
    return y * 3.0


class TestExplain:
    def test_explain_report(self):
        r = tangent.explain(explained, 2.0, 5.0, wrt=(0, 1), out=lambda s: None)
        assert r['max_error'] < 1e-6
        assert r['gradient'] == (pytest.approx(12.0), pytest.approx(0.0))
        assert 'def d' in r['gradient_source']

    def test_source_map_resolves_primal_lines(self):
        df = tangent.grad(explained)
        entries = tangent.source_map(df, explained)
        mapped = [e for e in entries if e['primal'] == 'y = x * x']
        assert mapped
        assert mapped[0]['primal_line'] is not None


# --- custom_vjp / defjvp / stop_gradient ------------------------------------


@tangent.custom_vjp
def scaled_tanh(x, a):
    return a * np.tanh(x)


@scaled_tanh.defvjp
def scaled_tanh_vjp(g, ans, x, a):
    return g * a * (1.0 - np.tanh(x) ** 2), g * np.tanh(x)


@scaled_tanh.defjvp
def scaled_tanh_jvp(ans, x, a, dx, da):
    return a * (1.0 - np.tanh(x) ** 2) * dx + np.tanh(x) * da


def uses_custom(x):
    return scaled_tanh(x * 2.0, 3.0) + x


def uses_stop(x):
    y = x * x
    return tangent.stop_gradient(y) + 5.0 * x


class TestCustomRules:
    def test_custom_vjp_reverse(self):
        assert tangent.grad(uses_custom)(0.4) == pytest.approx(numeric_grad(uses_custom)(0.4))

    def test_custom_jvp_forward(self):
        got = tangent.autodiff(uses_custom, mode='forward')(0.4, 1.0)
        assert got == pytest.approx(numeric_grad(uses_custom)(0.4))

    def test_stop_gradient_both_modes(self):
        assert tangent.grad(uses_stop)(2.0) == pytest.approx(5.0)
        assert tangent.autodiff(uses_stop, mode='forward')(2.0, 1.0) == pytest.approx(5.0)

    def test_varargs_custom_vjp_rejected(self):
        with pytest.raises(ValueError, match='fixed positional signature'):

            @tangent.custom_vjp
            def bad(*xs):
                return xs[0]


# --- dynamic concatenate / stack --------------------------------------------


def dyn_concat(x):
    xs = []
    for i in range(3):
        xs.append(x * float(i + 1))
    return np.sum(np.concatenate(xs))


def dyn_stack(x):
    xs = []
    for i in range(3):
        xs.append(x * float(i + 1))
    return np.sum(np.stack(xs) * 2.0)


class TestDynamicConcat:
    @pytest.mark.parametrize('fn', [dyn_concat, dyn_stack], ids=lambda f: f.__name__)
    def test_reverse_matches_fd(self, fn):
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(tangent.grad(fn)(x), numeric_grad(fn)(x), rtol=1e-5)


# --- `with tangent.checkpoint():` annotation --------------------------------


def annotated_loop(x, n):
    s = np.zeros(4)
    with tangent.checkpoint():
        for i in range(n):  # runtime bound: only allowed via the annotation
            s = s * 0.9 + x * x
    return np.sum(s)


def plain_loop(x, n):
    s = np.zeros(4)
    for i in range(n):
        s = s * 0.9 + x * x
    return np.sum(s)


class TestCheckpointAnnotation:
    def test_gradients_identical(self):
        x = np.full(4, 0.5)
        for n in (1, 3, 17, 200):
            np.testing.assert_array_equal(
                tangent.grad(annotated_loop)(x, n), tangent.grad(plain_loop)(x, n)
            )

    def test_memory_drops(self):
        x = np.full(400, 0.5)

        def big_annotated(x, n):
            s = np.zeros(400)
            with tangent.checkpoint():
                for i in range(n):
                    s = s * 0.999 + x * x
            return np.sum(s)

        def big_plain(x, n):
            s = np.zeros(400)
            for i in range(n):
                s = s * 0.999 + x * x
            return np.sum(s)

        def peak(df):
            tracemalloc.start()
            df(x, 900)
            tracemalloc.reset_peak()
            df(x, 900)
            _, p = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return p

        da, dp = tangent.grad(big_annotated), tangent.grad(big_plain)
        np.testing.assert_array_equal(da(x, 900), dp(x, 900))
        assert peak(da) * 3 < peak(dp)

    def test_runtime_noop(self):
        assert annotated_loop(np.full(4, 0.5), 3) == plain_loop(np.full(4, 0.5), 3)


# --- python -m tangent doctor ------------------------------------------------


def test_doctor_runs(capsys):
    from tangent.__main__ import main

    code = main(['doctor'])
    outp = capsys.readouterr().out
    assert 'smoke gradient' in outp
    assert code in (0, 1)  # 1 only when it found a real problem to report
