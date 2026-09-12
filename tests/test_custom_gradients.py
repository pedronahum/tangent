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
"""The public custom-gradient API.

Two documented ways for a user to supply a gradient Tangent cannot derive:

  * ``custom_vjp`` / ``defvjp`` / ``defjvp`` - a plain-Python rule for a
    function you own or a black box you call.
  * ``register_adjoint`` / ``register_tangent`` - the low-level template DSL
    for attaching a rule to a library op you do not own.
"""

import numpy as np
import pytest

import tangent


# ---------------------------------------------------------------------------
# register_adjoint / register_tangent: a rule for a library op (numpy.hypot).
# The templates below are parsed (not executed) by Tangent, so their bodies use
# the d[...] DSL. They must live at module scope to have retrievable source.
# ---------------------------------------------------------------------------


@tangent.register_adjoint(np.hypot)
def _hypot_adjoint(z, x, y):
    d[x] = d[z] * x / z  # noqa: F821  (d[...] is Tangent DSL, resolved by the transform)
    d[y] = d[z] * y / z  # noqa: F821


@tangent.register_tangent(np.hypot)
def _hypot_tangent(z, x, y):
    d[z] = (x * d[x] + y * d[y]) / z  # noqa: F821


def hypot_loss(x, y):
    return np.hypot(x, y)


class TestRegisterAdjoint:
    def test_reverse_mode_matches_analytic(self):
        gx, gy = tangent.grad(hypot_loss, wrt=(0, 1))(3.0, 4.0)
        z = 5.0  # hypot(3, 4)
        assert gx == pytest.approx(3.0 / z)
        assert gy == pytest.approx(4.0 / z)

    def test_reverse_mode_matches_finite_difference(self):
        x0, y0, h = 1.7, 2.3, 1e-6
        gx = tangent.grad(hypot_loss, wrt=(0,))(x0, y0)
        fd = (np.hypot(x0 + h, y0) - np.hypot(x0 - h, y0)) / (2 * h)
        assert gx == pytest.approx(fd, rel=1e-4)

    def test_forward_mode_matches_reverse(self):
        x0, y0 = 3.0, 4.0
        # directional derivative along (1, 0) is d/dx
        dz = tangent.jvp(hypot_loss, wrt=(0,))(x0, y0, 1.0)
        assert dz == pytest.approx(3.0 / 5.0)

    def test_returns_template_unchanged(self):
        # The decorator returns its argument, so a name stays bound to it.
        assert _hypot_adjoint.__name__ == '_hypot_adjoint'


# ---------------------------------------------------------------------------
# custom_vjp: wrapping a black box (a callable Tangent must not transform).
# ---------------------------------------------------------------------------

_BLACKBOX_CALLS = []


def _blackbox_forward(v):
    """Stand-in for un-transformable code (a C ext / external solver). It
    records that it was called through, so the test can prove the primal ran
    the real op rather than a differentiated copy of it."""
    _BLACKBOX_CALLS.append(v)
    return float(np.sin(v))


@tangent.custom_vjp
def blackbox(x):
    return _blackbox_forward(x)


@blackbox.defvjp
def blackbox_vjp(g, ans, x):
    # We know d/dx sin(x) = cos(x) analytically even though the forward pass is
    # opaque to Tangent.
    return g * float(np.cos(x))


@blackbox.defjvp
def blackbox_jvp(ans, x, dx):
    return dx * float(np.cos(x))


def uses_blackbox(x):
    return blackbox(x) * 2.0


class TestCustomVjpBlackBox:
    def test_reverse_mode(self):
        _BLACKBOX_CALLS.clear()
        g = tangent.grad(uses_blackbox)(0.5)
        # d/dx (2 sin x) = 2 cos x
        assert g == pytest.approx(2.0 * np.cos(0.5))

    def test_primal_ran_the_real_black_box(self):
        # The custom rule must not cause Tangent to transform the opaque body;
        # the real forward function is what executes in the primal.
        _BLACKBOX_CALLS.clear()
        tangent.grad(uses_blackbox)(0.5)
        assert _BLACKBOX_CALLS, 'the black-box forward was never called'

    def test_forward_mode(self):
        dz = tangent.jvp(uses_blackbox)(0.5, 1.0)
        assert dz == pytest.approx(2.0 * np.cos(0.5))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
