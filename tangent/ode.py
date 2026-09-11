# Copyright 2018 Google Inc.
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
"""Differentiable ODE integration via the continuous adjoint method.

``tangent.odeint(func, y0, ts, args)`` integrates ``dy/dt = func(y, t, *args)``
and returns the state at each time in ``ts``. Its gradient w.r.t. ``y0`` and
``args`` is computed by the continuous adjoint method (Pontryagin; Chen et al.,
"Neural Ordinary Differential Equations", 2018): the backward pass solves an
augmented ODE rather than taping every step, so memory is constant in the
number of integration steps.

The adjoint dynamics need vector-Jacobian products of ``func`` - and Tangent
computes those *of the user's own function* with ``tangent.vjp``. Define the
dynamics in NumPy; the VJP that drives the adjoint solve is generated for you.

Scope: fixed-step RK4, gradients w.r.t. ``y0`` and ``args`` (not the time
points ``ts``). ``func`` must return an array the same shape as ``y``.
"""

from __future__ import absolute_import

import numpy

# Cache of generated VJP functions, keyed by (func, arg count): building one
# runs the whole AD pipeline, and odeint calls it once per step.
_vjp_cache = {}


def _get_vjp(func, n_args):
    key = (func, n_args)
    fn = _vjp_cache.get(key)
    if fn is None:
        import tangent

        wrt = (0,) + tuple(2 + i for i in range(n_args))  # y, then each arg (skip t)
        fn = tangent.vjp(func, wrt=wrt)
        _vjp_cache[key] = fn
    return fn


def _rk4_step(func, y, t, dt, args):
    """One classical RK4 step of dy/dt = func(y, t, *args)."""
    k1 = func(y, t, *args)
    k2 = func(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = func(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = func(y + dt * k3, t + dt, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def odeint(func, y0, ts, args=(), num_steps=10):
    """Integrate dy/dt = func(y, t, *args), returning y at each time in `ts`.

    Args:
      func: Dynamics `func(y, t, *args) -> dy/dt`, returning an array shaped
          like `y`.
      y0: Initial state (array).
      ts: 1-D increasing array of output times; `ts[0]` is the initial time.
      args: Tuple of extra arguments (the parameters `theta`) passed to `func`.
      num_steps: Fixed RK4 sub-steps per output interval.

    Returns:
      Array of shape `(len(ts),) + y0.shape`: the state at each time in `ts`.
    """
    ts = numpy.asarray(ts, dtype=float)
    y0 = numpy.asarray(y0, dtype=float)
    out = numpy.empty((len(ts),) + y0.shape, dtype=float)
    out[0] = y0
    y = y0
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        dt = (t1 - t0) / num_steps
        t = t0
        for _ in range(num_steps):
            y = _rk4_step(func, y, t, dt, args)
            t = t + dt
        out[i + 1] = y
    return out


def _as_arg_tuple(vjp_result, n_args):
    """Normalize tangent.vjp's return into (grad_y, [grad_arg, ...])."""
    if n_args == 0:
        return numpy.asarray(vjp_result), []
    grad_y = numpy.asarray(vjp_result[0])
    grad_args = [numpy.asarray(g) for g in vjp_result[1:]]
    return grad_y, grad_args


def _aug_rk4_step(func, vjp, y, a, g, t, dt, args, n_args):
    """One RK4 step of the augmented backward system (y, a, g).

    Forward-time dynamics (integrated with negative dt in the backward sweep):
        dy/dt = f(y, t, θ)
        da/dt = -aᵀ ∂f/∂y
        dg/dt = -aᵀ ∂f/∂θ
    """

    def deriv(y, a):
        dy = func(y, t, *args)
        grad_y, grad_args = _as_arg_tuple(vjp(y, t, *(args + (a,))), n_args)
        return dy, -grad_y, [-ga for ga in grad_args]

    dy1, da1, dg1 = deriv(y, a)
    dy2, da2, dg2 = deriv(y + 0.5 * dt * dy1, a + 0.5 * dt * da1)
    dy3, da3, dg3 = deriv(y + 0.5 * dt * dy2, a + 0.5 * dt * da2)
    dy4, da4, dg4 = deriv(y + dt * dy3, a + dt * da3)
    ny = y + (dt / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    na = a + (dt / 6.0) * (da1 + 2 * da2 + 2 * da3 + da4)
    ng = [
        gi + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        for gi, d1, d2, d3, d4 in zip(g, dg1, dg2, dg3, dg4)
    ]
    return ny, na, ng


def odeint_grad(dys, func, ys, y0, ts, args, num_steps=10):
    """Adjoint-method gradient of `odeint` w.r.t. `y0` and `args`.

    Args:
      dys: Cotangent of the output `ys` (same shape): dL/d ys[i].
      func, ys, y0, ts, args, num_steps: as produced by the forward `odeint`.

    Returns:
      `(grad_y0, grad_args)` where grad_args is a tuple aligned with `args`.
    """
    ts = numpy.asarray(ts, dtype=float)
    dys = numpy.asarray(dys, dtype=float)
    n_args = len(args)
    vjp = _get_vjp(func, n_args)

    # Adjoint at the final time, plus the direct cotangent there.
    a = numpy.array(dys[-1], dtype=float)
    g = [numpy.zeros_like(numpy.asarray(arg, dtype=float)) for arg in args]

    for i in range(len(ts) - 1, 0, -1):
        # Integrate the augmented system backward across [ts[i], ts[i-1]],
        # recomputing y from the stored ys[i] (constant memory in step count).
        y = numpy.array(ys[i], dtype=float)
        t1, t0 = ts[i], ts[i - 1]
        dt = -(t1 - t0) / num_steps
        t = t1
        for _ in range(num_steps):
            y, a, g = _aug_rk4_step(func, vjp, y, a, g, t, dt, args, n_args)
            t = t + dt
        # Add the direct dependence of L on the output at ts[i-1].
        a = a + dys[i - 1]

    grad_y0 = a
    grad_args = tuple(g)
    return grad_y0, grad_args
