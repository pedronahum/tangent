# Differentiable ODE Integration

`tangent.odeint` integrates an ordinary differential equation and lets you
differentiate the solution with respect to the initial state and the dynamics
parameters — the "differentiate my simulator" case.

```python
import numpy as np
import tangent

def dynamics(y, t, k):        # dy/dt = -k y
    return -k * y

def loss(y0, k):
    ts = np.array([0.0, 0.5, 1.0])
    ys = tangent.odeint(dynamics, y0, ts, (k,))
    return np.sum(ys[-1])

tangent.grad(loss, wrt=(0, 1))(np.array([2.0, 1.0]), 0.5)
# (d/dy0, d/dk)
```

## The adjoint method

The gradient is **not** computed by taping every integration step. Instead the
backward pass solves the *continuous adjoint ODE* (Pontryagin; Chen et al.,
[Neural ODEs](https://arxiv.org/abs/1806.07366), 2018):

$$\frac{da}{dt} = -a^\top \frac{\partial f}{\partial y}, \qquad
  \frac{dL}{d\theta} = -\int a^\top \frac{\partial f}{\partial \theta}\, dt$$

so **memory is constant in the number of integration steps** — a 10-step and a
10,000-step solve cost the same to differentiate.

The elegant part: the adjoint dynamics need vector-Jacobian products of your
`func` (the $a^\top \partial f/\partial y$ and $a^\top \partial f/\partial
\theta$ terms), and **Tangent generates those from your own function** with
`tangent.vjp`. You write the dynamics in NumPy; the VJP that drives the adjoint
solve is produced for you. No hand-derived Jacobians.

## Signature

```python
tangent.odeint(func, y0, ts, args=(), num_steps=10)
```

| | |
|---|---|
| `func` | dynamics `func(y, t, *args) -> dy/dt`, returning an array shaped like `y` |
| `y0` | initial state |
| `ts` | 1-D increasing array of output times (`ts[0]` is the initial time) |
| `args` | tuple of parameters passed to `func` |
| `num_steps` | fixed RK4 sub-steps per output interval |

Returns the state at each time in `ts`, shape `(len(ts),) + y0.shape`.

## Fitting a simulator

The [`examples/neural_ode.py`](https://github.com/pedronahum/tangent/blob/master/examples/neural_ode.py)
example fits a decay rate by gradient descent straight through the solver:

```
fitting dy/dt = -k y  (true k = 0.7)
  step  0   k = 1.5776   loss = 3.367e-01
  step  5   k = 0.7032   loss = 9.857e-06
  step 24   k = 0.7000   loss = 0.000e+00
```

## Scope

- Gradients flow to `y0` and `args`; the time points `ts` are treated as
  constants. Fixed-step RK4 (raise `num_steps` for accuracy).
- Reverse mode, first order. Forward mode and higher-order derivatives through
  `odeint` raise a clean `NotImplementedError` rather than returning a wrong
  value — consistent with Tangent's reject-or-be-correct policy.
