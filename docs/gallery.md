# Notebook Gallery

Every notebook below **runs in your browser** — click *Open in Colab*, no
install required. They are ordinary, executed Jupyter notebooks in the repo, so
you can also run them locally (`jupyter lab notebooks/` or `examples/`).

New to Tangent? Start with the **Tutorial**, then see the **readable gradients**
that are the whole point, then a **case study** in your domain.

---

## Start here

- **Tangent Tutorial**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/tangent_tutorial.ipynb)
  <br>End-to-end tour: `grad`, the NumPy/JAX/PyTorch/TensorFlow backends, control
  flow, containers, and higher-order derivatives.

---

## See the readable gradients — the niche

Tangent differentiates by **source-to-source transformation**: the gradient is
Python you can read, debug, and edit. Tracing autodiff (JAX, PyTorch,
TensorFlow) gives an opaque graph or tape; these notebooks show what having the
actual adjoint source buys you.

- **Gallery of Gradients**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Gallery_of_Gradients.ipynb)
  <br>Eight functions, each shown next to its generated gradient code — from a
  polynomial to loops and control flow.

- **Gradient Surgery**
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/gradient_surgery.ipynb)
  <br>Read **and edit** the backward pass with `insert_grad_of` — scale, clip,
  log, and guard gradients mid-flow. The demo no tracing AD can copy.

---

## Case studies — differentiate a messy NumPy simulator

Gnarly, loop-and-branch scientific and quantitative code, differentiated **as
written** (no framework rewrite), with the gradient validated against finite
differences.

- **LIBOR Market Model greeks** (finance)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/LIBOR_Market_Model_Greeks_with_Tangent.ipynb)
  <br>A Monte-Carlo forward-rate simulator with state-dependent drift; exact
  deltas and vegas in **one reverse pass** instead of bump-and-revalue.

- **Projectile with drag** (physics / ODE)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Projectile_with_Drag_Sensitivities_with_Tangent.ipynb)
  <br>A nonlinear ODE integrator (quadratic drag, no closed form); sensitivities
  to the launch parameters, then **optimize the launch** by gradient descent.

- **SIR epidemic calibration** (epidemiology)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/SIR_Epidemic_Calibration_with_Tangent.ipynb)
  <br>Differentiate **through a data-dependent lockdown branch** to calibrate the
  transmission and recovery rates from a case curve.

- **Building energy optimization** (applied)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Building_Energy_Optimization_with_Tangent.ipynb)
  <br>A thermal building simulation differentiated for gradient-based control /
  design optimization.

---

## Run locally

```bash
pip install tangent-ad[all]
git clone https://github.com/pedronahum/tangent.git
jupyter lab tangent/notebooks tangent/examples
```

Prefer the terminal? `python -m tangent doctor` checks your install (and warns
if the dead 2017 `tangent` package is shadowing `tangent-ad`).
