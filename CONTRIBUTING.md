# Contributing to Tangent

Thanks for your interest in contributing! This repository is a maintained fork
of [Google's Tangent](https://github.com/google/tangent); development happens
here via ordinary GitHub pull requests — there is no CLA.

## Getting set up

```bash
git clone https://github.com/pedronahum/tangent.git
cd tangent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # core deps + pytest + the autograd test oracle
pytest tests/                     # the core suite needs no optional backends
```

The suite must pass with **no optional backends installed** — backend-specific
tests skip cleanly when their library is absent. To exercise a backend, install
it (e.g. `pip install jax`) and rerun; `tests/test_backend_coverage.py` picks up
every installed backend automatically.

## Adding new derivatives

To add a derivative for a primitive operation:

- Read the docs on how to write derivatives at the top of
  [`tangent/grads.py`](https://github.com/pedronahum/tangent/blob/master/tangent/grads.py), and look at the existing examples
  there (e.g. `np.sin`).
- Add the reverse-mode derivative (adjoint) to
  [`tangent/grads.py`](https://github.com/pedronahum/tangent/blob/master/tangent/grads.py), and the forward-mode derivative to
  [`tangent/tangents.py`](https://github.com/pedronahum/tangent/blob/master/tangent/tangents.py).
- Backend-specific ops live in the extension modules:
  [`tangent/jax_extensions.py`](https://github.com/pedronahum/tangent/blob/master/tangent/jax_extensions.py),
  [`tangent/tf_extensions.py`](https://github.com/pedronahum/tangent/blob/master/tangent/tf_extensions.py),
  [`tangent/torch_extensions.py`](https://github.com/pedronahum/tangent/blob/master/tangent/torch_extensions.py),
  [`tangent/keras_extensions.py`](https://github.com/pedronahum/tangent/blob/master/tangent/keras_extensions.py),
  [`tangent/tinygrad_extensions.py`](https://github.com/pedronahum/tangent/blob/master/tangent/tinygrad_extensions.py).
- Register the op in the cross-backend catalog in
  [`tests/test_backend_coverage.py`](https://github.com/pedronahum/tangent/blob/master/tests/test_backend_coverage.py) so it is
  verified against every installed backend, and/or add a function using it to
  [`tests/functions.py`](https://github.com/pedronahum/tangent/blob/master/tests/functions.py), which the parameterized tests pick
  up automatically.
- Run `pytest tests/` locally before opening the PR. CI (GitHub Actions,
  [`.github/workflows/ci.yml`](https://github.com/pedronahum/tangent/blob/master/.github/workflows/ci.yml)) runs the suite on
  Python 3.9–3.13 with and without the optional backends.

## Other functionality

If you've fixed a bug or built an enhancement, open a PR. For larger feature
work, open a GitHub issue first to discuss the approach — the
[docs/](https://github.com/pedronahum/tangent/tree/master/docs) tree records design notes for several subsystems
(checkpointing, optimizations, backend coverage) that are worth reading before
touching those areas.
