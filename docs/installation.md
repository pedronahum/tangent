# Installation

Tangent is distributed on PyPI as **`tangent-ad`**. The *import* name is
unchanged — your code says `import tangent`, exactly like the original Google
release (the same arrangement as Pillow, which installs as `pillow` and
imports as `PIL`).

```bash
pip install tangent-ad
```

!!! warning "Upgrading from Google's 2017 `tangent` package"

    PyPI's `tangent` project is Google's unmaintained 0.1.9 release from
    2017. Both packages install a `tangent/` module, so if you have the old
    one, remove it first:

    ```bash
    pip uninstall tangent
    pip install tangent-ad
    ```

## Optional backends

The core install differentiates NumPy code. Each additional backend is an
extra:

```bash
pip install "tangent-ad[jax]"       # JAX
pip install "tangent-ad[tf]"        # TensorFlow 2.x
pip install "tangent-ad[torch]"     # PyTorch
pip install "tangent-ad[keras]"     # Keras 3 (backend-agnostic keras.ops)
pip install "tangent-ad[tinygrad]"  # tinygrad
pip install "tangent-ad[viz]"       # gradient-flow visualization (matplotlib + networkx)
pip install "tangent-ad[symbolic]"  # SymPy-based algebraic simplification
pip install "tangent-ad[all]"       # everything above, plus the test tooling
```

Backends are optional by design: extensions load when the backend is
importable and stay silent otherwise. Core NumPy autodiff always works.

## From source

```bash
pip install "tangent-ad @ git+https://github.com/pedronahum/tangent.git"
```

or for development:

```bash
git clone https://github.com/pedronahum/tangent.git
cd tangent
pip install -e ".[test]"
python -m pytest tests/ -q --short
```

## Requirements

- Python 3.9 – 3.13 (tested in CI on all five)
- `numpy` and `gast` (installed automatically)

Platform note: TensorFlow has no CUDA build for aarch64; on such machines the
TF backend runs on CPU.
