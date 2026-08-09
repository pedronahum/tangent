# Tangent Examples

Runnable scripts and notebooks demonstrating Tangent's capabilities. Every
script is self-contained and prints `PASS`/`FAIL` for each check. Optional
backends (JAX, TensorFlow, PyTorch, Keras) are used when installed and
skipped gracefully otherwise; NumPy always runs.

## Start here

- **`recent_features.py`** - Showcase of recent capabilities in one script:
  straight-line **coarsening** (one symbolic VJP instead of per-op adjoints),
  **pytree (container) arguments**, **multi-backend** gradients (NumPy /
  PyTorch / Keras), differentiable **concat/stack**, and **second
  derivatives** through loops.
  ```bash
  python examples/recent_features.py
  ```

## Core examples

| File | Demonstrates |
|---|---|
| `test_basic.py` | Basics on NumPy: polynomials, second derivatives, readable code |
| `test_jax_basic.py` | JAX integration (`jax.numpy` ops, `jax.nn` activations) |
| `test_tf2_basic.py` | TensorFlow 2.x integration (eager mode) |
| `class_examples.py` | User-defined classes, method inlining, attributes |
| `lambda_examples.py` | Lambdas, closures, and higher-order functions |
| `checkpoint_demo.py` | Gradient checkpointing (memory-efficient reverse mode) |
| `demo_error_messages.py` | Enhanced, actionable error messages |
| `demo_visualization.py` | Generates the computation-graph / gradient-flow plots |

## Notebooks

| Notebook | Contents |
|---|---|
| [`Gallery_of_Gradients.ipynb`](Gallery_of_Gradients.ipynb) | Readable-gradient showcase: loops run in reverse, tape recording, conditionals, broadcasting. See [`README_GALLERY.md`](README_GALLERY.md) |
| [`Building_Energy_Optimization_with_Tangent.ipynb`](Building_Energy_Optimization_with_Tangent.ipynb) | Real-world differentiable simulation. See [`README_BUILDING_EXAMPLE.md`](README_BUILDING_EXAMPLE.md) |

A general introduction lives in [`../notebooks/tangent_tutorial.ipynb`](../notebooks/tangent_tutorial.ipynb).

## Extended NumPy op demos

`numpy_extended/` demonstrates the extended NumPy adjoints (matmul, reductions,
statistics, shape ops) with its own `README.md`, `demo.py`, and tests.

## Quick start

```python
import tangent
import numpy as np

def f(x):
    return np.sum(x ** 2)

df = tangent.grad(f)
print(df(np.array([1.0, 2.0, 3.0])))   # [2. 4. 6.]
```
