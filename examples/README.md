# Tangent Examples

This directory contains example scripts and demonstrations of Tangent's capabilities.

## Files

- **test_basic.py** - Basic Tangent examples with NumPy
- **test_tf2_basic.py** - TensorFlow 2.x integration examples (9 tests)
- **test_jax_basic.py** - JAX integration examples (13 tests)
- **demo_error_messages.py** - Demonstration of enhanced error messages

## Running Examples

### JAX Examples
```bash
python examples/test_jax_basic.py
```

### TensorFlow 2.x Examples
```bash
python examples/test_tf2_basic.py
```

### Error Messages Demo
```bash
python examples/demo_error_messages.py
```

## Quick Start

```python
import tangent
import jax.numpy as jnp

def f(x):
    return jnp.sum(x ** 2)

df = tangent.grad(f)
gradient = df(jnp.array([1.0, 2.0, 3.0]))
# Result: [2., 4., 6.]
```
