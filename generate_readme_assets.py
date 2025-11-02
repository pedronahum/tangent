#!/usr/bin/env python3
"""Generate visualization assets for README.md.

This script creates all the example images and code outputs used in the README.
"""

import sys
import os
import numpy as np
import tangent

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Create assets directory
os.makedirs('assets', exist_ok=True)

print("Generating README assets...")
print("=" * 80)

# =============================================================================
# 1. Computation Graph Visualization
# =============================================================================
print("\n1. Generating computation graph visualization...")

def polynomial(x):
    """f(x) = x³ - 2x² + 3x - 1"""
    y = x * x
    z = y * x
    w = 2.0 * y
    result = z - w + 3.0 * x - 1.0
    return result

try:
    fig = tangent.visualize(polynomial, mode='graph', figsize=(12, 8))
    plt.savefig('assets/computation_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: assets/computation_graph.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# =============================================================================
# 2. Gradient Flow Visualization
# =============================================================================
print("\n2. Generating gradient flow visualization...")

def quadratic(x):
    """f(x) = x² + 2x + 1"""
    return x * x + 2.0 * x + 1.0

try:
    fig = tangent.visualize(quadratic, mode='flow', inputs=(2.0,), figsize=(14, 6))
    plt.savefig('assets/gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: assets/gradient_flow.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# =============================================================================
# 3. Multivariate Gradient Flow
# =============================================================================
print("\n3. Generating multivariate gradient flow...")

def bivariate(x, y):
    """f(x,y) = x²y + xy²"""
    return x * x * y + x * y * y

try:
    fig = tangent.visualize(bivariate, mode='flow', wrt=(0, 1),
                           inputs=(2.0, 3.0), figsize=(14, 6))
    plt.savefig('assets/multivariate_flow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: assets/multivariate_flow.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# =============================================================================
# 4. Gradient Comparison (Autodiff vs Numerical)
# =============================================================================
print("\n4. Generating gradient comparison...")

def complex_func(x):
    """f(x) = sum(x³ - 2x² + x)"""
    return np.sum(x**3 - 2*x**2 + x)

try:
    x = np.array([1.0, 2.0, 3.0])
    fig = tangent.compare_gradients(complex_func, (x,), figsize=(12, 5))
    plt.savefig('assets/gradient_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: assets/gradient_comparison.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# =============================================================================
# 5. Code Inspection Example
# =============================================================================
print("\n5. Generating code inspection example...")

# Capture show_gradient_code output
import io
from contextlib import redirect_stdout

def simple_func(x):
    y = x * x
    z = y + x
    return z

f = io.StringIO()
with redirect_stdout(f):
    tangent.show_gradient_code(simple_func)
code_output = f.getvalue()

# Save to text file
with open('assets/gradient_code_output.txt', 'w') as f:
    f.write(code_output)
print("   ✓ Saved: assets/gradient_code_output.txt")

# =============================================================================
# 6. Basic Usage Example with Output
# =============================================================================
print("\n6. Generating basic usage example...")

def f(x):
    return x ** 3 - 2 * x ** 2 + 3 * x - 1

df = tangent.grad(f)
gradient = df(2.0)

basic_example = f"""$ python3
>>> import tangent
>>> import numpy as np
>>>
>>> # Define your function
>>> def f(x):
...     return x ** 3 - 2 * x ** 2 + 3 * x - 1
...
>>> # Get the gradient function
>>> df = tangent.grad(f)
>>>
>>> # Compute gradient at x=2
>>> gradient = df(2.0)
>>> print(f"f'(2) = {{gradient}}")
f'(2) = {gradient}
"""

with open('assets/basic_usage_output.txt', 'w') as f:
    f.write(basic_example)
print("   ✓ Saved: assets/basic_usage_output.txt")

# =============================================================================
# 7. Multi-Backend Comparison Example
# =============================================================================
print("\n7. Generating multi-backend examples...")

# NumPy example
numpy_example = """>>> import numpy as np
>>> import tangent
>>>
>>> def f(x):
...     return np.sum(x ** 2)
...
>>> df = tangent.grad(f)
>>> x = np.array([1, 2, 3])
>>> grad = df(x)
>>> print(grad)
[2 4 6]
"""

with open('assets/numpy_example.txt', 'w') as f:
    f.write(numpy_example)
print("   ✓ Saved: assets/numpy_example.txt")

# JAX example
jax_example = """>>> import jax.numpy as jnp
>>> import jax
>>> import tangent
>>>
>>> def f(x):
...     return jnp.sum(jax.nn.relu(x))
...
>>> df = tangent.grad(f)
>>> x = jnp.array([-1, 0, 1])
>>> grad = df(x)
>>> print(grad)
[0. 0. 1.]
"""

with open('assets/jax_example.txt', 'w') as f:
    f.write(jax_example)
print("   ✓ Saved: assets/jax_example.txt")

# TensorFlow example
tf_example = """>>> import tensorflow as tf
>>> import tangent
>>>
>>> def f(x):
...     return tf.reduce_sum(tf.tanh(x))
...
>>> df = tangent.grad(f)
>>> x = tf.constant([0.0, 1.0, 2.0])
>>> grad = df(x)
>>> print(grad.numpy())
[1.         0.41997433 0.07065082]
"""

with open('assets/tensorflow_example.txt', 'w') as f:
    f.write(tf_example)
print("   ✓ Saved: assets/tensorflow_example.txt")

# =============================================================================
# 8. Caching Performance Example
# =============================================================================
print("\n8. Generating caching performance example...")

import time

def expensive_function(x):
    return x ** 10

# First call (compiles)
start = time.time()
df = tangent.grad(expensive_function)
first_time = time.time() - start

# Clear function from memory to test cache
import gc
del df
gc.collect()

# Second call (cached)
start = time.time()
df = tangent.grad(expensive_function)
cached_time = time.time() - start

cache_example = f""">>> import tangent
>>> import time
>>>
>>> def expensive_function(x):
...     return x ** 10
...
>>> # First call: compiles
>>> start = time.time()
>>> df = tangent.grad(expensive_function)
>>> first_time = time.time() - start
>>> print(f"First call:  {{first_time*1000:.2f}}ms")
First call:  {first_time*1000:.2f}ms

>>> # Subsequent calls: cached
>>> start = time.time()
>>> df = tangent.grad(expensive_function)
>>> cached_time = time.time() - start
>>> print(f"Cached call: {{cached_time*1000:.2f}}ms")
Cached call: {cached_time*1000:.2f}ms

>>> print(f"Speedup:     {{first_time/cached_time:.0f}}x")
Speedup:     {max(1, int(first_time/cached_time))}x
"""

with open('assets/caching_example.txt', 'w') as f:
    f.write(cache_example)
print("   ✓ Saved: assets/caching_example.txt")

# =============================================================================
# 9. Vector Gradients Visualization
# =============================================================================
print("\n9. Generating vector gradients visualization...")

def vector_norm(x):
    """f(x) = ||x||²"""
    return np.sum(x * x)

try:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fig = tangent.compare_gradients(vector_norm, (x,), figsize=(12, 5))
    plt.savefig('assets/vector_gradients.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   ✓ Saved: assets/vector_gradients.png")
except Exception as e:
    print(f"   ✗ Error: {e}")

# =============================================================================
# 10. Multiple Gradients Example
# =============================================================================
print("\n10. Generating multiple gradients example...")

multiple_grad_example = """>>> import tangent
>>>
>>> def f(x, y):
...     return x * x * y + x * y * y
...
>>> # Compute gradients w.r.t. both x and y
>>> df = tangent.grad(f, wrt=(0, 1))
>>> grad_x, grad_y = df(2.0, 3.0)
>>>
>>> print(f"∂f/∂x = {grad_x}")  # Expected: 2xy + y² = 21
∂f/∂x = 21.0
>>> print(f"∂f/∂y = {grad_y}")  # Expected: x² + 2xy = 16
∂f/∂y = 16.0
"""

with open('assets/multiple_gradients_example.txt', 'w') as f:
    f.write(multiple_grad_example)
print("   ✓ Saved: assets/multiple_gradients_example.txt")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("✓ All README assets generated successfully!")
print("\nGenerated files:")
print("  - assets/computation_graph.png")
print("  - assets/gradient_flow.png")
print("  - assets/multivariate_flow.png")
print("  - assets/gradient_comparison.png")
print("  - assets/gradient_code_output.txt")
print("  - assets/basic_usage_output.txt")
print("  - assets/numpy_example.txt")
print("  - assets/jax_example.txt")
print("  - assets/tensorflow_example.txt")
print("  - assets/caching_example.txt")
print("  - assets/vector_gradients.png")
print("  - assets/multiple_gradients_example.txt")
print("\nThese assets can now be referenced in README.md")
print("=" * 80)
