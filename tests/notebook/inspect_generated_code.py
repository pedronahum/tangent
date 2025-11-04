"""Inspect actual generated gradient code"""
import tangent
import jax.numpy as jnp
import inspect
import tempfile

def simple_sum(x):
    """Simple JAX sum"""
    return jnp.sum(x)

print("=" * 80)
print("Generated Gradient Code for jnp.sum")
print("=" * 80)

# Generate the gradient
try:
    grad_fn = tangent.grad(simple_sum)

    # Get the source code
    source = inspect.getsource(grad_fn)
    print("\n" + source)

    # Find the problematic line
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        if 'numpy.sum' in line or 'jnp' in line and 'sum' in line:
            print(f"\n⚠️  Line {i}: {line}")

except Exception as e:
    print(f"Error generating gradient: {e}")
    import traceback
    traceback.print_exc()
