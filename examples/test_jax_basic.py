"""Basic JAX integration tests for Tangent.

This test suite validates that Tangent can differentiate functions using
JAX's numpy-like API (jax.numpy).
"""

import jax
import jax.numpy as jnp
import tangent
import numpy as np


def jax_square(x):
    """Simple squaring function."""
    return x * x


def jax_polynomial(x):
    """Polynomial function: 3x^2 + 2x + 1"""
    return 3.0 * x * x + 2.0 * x + 1.0


def jax_matmul_sum(x):
    """Matrix multiplication and sum."""
    return jnp.sum(jnp.dot(x, x))


# Activation functions
def jax_tanh(x):
    """Tanh activation."""
    return jnp.sum(jnp.tanh(x))


def jax_sigmoid(x):
    """Sigmoid activation."""
    return jnp.sum(jax.nn.sigmoid(x))


def jax_relu(x):
    """ReLU activation."""
    return jnp.sum(jax.nn.relu(x))


def jax_exp(x):
    """Exponential function."""
    return jnp.sum(jnp.exp(x))


# Reduction operations
def jax_reduce_mean(x):
    """Reduce mean."""
    return jnp.mean(x * x)


def jax_reduce_max(x):
    """Reduce max with computation."""
    return jnp.max(x * x + x)


# Element-wise operations
def jax_multiply_add(x):
    """Combined multiply and add."""
    y = x * 2.0
    z = y + 3.0
    return jnp.sum(z * z)


def jax_divide(x):
    """Division operation."""
    y = x * x
    z = y / (x + 1.0)
    return jnp.sum(z)


def jax_log(x):
    """Logarithm operation."""
    return jnp.sum(jnp.log(x + 1.0))


def jax_sqrt(x):
    """Square root operation."""
    return jnp.sum(jnp.sqrt(x))


def test_simple_grad():
    """Test basic gradient computation."""
    print("\n1. Testing grad(x^2) at x=3.0")
    df = tangent.grad(jax_square)
    x = jnp.array(3.0)
    result = df(x)
    expected = 6.0

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert abs(float(result) - expected) < 0.001, f"Expected {expected}, got {result}"
    print("   ✓ PASS")


def test_polynomial_grad():
    """Test polynomial gradient."""
    print("\n2. Testing grad(3x^2 + 2x + 1) at x=2.0")
    df = tangent.grad(jax_polynomial)
    x = jnp.array(2.0)
    result = df(x)
    expected = 14.0  # 6*2 + 2

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert abs(float(result) - expected) < 0.001, f"Expected {expected}, got {result}"
    print("   ✓ PASS")


def test_matmul_grad():
    """Test matrix multiplication gradient."""
    print("\n3. Testing grad(sum(x @ x)) for 2x2 matrix")
    df = tangent.grad(jax_matmul_sum)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = df(x)

    print(f"   Result shape: {result.shape}")
    print(f"   Result:\n{result}")

    # The gradient should have the same shape as input
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    print("   ✓ PASS (shape matches)")


def test_tanh_grad():
    """Test tanh gradient."""
    print("\n4. Testing grad(sum(tanh(x))) at x=[1.0, 2.0]")
    df = tangent.grad(jax_tanh)
    x = jnp.array([1.0, 2.0])
    result = df(x)

    # Gradient of tanh(x) is 1 - tanh(x)^2
    expected = 1.0 - jnp.tanh(x)**2

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_sigmoid_grad():
    """Test sigmoid gradient."""
    print("\n5. Testing grad(sum(sigmoid(x))) at x=[0.0, 1.0]")
    df = tangent.grad(jax_sigmoid)
    x = jnp.array([0.0, 1.0])
    result = df(x)

    # Gradient of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    sig = jax.nn.sigmoid(x)
    expected = sig * (1.0 - sig)

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_relu_grad():
    """Test ReLU gradient."""
    print("\n6. Testing grad(sum(relu(x))) at x=[-1.0, 0.0, 1.0, 2.0]")
    df = tangent.grad(jax_relu)
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = df(x)

    # Gradient of relu(x) is 1 if x > 0, else 0
    expected = (x > 0).astype(float)

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_exp_grad():
    """Test exponential gradient."""
    print("\n7. Testing grad(sum(exp(x))) at x=[0.0, 1.0]")
    df = tangent.grad(jax_exp)
    x = jnp.array([0.0, 1.0])
    result = df(x)

    # Gradient of exp(x) is exp(x)
    expected = jnp.exp(x)

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_reduce_mean_grad():
    """Test reduce_mean gradient."""
    print("\n8. Testing grad(mean(x^2)) at x=[1.0, 2.0, 3.0]")
    df = tangent.grad(jax_reduce_mean)
    x = jnp.array([1.0, 2.0, 3.0])
    result = df(x)

    # Gradient of mean(x^2) is 2x/n
    expected = 2.0 * x / 3.0

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_reduce_max_grad():
    """Test reduce_max gradient."""
    print("\n9. Testing grad(max(x^2 + x)) at x=[1.0, 2.0, 3.0]")
    df = tangent.grad(jax_reduce_max)
    x = jnp.array([1.0, 2.0, 3.0])
    result = df(x)

    print(f"   Result: {result}")
    # Gradient is non-zero only at the maximum
    assert result.shape == x.shape
    print("   ✓ PASS (shape matches)")


def test_multiply_add_grad():
    """Test combined multiply and add gradient."""
    print("\n10. Testing grad(sum((2x + 3)^2)) at x=1.0")
    df = tangent.grad(jax_multiply_add)
    x = jnp.array(1.0)
    result = df(x)

    # d/dx sum((2x + 3)^2) = 2 * (2x + 3) * 2 = 4(2x + 3)
    expected = 4.0 * (2.0 * x + 3.0)

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert abs(float(result) - float(expected)) < 0.001
    print("   ✓ PASS")


def test_divide_grad():
    """Test division gradient."""
    print("\n11. Testing grad(sum(x^2 / (x + 1))) at x=2.0")
    df = tangent.grad(jax_divide)
    x = jnp.array(2.0)
    result = df(x)

    print(f"   Result: {result}")
    # Just check it computes without error
    assert isinstance(float(result), (float, np.floating))
    print("   ✓ PASS (computed successfully)")


def test_log_grad():
    """Test logarithm gradient."""
    print("\n12. Testing grad(sum(log(x + 1))) at x=[1.0, 2.0]")
    df = tangent.grad(jax_log)
    x = jnp.array([1.0, 2.0])
    result = df(x)

    # Gradient of log(x+1) is 1/(x+1)
    expected = 1.0 / (x + 1.0)

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


def test_sqrt_grad():
    """Test square root gradient."""
    print("\n13. Testing grad(sum(sqrt(x))) at x=[1.0, 4.0, 9.0]")
    df = tangent.grad(jax_sqrt)
    x = jnp.array([1.0, 4.0, 9.0])
    result = df(x)

    # Gradient of sqrt(x) is 1/(2*sqrt(x))
    expected = 1.0 / (2.0 * jnp.sqrt(x))

    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    assert jnp.allclose(result, expected, atol=0.001)
    print("   ✓ PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Tangent with JAX")
    print(f"JAX version: {jax.__version__}")
    print("=" * 60)

    try:
        # Basic operations
        test_simple_grad()
        test_polynomial_grad()
        test_matmul_grad()

        # Activation functions
        test_tanh_grad()
        test_sigmoid_grad()
        test_relu_grad()
        test_exp_grad()

        # Reduction operations
        test_reduce_mean_grad()
        test_reduce_max_grad()

        # Element-wise operations
        test_multiply_add_grad()
        test_divide_grad()
        test_log_grad()
        test_sqrt_grad()

        print("\n" + "=" * 60)
        print("All JAX tests PASSED! (13/13)")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
