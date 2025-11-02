"""Unit tests for JAX integration.

This module tests Tangent's automatic differentiation with JAX arrays and operations.
Tests are organized by operation category and use pytest fixtures for setup.
"""
import pytest
import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    import tangent

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestBasicOperations:
    """Test basic arithmetic operations with JAX."""

    def test_square(self):
        """Test gradient of x^2."""
        def f(x):
            return x * x

        df = tangent.grad(f)
        x = jnp.array(3.0)
        result = df(x)
        expected = 6.0

        assert abs(float(result) - expected) < 1e-5

    def test_polynomial(self):
        """Test gradient of polynomial: 3x^2 + 2x + 1."""
        def f(x):
            return 3.0 * x * x + 2.0 * x + 1.0

        df = tangent.grad(f)
        x = jnp.array(2.0)
        result = df(x)
        expected = 14.0  # 6*2 + 2

        assert abs(float(result) - expected) < 1e-5

    def test_add(self):
        """Test gradient of addition."""
        def f(x, y):
            return jnp.sum(x + y)

        df_dx = tangent.grad(f, wrt=(0,))
        df_dy = tangent.grad(f, wrt=(1,))

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        assert jnp.allclose(grad_x, jnp.ones_like(x))
        assert jnp.allclose(grad_y, jnp.ones_like(y))

    def test_multiply(self):
        """Test gradient of multiplication."""
        def f(x, y):
            return jnp.sum(x * y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([2.0, 3.0])
        y = jnp.array([4.0, 5.0])

        result = df_dx(x, y)
        expected = y

        assert jnp.allclose(result, expected)

    def test_divide(self):
        """Test gradient of division."""
        def f(x, y):
            return jnp.sum(x / y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([4.0, 9.0])
        y = jnp.array([2.0, 3.0])

        result = df_dx(x, y)
        expected = 1.0 / y

        assert jnp.allclose(result, expected)


class TestMathFunctions:
    """Test mathematical functions."""

    def test_exp(self):
        """Test gradient of exponential function."""
        def f(x):
            return jnp.sum(jnp.exp(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, 1.0])
        result = df(x)
        expected = jnp.exp(x)

        assert jnp.allclose(result, expected)

    def test_log(self):
        """Test gradient of logarithm."""
        def f(x):
            return jnp.sum(jnp.log(x))

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 4.0])
        result = df(x)
        expected = 1.0 / x

        assert jnp.allclose(result, expected)

    def test_sqrt(self):
        """Test gradient of square root."""
        def f(x):
            return jnp.sum(jnp.sqrt(x))

        df = tangent.grad(f)
        x = jnp.array([1.0, 4.0, 9.0])
        result = df(x)
        expected = 1.0 / (2.0 * jnp.sqrt(x))

        assert jnp.allclose(result, expected)

    def test_power(self):
        """Test gradient of power function."""
        def f(x):
            return jnp.sum(jnp.power(x, 3.0))

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = df(x)
        expected = 3.0 * jnp.power(x, 2.0)

        assert jnp.allclose(result, expected)

    def test_sin(self):
        """Test gradient of sine function."""
        def f(x):
            return jnp.sum(jnp.sin(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, jnp.pi/4, jnp.pi/2])
        result = df(x)
        expected = jnp.cos(x)

        assert jnp.allclose(result, expected)

    def test_cos(self):
        """Test gradient of cosine function."""
        def f(x):
            return jnp.sum(jnp.cos(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, jnp.pi/4, jnp.pi/2])
        result = df(x)
        expected = -jnp.sin(x)

        assert jnp.allclose(result, expected)

    def test_tanh(self):
        """Test gradient of hyperbolic tangent."""
        def f(x):
            return jnp.sum(jnp.tanh(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, 1.0, 2.0])
        result = df(x)
        expected = 1.0 - jnp.tanh(x) ** 2

        assert jnp.allclose(result, expected)


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_dot_vectors(self):
        """Test gradient of vector dot product."""
        def f(x, y):
            return jnp.dot(x, y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])

        result = df_dx(x, y)
        expected = y

        assert jnp.allclose(result, expected)

    def test_dot_matrices(self):
        """Test gradient of matrix multiplication."""
        def f(x, y):
            return jnp.sum(jnp.dot(x, y))

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = df_dx(x, y)

        # Gradient should have same shape as x
        assert result.shape == x.shape

    def test_matmul(self):
        """Test gradient of matmul."""
        def f(x):
            return jnp.sum(jnp.matmul(x, x))

        df = tangent.grad(f)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = df(x)

        # When x appears twice, result may be a tuple
        if isinstance(result, tuple):
            result = result[0]

        # Gradient should have same shape as input
        assert result.shape == x.shape


class TestReductions:
    """Test reduction operations."""

    def test_sum(self):
        """Test gradient of sum."""
        def f(x):
            return jnp.sum(x * x)

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = df(x)
        expected = 2.0 * x

        assert jnp.allclose(result, expected)

    def test_mean(self):
        """Test gradient of mean."""
        def f(x):
            return jnp.mean(x * x)

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = df(x)
        expected = 2.0 * x / 3.0

        assert jnp.allclose(result, expected)

    def test_max(self):
        """Test gradient of max."""
        def f(x):
            return jnp.max(x * x)

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = df(x)

        # Gradient should be non-zero only at maximum
        assert result.shape == x.shape
        assert float(result[2]) != 0.0  # Max is at index 2

    def test_sum_with_axis(self):
        """Test gradient of sum with axis parameter."""
        def f(x):
            return jnp.sum(jnp.sum(x, axis=0))

        df = tangent.grad(f)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = df(x)
        expected = jnp.ones_like(x)

        assert jnp.allclose(result, expected)


class TestActivations:
    """Test neural network activation functions."""

    def test_relu(self):
        """Test gradient of ReLU."""
        def f(x):
            return jnp.sum(jax.nn.relu(x))

        df = tangent.grad(f)
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = df(x)
        expected = (x > 0).astype(float)

        assert jnp.allclose(result, expected)

    def test_sigmoid(self):
        """Test gradient of sigmoid."""
        def f(x):
            return jnp.sum(jax.nn.sigmoid(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, 1.0, 2.0])
        result = df(x)

        sig = jax.nn.sigmoid(x)
        expected = sig * (1.0 - sig)

        assert jnp.allclose(result, expected)

    def test_softplus(self):
        """Test gradient of softplus."""
        def f(x):
            return jnp.sum(jax.nn.softplus(x))

        df = tangent.grad(f)
        x = jnp.array([0.0, 1.0, 2.0])
        result = df(x)
        expected = jax.nn.sigmoid(x)

        assert jnp.allclose(result, expected)

    def test_elu(self):
        """Test gradient of ELU."""
        def f(x):
            return jnp.sum(jax.nn.elu(x))

        df = tangent.grad(f)
        x = jnp.array([-1.0, 0.0, 1.0])
        result = df(x)

        # Gradient should be 1 for x > 0, alpha * exp(x) for x <= 0
        assert result.shape == x.shape

    def test_leaky_relu(self):
        """Test gradient of Leaky ReLU."""
        def f(x):
            return jnp.sum(jax.nn.leaky_relu(x))

        df = tangent.grad(f)
        x = jnp.array([-1.0, 0.0, 1.0])
        result = df(x)

        # Gradient should be 1 for x > 0, negative_slope for x <= 0
        expected = jnp.where(x > 0, 1.0, 0.01)
        assert jnp.allclose(result, expected)


class TestElementwise:
    """Test element-wise operations."""

    def test_maximum(self):
        """Test gradient of maximum."""
        def f(x, y):
            return jnp.sum(jnp.maximum(x, y))

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([1.0, 3.0, 2.0])
        y = jnp.array([2.0, 1.0, 2.0])

        result = df_dx(x, y)

        # Gradient flows to larger argument
        expected = (x >= y).astype(float)
        assert jnp.allclose(result, expected)

    def test_minimum(self):
        """Test gradient of minimum."""
        def f(x, y):
            return jnp.sum(jnp.minimum(x, y))

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([1.0, 3.0, 2.0])
        y = jnp.array([2.0, 1.0, 2.0])

        result = df_dx(x, y)

        # Gradient flows to smaller argument
        expected = (x <= y).astype(float)
        assert jnp.allclose(result, expected)

    def test_negative(self):
        """Test gradient of negation."""
        def f(x):
            return jnp.sum(jnp.negative(x))

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        result = df(x)
        expected = -jnp.ones_like(x)

        assert jnp.allclose(result, expected)


class TestBroadcasting:
    """Test operations with broadcasting."""

    def test_broadcast_add(self):
        """Test gradient with broadcasted addition."""
        def f(x, y):
            return jnp.sum(x + y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([5.0, 6.0])  # Will be broadcast

        result = df_dx(x, y)

        # Gradient should have same shape as x
        assert result.shape == x.shape
        assert jnp.allclose(result, jnp.ones_like(x))

    def test_broadcast_multiply(self):
        """Test gradient with broadcasted multiplication."""
        def f(x, y):
            return jnp.sum(x * y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([2.0, 3.0])  # Will be broadcast

        result = df_dx(x, y)

        # Gradient should be y broadcast to x's shape
        assert result.shape == x.shape


class TestComposedOperations:
    """Test composed operations."""

    def test_neural_network_layer(self):
        """Test gradient of simple neural network layer."""
        def f(x, W, b):
            return jnp.sum(jax.nn.relu(jnp.dot(x, W) + b))

        df_dW = tangent.grad(f, wrt=(1,))

        x = jnp.array([1.0, 2.0])
        W = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        b = jnp.array([0.1, 0.2])

        result = df_dW(x, W, b)

        # Gradient should have same shape as W
        assert result.shape == W.shape

    def test_polynomial_composition(self):
        """Test gradient of composed polynomial."""
        def f(x):
            y = x * x
            z = y * y
            return jnp.sum(z)

        df = tangent.grad(f)
        x = jnp.array([1.0, 2.0])
        result = df(x)

        # d/dx(x^4) = 4x^3
        expected = 4.0 * x ** 3
        assert jnp.allclose(result, expected)


class TestMultipleGradients:
    """Test gradients with respect to multiple arguments."""

    def test_multiple_wrt(self):
        """Test gradient with respect to multiple arguments."""
        def f(x, y):
            return jnp.sum(x * y)

        # tangent.grad with wrt=(0, 1) returns a single function that returns tuple of gradients
        df = tangent.grad(f, wrt=(0, 1))

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        grad_x, grad_y = df(x, y)

        assert jnp.allclose(grad_x, y)
        assert jnp.allclose(grad_y, x)


class TestPreserveResult:
    """Test preserve_result parameter."""

    def test_preserve_result_true(self):
        """Test gradient computation with result preservation."""
        def f(x):
            return jnp.sum(x * x)

        df = tangent.grad(f, preserve_result=True)
        x = jnp.array([1.0, 2.0, 3.0])
        grad, result = df(x)

        expected_grad = 2.0 * x
        expected_result = jnp.sum(x * x)

        assert jnp.allclose(grad, expected_grad)
        assert abs(float(result) - float(expected_result)) < 1e-5


class TestCaching:
    """Test function caching."""

    def test_cache_hit(self):
        """Test that gradient functions are cached."""
        def f(x):
            return x * x

        # First call - should compile
        df1 = tangent.grad(f)

        # Second call - should use cache
        df2 = tangent.grad(f)

        # Both should work
        x = jnp.array(3.0)
        result1 = df1(x)
        result2 = df2(x)

        assert abs(float(result1) - 6.0) < 1e-5
        assert abs(float(result2) - 6.0) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
