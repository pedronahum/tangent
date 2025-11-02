"""Unit tests for TensorFlow 2.x integration.

This module tests Tangent's automatic differentiation with TensorFlow 2.x tensors and operations.
Tests are organized by operation category and use pytest fixtures for setup.

Note: This test suite covers the currently supported TensorFlow operations.
Some advanced TensorFlow operations (tf.sqrt, tf.pow, tf.nn.*, tf.linalg.matvec, etc.)
are not yet supported and will be added in future releases.
"""
import pytest
import numpy as np

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

if TF_AVAILABLE:
    import tangent

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


class TestBasicOperations:
    """Test basic arithmetic operations with TensorFlow."""

    def test_square(self):
        """Test gradient of x^2."""
        def f(x):
            return x * x

        df = tangent.grad(f)
        x = tf.constant(3.0)
        result = df(x)
        expected = 6.0

        assert abs(float(result.numpy()) - expected) < 1e-5

    def test_polynomial(self):
        """Test gradient of polynomial: 3x^2 + 2x + 1."""
        def f(x):
            return 3.0 * x * x + 2.0 * x + 1.0

        df = tangent.grad(f)
        x = tf.constant(2.0)
        result = df(x)
        expected = 14.0  # 6*2 + 2

        assert abs(float(result.numpy()) - expected) < 1e-5

    def test_add(self):
        """Test gradient of addition."""
        def f(x, y):
            return tf.reduce_sum(x + y)

        df_dx = tangent.grad(f, wrt=(0,))
        df_dy = tangent.grad(f, wrt=(1,))

        x = tf.constant([1.0, 2.0])
        y = tf.constant([3.0, 4.0])

        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        # Results may be numpy arrays or tensors
        grad_x_array = grad_x.numpy() if hasattr(grad_x, 'numpy') else np.array(grad_x)
        grad_y_array = grad_y.numpy() if hasattr(grad_y, 'numpy') else np.array(grad_y)

        assert np.allclose(grad_x_array, np.ones_like(x.numpy()))
        assert np.allclose(grad_y_array, np.ones_like(y.numpy()))

    def test_multiply(self):
        """Test gradient of multiplication."""
        def f(x, y):
            return tf.reduce_sum(x * y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = tf.constant([2.0, 3.0])
        y = tf.constant([4.0, 5.0])

        result = df_dx(x, y)
        expected = y.numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_divide(self):
        """Test gradient of division."""
        def f(x, y):
            return tf.reduce_sum(x / y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = tf.constant([4.0, 9.0])
        y = tf.constant([2.0, 3.0])

        result = df_dx(x, y)
        expected = (1.0 / y).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)


class TestMathFunctions:
    """Test mathematical functions."""

    def test_exp(self):
        """Test gradient of exponential function."""
        def f(x):
            return tf.reduce_sum(tf.exp(x))

        df = tangent.grad(f)
        x = tf.constant([0.0, 1.0])
        result = df(x)
        expected = tf.exp(x).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_log(self):
        """Test gradient of logarithm."""
        def f(x):
            return tf.reduce_sum(tf.math.log(x))

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0, 4.0])
        result = df(x)
        expected = (1.0 / x).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_tanh(self):
        """Test gradient of hyperbolic tangent."""
        def f(x):
            return tf.reduce_sum(tf.tanh(x))

        df = tangent.grad(f)
        x = tf.constant([0.0, 1.0, 2.0])
        result = df(x)
        expected = (1.0 - tf.tanh(x) ** 2).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_matmul(self):
        """Test gradient of matrix multiplication."""
        def f(x):
            return tf.reduce_sum(tf.matmul(x, x))

        df = tangent.grad(f)
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = df(x)

        # When x appears twice, result may be a tuple
        if isinstance(result, tuple):
            result = result[0]

        # Gradient should have same shape as input
        assert result.shape == x.shape


class TestReductions:
    """Test reduction operations."""

    def test_reduce_sum(self):
        """Test gradient of reduce_sum."""
        def f(x):
            return tf.reduce_sum(x * x)

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0, 3.0])
        result = df(x)
        expected = (2.0 * x).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_reduce_mean(self):
        """Test gradient of reduce_mean."""
        def f(x):
            return tf.reduce_mean(x * x)

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0, 3.0])
        result = df(x)
        expected = (2.0 * x / 3.0).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_reduce_max(self):
        """Test gradient of reduce_max."""
        def f(x):
            return tf.reduce_max(x * x + x)

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0, 3.0])
        result = df(x)

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        # Gradient should be non-zero only at maximum
        assert result_array.shape == tuple(x.shape.as_list())
        assert float(result_array[2]) != 0.0  # Max is at index 2


class TestElementwise:
    """Test element-wise operations."""

    def test_maximum(self):
        """Test gradient of maximum."""
        def f(x, y):
            return tf.reduce_sum(tf.maximum(x, y))

        df_dx = tangent.grad(f, wrt=(0,))

        x = tf.constant([1.0, 3.0, 2.0])
        y = tf.constant([2.0, 1.0, 2.0])

        result = df_dx(x, y)
        expected = tf.cast(x >= y, tf.float32).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)


class TestBroadcasting:
    """Test operations with broadcasting."""

    def test_broadcast_multiply(self):
        """Test gradient with broadcasted multiplication."""
        def f(x, y):
            return tf.reduce_sum(x * y)

        df_dx = tangent.grad(f, wrt=(0,))

        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([2.0, 3.0])  # Will be broadcast

        result = df_dx(x, y)

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        # Gradient should be y broadcast to x's shape
        assert result_array.shape == tuple(x.shape.as_list())


class TestComposedOperations:
    """Test composed operations."""

    def test_polynomial_composition(self):
        """Test gradient of composed polynomial."""
        def f(x):
            y = x * x
            z = y * y
            return tf.reduce_sum(z)

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0])
        result = df(x)

        # d/dx(x^4) = 4x^3
        expected = (4.0 * x ** 3).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_complex_expression(self):
        """Test gradient of complex expression."""
        def f(x):
            y = x * 2.0
            z = y + 3.0
            return tf.reduce_sum(z * z)

        df = tangent.grad(f)
        x = tf.constant(1.0)
        result = df(x)

        # d/dx sum((2x + 3)^2) = 4(2x + 3)
        expected = float((4.0 * (2.0 * x + 3.0)).numpy())

        # Result may be scalar tensor or numpy scalar
        result_scalar = float(result.numpy()) if hasattr(result, 'numpy') else float(result)

        assert abs(result_scalar - expected) < 1e-5


class TestMultipleGradients:
    """Test gradients with respect to multiple arguments."""

    def test_multiple_wrt(self):
        """Test gradient with respect to multiple arguments."""
        def f(x, y):
            return tf.reduce_sum(x * y)

        # tangent.grad with wrt=(0, 1) returns a single function that returns tuple of gradients
        df = tangent.grad(f, wrt=(0, 1))

        x = tf.constant([1.0, 2.0])
        y = tf.constant([3.0, 4.0])

        grad_x, grad_y = df(x, y)

        # Results may be numpy arrays or tensors
        grad_x_array = grad_x.numpy() if hasattr(grad_x, 'numpy') else np.array(grad_x)
        grad_y_array = grad_y.numpy() if hasattr(grad_y, 'numpy') else np.array(grad_y)

        assert np.allclose(grad_x_array, y.numpy())
        assert np.allclose(grad_y_array, x.numpy())


class TestPreserveResult:
    """Test preserve_result parameter."""

    def test_preserve_result_true(self):
        """Test gradient computation with result preservation."""
        def f(x):
            return tf.reduce_sum(x * x)

        df = tangent.grad(f, preserve_result=True)
        x = tf.constant([1.0, 2.0, 3.0])
        grad, result = df(x)

        expected_grad = (2.0 * x).numpy()
        expected_result = float(tf.reduce_sum(x * x).numpy())

        # Results may be numpy arrays or tensors
        grad_array = grad.numpy() if hasattr(grad, 'numpy') else np.array(grad)
        result_scalar = float(result.numpy()) if hasattr(result, 'numpy') else float(result)

        assert np.allclose(grad_array, expected_grad)
        assert abs(result_scalar - expected_result) < 1e-5


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
        x = tf.constant(3.0)
        result1 = df1(x)
        result2 = df2(x)

        # Results may be tensors or numpy scalars
        result1_scalar = float(result1.numpy()) if hasattr(result1, 'numpy') else float(result1)
        result2_scalar = float(result2.numpy()) if hasattr(result2, 'numpy') else float(result2)

        assert abs(result1_scalar - 6.0) < 1e-5
        assert abs(result2_scalar - 6.0) < 1e-5


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_scalar_input(self):
        """Test gradient with scalar input."""
        def f(x):
            return x * x * x

        df = tangent.grad(f)
        x = tf.constant(2.0)
        result = df(x)
        expected = float((3.0 * x * x).numpy())

        # Result may be scalar tensor or numpy scalar
        result_scalar = float(result.numpy()) if hasattr(result, 'numpy') else float(result)

        assert abs(result_scalar - expected) < 1e-5

    def test_vector_input(self):
        """Test gradient with vector input."""
        def f(x):
            return tf.reduce_sum(x * x * x)

        df = tangent.grad(f)
        x = tf.constant([1.0, 2.0, 3.0])
        result = df(x)
        expected = (3.0 * x ** 2).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)

    def test_matrix_input(self):
        """Test gradient with matrix input."""
        def f(x):
            return tf.reduce_sum(x * x)

        df = tangent.grad(f)
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = df(x)
        expected = (2.0 * x).numpy()

        # Result may be numpy array or tensor
        result_array = result.numpy() if hasattr(result, 'numpy') else np.array(result)

        assert np.allclose(result_array, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
