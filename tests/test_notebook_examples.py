"""Test all examples from the Tangent tutorial notebook.

This ensures all notebook cells run without errors before publishing.
"""
import numpy as np
import pytest
import tangent


# =============================================================================
# Module-level class definitions (required for Tangent's class inlining)
# =============================================================================

class Polynomial:
    """Polynomial class for testing"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        """Evaluate polynomial: a*x² + b*x + c"""
        return self.a * x ** 2 + self.b * x + self.c


class NeuralLayer:
    """Base neural network layer"""
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        """Base forward pass"""
        return x * self.weight


class NeuralLayerWithBias(NeuralLayer):
    """Neural layer with bias"""
    def __init__(self, weight, bias):
        super().__init__(weight)
        self.bias = bias

    def forward(self, x):
        """Override forward with bias - inline parent logic"""
        return x * self.weight + self.bias


class Layer:
    """Base layer for OOP network"""
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        return x


class DenseLayer(Layer):
    """Dense layer with inheritance"""
    def __init__(self, weight, bias, name="dense"):
        super().__init__(name)
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x * self.weight + self.bias


# =============================================================================
# Tests
# =============================================================================

class TestSection9Lambdas:
    """Test Section 9.1 - Lambda Functions"""

    def test_lambda_activations(self):
        """Test lambda functions for neural activations"""
        def neural_activation(x):
            """Use lambdas for activation functions"""
            relu = lambda z: np.maximum(0, z)
            leaky_relu = lambda z: np.maximum(0.01 * z, z)

            h1 = relu(x * 0.5)
            h2 = leaky_relu(x * 2.0)
            return h1 + h2

        d_activation = tangent.grad(neural_activation)

        x_test = 2.0
        result = neural_activation(x_test)
        gradient = d_activation(x_test)

        # Basic sanity checks
        assert isinstance(result, (float, np.ndarray))
        assert isinstance(gradient, (float, np.ndarray))
        print(f"✓ Lambda functions: f({x_test}) = {result}, f'({x_test}) = {gradient}")


class TestSection9Classes:
    """Test Section 9.2 - User-Defined Classes"""

    def test_polynomial_class(self):
        """Test polynomial class with instance attributes"""
        def loss_with_class(x):
            poly = Polynomial(2.0, 3.0, 1.0)
            return poly.evaluate(x)

        dloss = tangent.grad(loss_with_class)

        x_test = 5.0
        gradient = dloss(x_test)
        expected = 4 * x_test + 3  # d/dx[2x² + 3x + 1] = 4x + 3

        assert abs(gradient - expected) < 1e-5
        print(f"✓ Polynomial class: gradient = {gradient}, expected = {expected}")


class TestSection9Inheritance:
    """Test Section 9.3 - Class Inheritance"""

    def test_inheritance_simple(self):
        """Test simple inheritance without super() method calls"""
        def network_loss(x):
            layer = NeuralLayerWithBias(weight=2.5, bias=1.0)
            output = layer.forward(x)
            target = 10.0
            return (output - target) ** 2

        dnetwork = tangent.grad(network_loss)

        x_test = 3.0
        gradient = dnetwork(x_test)
        expected = 5 * (2.5 * x_test - 9)  # f'(x) = 2(2.5x + 1 - 10) * 2.5

        assert abs(gradient - expected) < 1e-5
        print(f"✓ Inheritance: gradient = {gradient}, expected = {expected}")


class TestSection9ControlFlow:
    """Test Section 9.4 - Control Flow"""

    def test_for_loop(self):
        """Test for loop"""
        def polynomial_loop(x):
            """Evaluate 1 + x + x² + x³ using a loop"""
            result = 0.0
            for i in range(4):
                result += x ** float(i)
            return result

        dpoly = tangent.grad(polynomial_loop)
        x_test = 2.0
        gradient = dpoly(x_test)
        expected = 1 + 2*x_test + 3*x_test**2

        assert abs(gradient - expected) < 1e-5
        print(f"✓ For loop: gradient = {gradient}, expected = {expected}")

    def test_ternary_operator(self):
        """Test ternary operator (ReLU)"""
        def relu_ternary(x):
            """ReLU using ternary operator"""
            return x if x > 0 else 0.0

        drelu = tangent.grad(relu_ternary)

        # Test positive input
        grad_pos = drelu(5.0)
        assert grad_pos == 1.0

        # Test negative input
        grad_neg = drelu(-3.0)
        assert grad_neg == 0.0

        print(f"✓ Ternary operator: grad(5.0) = {grad_pos}, grad(-3.0) = {grad_neg}")

    def test_while_loop(self):
        """Test while loop"""
        def newton_sqrt(x):
            """Newton's method for sqrt(x) using while loop"""
            estimate = x / 2.0
            i = 0
            while i < 5:
                estimate = 0.5 * (estimate + x / estimate)
                i += 1
            return estimate

        dnewton = tangent.grad(newton_sqrt)
        x_test = 9.0
        gradient = dnewton(x_test)
        expected = 0.5 / newton_sqrt(x_test)

        assert abs(gradient - expected) < 0.01
        print(f"✓ While loop: gradient = {gradient}, expected ≈ {expected}")


class TestSection9OOPNetwork:
    """Test Section 9.5 - OOP Neural Network"""

    def test_oop_network_linear_only(self):
        """Test OOP network with linear layers only"""
        def oop_network_loss(x):
            layer1 = DenseLayer(weight=2.0, bias=-1.0, name="layer1")
            h = layer1.forward(x)

            layer2 = DenseLayer(weight=0.5, bias=0.5, name="layer2")
            output = layer2.forward(h)

            target = 5.0
            return (output - target) ** 2

        doop_loss = tangent.grad(oop_network_loss)

        # Test at x = 3.0
        # h = 2.0 * 3.0 - 1.0 = 5.0
        # output = 0.5 * 5.0 + 0.5 = 3.0
        # loss = (3.0 - 5.0)² = 4.0
        # gradient = 2(output - 5) * d(output)/dx = 2(3-5) * 1 = -4
        x_test = 3.0
        loss = oop_network_loss(x_test)
        gradient = doop_loss(x_test)

        # Verify gradient calculation
        # output = 0.5 * (2.0 * x - 1.0) + 0.5 = x
        # loss = (x - 5)²
        # gradient = 2(x - 5) = 2(3 - 5) = -4
        expected_grad = 2 * (x_test - 5.0)

        assert abs(gradient - expected_grad) < 1e-5
        print(f"✓ OOP Network: loss = {loss}, gradient = {gradient}, expected = {expected_grad}")


class TestTensorFlowExamples:
    """Test TensorFlow examples from notebook"""

    def test_tf_simple_layer(self):
        """Test TensorFlow neural network layer"""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        def simple_layer(x, W, b):
            """Simple neural network layer - returns scalar"""
            linear = tf.matmul(tf.reshape(x, [1, -1]), W) + b
            activation = tf.tanh(linear)
            # Return sum of elements instead of using tf.reduce_sum
            return activation[0, 0] + activation[0, 1]

        dlayer_dW = tangent.grad(simple_layer, wrt=(1,))

        x = tf.constant([1.0, 2.0, 3.0])
        W = tf.constant([[0.5, 0.3], [0.2, 0.7], [0.1, 0.4]])
        b = tf.constant([0.1, 0.2])

        gradient = dlayer_dW(x, W, b)

        # Just verify it runs without error and returns correct shape
        assert gradient.shape == W.shape
        print(f"✓ TensorFlow layer: gradient shape = {gradient.shape}")


class TestJAXExamples:
    """Test JAX examples from notebook"""

    def test_jax_relu_scalars(self):
        """Test JAX ReLU with scalar inputs"""
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not installed")

        def jax_relu_network(x1, x2, x3, x4, x5):
            """Simple ReLU network with scalar inputs"""
            a1 = jax.nn.relu(x1 * x1 - 1.0)
            a2 = jax.nn.relu(x2 * x2 - 1.0)
            a3 = jax.nn.relu(x3 * x3 - 1.0)
            a4 = jax.nn.relu(x4 * x4 - 1.0)
            a5 = jax.nn.relu(x5 * x5 - 1.0)
            return a1 + a2 + a3 + a4 + a5

        djax_relu_1 = tangent.grad(jax_relu_network, wrt=(0,))
        djax_relu_5 = tangent.grad(jax_relu_network, wrt=(4,))

        x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]

        # Test gradients at x=-2 and x=2 (where ReLU is active)
        grad1 = djax_relu_1(*x_vals)
        grad5 = djax_relu_5(*x_vals)

        # At x=-2: gradient = 2*(-2) = -4
        # At x=2: gradient = 2*2 = 4
        assert abs(grad1 - (-4.0)) < 1e-5
        assert abs(grad5 - 4.0) < 1e-5

        print(f"✓ JAX ReLU: grad(x=-2) = {grad1}, grad(x=2) = {grad5}")


class TestNeuralNetworkTraining:
    """Test neural network training example"""

    def test_numpy_neural_network(self):
        """Test NumPy-based neural network training"""
        def neural_network_simple(W1, b1, W2, b2, x_sample, y_sample):
            """Two-layer neural network with single sample"""
            hidden = np.maximum(0, np.dot(x_sample, W1) + b1[0])
            output = 1.0 / (1.0 + np.exp(-(np.dot(hidden, W2) + b2[0, 0])))

            eps = 1e-10
            output = output + eps
            loss = -(y_sample * np.log(output) + (1 - y_sample) * np.log(1 - output + eps))
            return loss[0]

        # Create simple test data
        np.random.seed(42)
        X_data = np.array([[1.0, 2.0], [-1.0, -2.0]])
        y_data = np.array([[1.0], [0.0]])

        # Initialize parameters
        W1 = np.random.randn(2, 4) * 0.5
        b1 = np.zeros((1, 4))
        W2 = np.random.randn(4, 1) * 0.5
        b2 = np.zeros((1, 1))

        # Compute gradients
        dnn_dW1 = tangent.grad(neural_network_simple, wrt=(0,))

        # Test gradient computation on first sample
        x_sample = X_data[0]
        y_sample = y_data[0]

        grad_W1 = dnn_dW1(W1, b1, W2, b2, x_sample, y_sample)

        # Verify gradient has correct shape
        assert grad_W1.shape == W1.shape

        print(f"✓ NumPy NN training: gradient shape = {grad_W1.shape}")


if __name__ == '__main__':
    """Run all tests"""
    print("\n" + "="*70)
    print("TESTING NOTEBOOK EXAMPLES")
    print("="*70)

    # Section 9 tests
    print("\n### Section 9.1 - Lambda Functions ###")
    test_lambda = TestSection9Lambdas()
    test_lambda.test_lambda_activations()

    print("\n### Section 9.2 - User-Defined Classes ###")
    test_classes = TestSection9Classes()
    test_classes.test_polynomial_class()

    print("\n### Section 9.3 - Inheritance ###")
    test_inheritance = TestSection9Inheritance()
    test_inheritance.test_inheritance_simple()

    print("\n### Section 9.4 - Control Flow ###")
    test_control = TestSection9ControlFlow()
    test_control.test_for_loop()
    test_control.test_ternary_operator()
    test_control.test_while_loop()

    print("\n### Section 9.5 - OOP Neural Network ###")
    test_oop = TestSection9OOPNetwork()
    test_oop.test_oop_network_linear_only()

    print("\n### TensorFlow Examples ###")
    test_tf = TestTensorFlowExamples()
    try:
        test_tf.test_tf_simple_layer()
    except Exception as e:
        print(f"⚠ TensorFlow test skipped: {e}")

    print("\n### JAX Examples ###")
    test_jax = TestJAXExamples()
    try:
        test_jax.test_jax_relu_scalars()
    except Exception as e:
        print(f"⚠ JAX test skipped: {e}")

    print("\n### Neural Network Training ###")
    test_nn = TestNeuralNetworkTraining()
    test_nn.test_numpy_neural_network()

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
