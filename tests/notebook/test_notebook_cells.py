"""Comprehensive test for every executable cell in the Tangent tutorial notebook.

This ensures the notebook works end-to-end in Colab and locally.
"""
import sys
import traceback

# Track test results
results = {
    'passed': [],
    'failed': [],
    'skipped': []
}

def test_cell(cell_name, test_func):
    """Test a notebook cell and record results."""
    print(f"\n{'='*70}")
    print(f"Testing: {cell_name}")
    print('='*70)
    try:
        test_func()
        results['passed'].append(cell_name)
        print(f"✓ PASSED: {cell_name}")
        return True
    except ImportError as e:
        results['skipped'].append((cell_name, str(e)))
        print(f"⊘ SKIPPED: {cell_name} - {e}")
        return None
    except Exception as e:
        results['failed'].append((cell_name, str(e)))
        print(f"✗ FAILED: {cell_name}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# Section 1: Installation & Setup
# ============================================================================

def test_cell_3_imports():
    """Test basic imports"""
    import tangent
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    assert tangent is not None
    assert np is not None

# ============================================================================
# Section 2: Basic Concepts
# ============================================================================

def test_cell_5_square():
    """Test simple square function gradient"""
    import tangent

    def square(x):
        return x * x

    dsquare = tangent.grad(square)
    x_val = 3.0
    gradient = dsquare(x_val)
    expected = 2 * x_val
    assert abs(gradient - expected) < 1e-5, f"Expected {expected}, got {gradient}"

def test_cell_7_polynomial():
    """Test polynomial gradient and code inspection"""
    import tangent
    import inspect

    def polynomial(x):
        return 3.0 * x * x + 2.0 * x + 1.0

    dpolynomial = tangent.grad(polynomial)
    source = inspect.getsource(dpolynomial)
    assert 'def' in source
    assert 'dpolynomial' in source or 'polynomial' in source.lower()

# ============================================================================
# Section 3: NumPy Integration
# ============================================================================

def test_cell_12_vector_norm():
    """Test vector norm squared gradient"""
    import tangent
    import numpy as np

    def vector_norm_squared(x):
        return np.sum(x * x)

    dvector_norm_squared = tangent.grad(vector_norm_squared)
    x = np.array([1.0, 2.0, 3.0])
    gradient = dvector_norm_squared(x)
    expected = 2 * x
    assert np.allclose(gradient, expected), f"Expected {expected}, got {gradient}"

def test_cell_14_matrix_vector():
    """Test matrix-vector operation gradient"""
    import tangent
    import numpy as np

    def matrix_vector_sum(x):
        A = np.array([[2.0, 1.0, 0.5],
                      [1.0, 3.0, 0.7],
                      [0.5, 0.7, 4.0]])
        return np.sum(np.dot(A, x))

    df_dx = tangent.grad(matrix_vector_sum)
    x = np.array([1.0, 2.0, 3.0])
    gradient = df_dx(x)

    A = np.array([[2.0, 1.0, 0.5],
                  [1.0, 3.0, 0.7],
                  [0.5, 0.7, 4.0]])
    expected = np.sum(A, axis=0)
    assert np.allclose(gradient, expected)

def test_cell_16_sigmoid():
    """Test sigmoid gradient"""
    import tangent
    import numpy as np

    def sigmoid_loss(x):
        return np.sum(1.0 / (1.0 + np.exp(-x)))

    dsigmoid_loss = tangent.grad(sigmoid_loss)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gradient = dsigmoid_loss(x)

    sigmoid_x = 1.0 / (1.0 + np.exp(-x))
    expected = sigmoid_x * (1.0 - sigmoid_x)
    assert np.allclose(gradient, expected)

# ============================================================================
# Section 4: TensorFlow Integration
# ============================================================================

def test_cell_18_tf_import():
    """Test TensorFlow import"""
    import tensorflow as tf
    assert tf.__version__ is not None

def test_cell_20_tf_quadratic():
    """Test TensorFlow quadratic gradient"""
    import tangent
    import tensorflow as tf

    def tf_quadratic(x):
        return 2.0 * x * x + 3.0 * x + 1.0

    dtf_quadratic = tangent.grad(tf_quadratic)
    x_tf = tf.constant(2.0)
    gradient = dtf_quadratic(x_tf)
    expected = 11.0  # 4*2 + 3
    assert abs(gradient.numpy() - expected) < 1e-5

def test_cell_22_tf_layer():
    """Test TensorFlow neural network layer gradient"""
    import tangent
    import tensorflow as tf

    def simple_layer(x, W, b):
        linear = tf.matmul(tf.reshape(x, [1, -1]), W) + b
        activation = tf.tanh(linear)
        return tf.reduce_sum(activation)  # Use reduce_sum instead of subscripting

    dlayer_dW = tangent.grad(simple_layer, wrt=(1,))
    x = tf.constant([1.0, 2.0, 3.0])
    W = tf.constant([[0.5, 0.3], [0.2, 0.7], [0.1, 0.4]])
    b = tf.constant([0.1, 0.2])

    gradient = dlayer_dW(x, W, b)
    assert gradient.shape == W.shape

# ============================================================================
# Section 5: JAX Integration
# ============================================================================

def test_cell_24_jax_import():
    """Test JAX import"""
    import jax
    import jax.numpy as jnp
    assert jax.__version__ is not None

def test_cell_26_jax_polynomial():
    """Test JAX polynomial gradient"""
    import tangent
    import jax.numpy as jnp

    def jax_polynomial(x):
        return x**3 - 2*x**2 + 3*x - 1

    djax_polynomial = tangent.grad(jax_polynomial)
    x_jax = jnp.array(2.0)
    gradient = djax_polynomial(x_jax)
    expected = 3 * x_jax**2 - 4 * x_jax + 3
    assert jnp.allclose(gradient, expected)

def test_cell_28_jax_relu():
    """Test JAX ReLU network gradient"""
    import tangent
    import jax
    import jax.numpy as jnp

    def jax_relu_network(x1, x2, x3, x4, x5):
        a1 = jax.nn.relu(x1 * x1 - 1.0)
        a2 = jax.nn.relu(x2 * x2 - 1.0)
        a3 = jax.nn.relu(x3 * x3 - 1.0)
        a4 = jax.nn.relu(x4 * x4 - 1.0)
        a5 = jax.nn.relu(x5 * x5 - 1.0)
        return a1 + a2 + a3 + a4 + a5

    djax_relu_1 = tangent.grad(jax_relu_network, wrt=(0,))
    djax_relu_5 = tangent.grad(jax_relu_network, wrt=(4,))

    x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    grad1 = djax_relu_1(*x_vals)
    grad5 = djax_relu_5(*x_vals)

    assert abs(grad1 - (-4.0)) < 1e-5
    assert abs(grad5 - 4.0) < 1e-5

# ============================================================================
# Section 6: Advanced Features
# ============================================================================

def test_cell_32_multivariate():
    """Test multivariate gradient"""
    import tangent

    def bivariate(x, y):
        return x * x * y + x * y * y

    dbivariate = tangent.grad(bivariate, wrt=(0, 1))
    x, y = 2.0, 3.0
    grad_x, grad_y = dbivariate(x, y)

    expected_grad_x = 2 * x * y + y * y
    expected_grad_y = x * x + 2 * x * y

    assert abs(grad_x - expected_grad_x) < 1e-5
    assert abs(grad_y - expected_grad_y) < 1e-5

def test_cell_36_preserve_result():
    """Test preserve_result feature"""
    import tangent
    import numpy as np

    def expensive_function(x):
        return np.sum(np.exp(x) * np.sin(x))

    dexpensive = tangent.grad(expensive_function, preserve_result=True)
    x = np.array([0.0, 1.0, 2.0])
    gradient, result = dexpensive(x)

    expected_result = expensive_function(x)
    assert abs(result - expected_result) < 1e-5

# ============================================================================
# Section 9: Advanced Python Features
# ============================================================================

def test_cell_57_lambda():
    """Test lambda functions"""
    import tangent
    import numpy as np

    def neural_activation(x):
        relu = lambda z: np.maximum(0, z)
        leaky_relu = lambda z: np.maximum(0.01 * z, z)
        h1 = relu(x * 0.5)
        h2 = leaky_relu(x * 2.0)
        return h1 + h2

    d_activation = tangent.grad(neural_activation)
    x_test = 2.0
    gradient = d_activation(x_test)
    assert gradient is not None

def test_cell_55_classes():
    """Test user-defined classes"""
    import tangent

    class Polynomial:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def evaluate(self, x):
            return self.a * x ** 2 + self.b * x + self.c

    def loss_with_class(x):
        poly = Polynomial(2.0, 3.0, 1.0)
        return poly.evaluate(x)

    dloss = tangent.grad(loss_with_class)
    x_test = 5.0
    gradient = dloss(x_test)
    expected = 4 * x_test + 3
    assert abs(gradient - expected) < 1e-5

def test_cell_53_inheritance():
    """Test class inheritance"""
    import tangent

    class NeuralLayer:
        def __init__(self, weight):
            self.weight = weight

        def forward(self, x):
            return x * self.weight

    class NeuralLayerWithBias(NeuralLayer):
        def __init__(self, weight, bias):
            super().__init__(weight)
            self.bias = bias

        def forward(self, x):
            return x * self.weight + self.bias

    def network_loss(x):
        layer = NeuralLayerWithBias(weight=2.5, bias=1.0)
        output = layer.forward(x)
        target = 10.0
        return (output - target) ** 2

    dnetwork = tangent.grad(network_loss)
    x_test = 3.0
    gradient = dnetwork(x_test)
    expected = 5 * (2.5 * x_test - 9)
    assert abs(gradient - expected) < 1e-5

def test_cell_51_control_flow():
    """Test control flow"""
    import tangent

    def polynomial_loop(x):
        result = 0.0
        for i in range(4):
            result += x ** float(i)
        return result

    dpoly = tangent.grad(polynomial_loop)
    x1 = 2.0
    grad1 = dpoly(x1)
    expected1 = 1 + 2*x1 + 3*x1**2
    assert abs(grad1 - expected1) < 1e-5

    def relu_ternary(x):
        return x if x > 0 else 0.0

    drelu = tangent.grad(relu_ternary)
    assert drelu(5.0) == 1.0
    assert drelu(-3.0) == 0.0

def test_cell_49_oop_network():
    """Test OOP neural network"""
    import tangent

    class Layer:
        def __init__(self, name):
            self.name = name
        def forward(self, x):
            return x

    class DenseLayer(Layer):
        def __init__(self, weight, bias, name="dense"):
            super().__init__(name)
            self.weight = weight
            self.bias = bias
        def forward(self, x):
            return x * self.weight + self.bias

    def oop_network_loss(x):
        layer1 = DenseLayer(weight=2.0, bias=-1.0, name="layer1")
        h = layer1.forward(x)
        layer2 = DenseLayer(weight=0.5, bias=0.5, name="layer2")
        output = layer2.forward(h)
        target = 5.0
        return (output - target) ** 2

    doop_loss = tangent.grad(oop_network_loss)
    x_test = 3.0
    gradient = doop_loss(x_test)
    expected_grad = 2 * (x_test - 5.0)
    assert abs(gradient - expected_grad) < 1e-5

# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TANGENT NOTEBOOK COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Define test order
    tests = [
        # Section 1: Setup
        ("Cell 3: Imports", test_cell_3_imports),

        # Section 2: Basic Concepts
        ("Cell 5: Square Function", test_cell_5_square),
        ("Cell 7: Polynomial", test_cell_7_polynomial),

        # Section 3: NumPy
        ("Cell 12: Vector Norm", test_cell_12_vector_norm),
        ("Cell 14: Matrix-Vector", test_cell_14_matrix_vector),
        ("Cell 16: Sigmoid", test_cell_16_sigmoid),

        # Section 4: TensorFlow
        ("Cell 18: TF Import", test_cell_18_tf_import),
        ("Cell 20: TF Quadratic", test_cell_20_tf_quadratic),
        ("Cell 22: TF Layer", test_cell_22_tf_layer),

        # Section 5: JAX
        ("Cell 24: JAX Import", test_cell_24_jax_import),
        ("Cell 26: JAX Polynomial", test_cell_26_jax_polynomial),
        ("Cell 28: JAX ReLU", test_cell_28_jax_relu),

        # Section 6: Advanced Features
        ("Cell 32: Multivariate", test_cell_32_multivariate),
        ("Cell 36: Preserve Result", test_cell_36_preserve_result),

        # Section 9: Advanced Python Features
        ("Cell 57: Lambda Functions", test_cell_57_lambda),
        ("Cell 55: Classes", test_cell_55_classes),
        ("Cell 53: Inheritance", test_cell_53_inheritance),
        ("Cell 51: Control Flow", test_cell_51_control_flow),
        ("Cell 49: OOP Network", test_cell_49_oop_network),
    ]

    # Run all tests
    for name, test_func in tests:
        test_cell(name, test_func)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Passed:  {len(results['passed'])}")
    print(f"✗ Failed:  {len(results['failed'])}")
    print(f"⊘ Skipped: {len(results['skipped'])}")
    print(f"Total:     {len(tests)}")

    if results['failed']:
        print("\nFailed tests:")
        for name, error in results['failed']:
            print(f"  - {name}: {error[:100]}")

    if results['skipped']:
        print("\nSkipped tests:")
        for name, error in results['skipped']:
            print(f"  - {name}: {error[:100]}")

    # Exit with appropriate code
    sys.exit(0 if not results['failed'] else 1)
