"""Test TF subscripting issue from notebook cell 22"""
import tangent
import tensorflow as tf

def simple_layer(x, W, b):
    """Simple neural network layer - returns scalar"""
    linear = tf.matmul(tf.reshape(x, [1, -1]), W) + b
    activation = tf.tanh(linear)
    # Cell 22 uses subscripting: activation[0, 0] + activation[0, 1]
    return activation[0, 0] + activation[0, 1]

print("Testing TF layer with subscripting...")
dlayer_dW = tangent.grad(simple_layer, wrt=(1,))

x = tf.constant([1.0, 2.0, 3.0])
W = tf.constant([[0.5, 0.3], [0.2, 0.7], [0.1, 0.4]])
b = tf.constant([0.1, 0.2])

try:
    gradient = dlayer_dW(x, W, b)
    print(f"✓ Gradient shape: {gradient.shape}")
    print(f"✓ Gradient:\n{gradient.numpy()}")
    print("\n✓ TensorFlow subscripting works!")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
