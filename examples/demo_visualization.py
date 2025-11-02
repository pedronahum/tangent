"""Demonstration of Tangent's visualization tools.

This script demonstrates all the visualization features:
1. Computation graph visualization
2. Gradient flow visualization
3. Gradient comparison (autodiff vs numerical)
4. Generated code inspection
"""

import numpy as np
import tangent

# Check if visualization is available
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("⚠️  Visualization requires matplotlib and networkx")
    print("   Install with: pip install matplotlib networkx")
    exit(1)


def demo_computation_graph():
    """Demo 1: Visualize computation graph."""
    print("\n" + "=" * 80)
    print("DEMO 1: Computation Graph Visualization")
    print("=" * 80)

    def polynomial(x):
        """A polynomial function: f(x) = x³ - 2x² + 3x - 1"""
        y = x * x
        z = y * x
        w = 2.0 * y
        result = z - w + 3.0 * x - 1.0
        return result

    print(f"\nVisualizing computation graph for: {polynomial.__name__}")
    print("Function: f(x) = x³ - 2x² + 3x - 1")

    # Visualize computation graph
    fig = tangent.visualize(polynomial, mode='graph')
    plt.savefig('/tmp/tangent_comp_graph.png', dpi=150, bbox_inches='tight')
    print("\n✓ Computation graph saved to: /tmp/tangent_comp_graph.png")
    plt.show()


def demo_gradient_flow():
    """Demo 2: Visualize gradient flow."""
    print("\n" + "=" * 80)
    print("DEMO 2: Gradient Flow Visualization")
    print("=" * 80)

    def quadratic(x):
        """Simple quadratic: f(x) = 3x² + 2x + 1"""
        return 3.0 * x * x + 2.0 * x + 1.0

    x_val = 2.0
    print(f"\nVisualizing gradient flow for: {quadratic.__name__}")
    print(f"Function: f(x) = 3x² + 2x + 1")
    print(f"Input value: x = {x_val}")

    # Visualize gradient flow
    fig = tangent.visualize(quadratic, mode='flow', inputs=(x_val,))
    plt.savefig('/tmp/tangent_grad_flow.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gradient flow saved to: /tmp/tangent_grad_flow.png")
    plt.show()


def demo_multivariate_flow():
    """Demo 3: Multivariate function gradient flow."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multivariate Gradient Flow")
    print("=" * 80)

    def bivariate(x, y):
        """Bivariate function: f(x,y) = x²y + xy²"""
        return x * x * y + x * y * y

    x_val, y_val = 2.0, 3.0
    print(f"\nVisualizing gradient flow for: {bivariate.__name__}")
    print(f"Function: f(x,y) = x²y + xy²")
    print(f"Input values: x = {x_val}, y = {y_val}")

    # Visualize gradient flow
    fig = tangent.visualize(bivariate, mode='flow', wrt=(0, 1), inputs=(x_val, y_val))
    plt.savefig('/tmp/tangent_multivar_flow.png', dpi=150, bbox_inches='tight')
    print("\n✓ Multivariate gradient flow saved to: /tmp/tangent_multivar_flow.png")
    plt.show()


def demo_gradient_comparison():
    """Demo 4: Compare autodiff vs numerical gradients."""
    print("\n" + "=" * 80)
    print("DEMO 4: Gradient Comparison (Autodiff vs Numerical)")
    print("=" * 80)

    def complex_func(x):
        """Complex function: f(x) = sum(x³ - 2x² + e^(x/10))"""
        return np.sum(x**3 - 2*x**2 + np.exp(x/10))

    x = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"\nComparing gradients for: {complex_func.__name__}")
    print(f"Function: f(x) = sum(x³ - 2x² + e^(x/10))")
    print(f"Input: x = {x}")

    # Compare gradients
    fig = tangent.compare_gradients(complex_func, (x,))
    plt.savefig('/tmp/tangent_grad_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gradient comparison saved to: /tmp/tangent_grad_comparison.png")
    plt.show()

    # Show numerical verification
    df = tangent.grad(complex_func)
    auto_grad = df(x)
    print(f"\nAutodiff gradient: {auto_grad}")

    # Numerical gradient for reference
    eps = 1e-7
    num_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        num_grad[i] = (complex_func(x_plus) - complex_func(x_minus)) / (2 * eps)

    print(f"Numerical gradient:  {num_grad}")
    print(f"Difference:          {np.abs(auto_grad - num_grad)}")
    print(f"Max error:           {np.max(np.abs(auto_grad - num_grad)):.2e}")


def demo_code_inspection():
    """Demo 5: Inspect generated gradient code."""
    print("\n" + "=" * 80)
    print("DEMO 5: Generated Gradient Code Inspection")
    print("=" * 80)

    def neural_layer(x):
        """Simple neural layer: f(x) = sum(tanh(x² + x))"""
        z = x * x + x
        activation = np.tanh(z)
        return np.sum(activation)

    print("\nInspecting generated gradient code:")
    tangent.show_gradient_code(neural_layer)


def demo_vector_gradients():
    """Demo 6: Vector function gradients."""
    print("\n" + "=" * 80)
    print("DEMO 6: Vector Function Gradients")
    print("=" * 80)

    def vector_norm(x):
        """Vector norm: f(x) = ||x||² = sum(x²)"""
        return np.sum(x * x)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nFunction: f(x) = ||x||²")
    print(f"Input: x = {x}")

    # Compute gradient
    df = tangent.grad(vector_norm)
    gradient = df(x)

    print(f"\nGradient: ∇f(x) = {gradient}")
    print(f"Expected (2x): {2 * x}")
    print(f"Match: {np.allclose(gradient, 2 * x)} ✓")

    # Compare with numerical
    fig = tangent.compare_gradients(vector_norm, (x,))
    plt.savefig('/tmp/tangent_vector_grad.png', dpi=150, bbox_inches='tight')
    print("\n✓ Vector gradient comparison saved to: /tmp/tangent_vector_grad.png")
    plt.show()


def demo_matrix_operations():
    """Demo 7: Matrix operations."""
    print("\n" + "=" * 80)
    print("DEMO 7: Matrix Operations")
    print("=" * 80)

    def matrix_vector_sum(x):
        """Compute sum(A @ x) where A is a fixed matrix"""
        A = np.array([[2.0, 1.0, 0.5],
                      [1.0, 3.0, 0.7],
                      [0.5, 0.7, 4.0]])
        return np.sum(np.dot(A, x))

    x = np.array([1.0, 2.0, 3.0])
    A = np.array([[2.0, 1.0, 0.5],
                  [1.0, 3.0, 0.7],
                  [0.5, 0.7, 4.0]])

    print(f"\nFunction: f(x) = sum(A @ x)")
    print(f"Input x: {x}")
    print(f"Matrix A:\n{A}")

    # Gradient w.r.t. x
    df_dx = tangent.grad(matrix_vector_sum)
    gradient = df_dx(x)

    print(f"\n∇_x f = {gradient}")
    print(f"Expected (sum of A's columns): {np.sum(A, axis=0)}")
    print(f"Match: {np.allclose(gradient, np.sum(A, axis=0))} ✓")

    # Visualize
    fig = tangent.compare_gradients(matrix_vector_sum, (x,))
    plt.savefig('/tmp/tangent_matrix_grad.png', dpi=150, bbox_inches='tight')
    print("\n✓ Matrix gradient comparison saved to: /tmp/tangent_matrix_grad.png")
    plt.show()


def main():
    """Run all visualization demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "TANGENT VISUALIZATION DEMOS")
    print("=" * 80)
    print("\nThis demo showcases Tangent's visualization and debugging tools:")
    print("  1. Computation graph visualization")
    print("  2. Gradient flow visualization")
    print("  3. Multivariate gradient flow")
    print("  4. Autodiff vs numerical gradient comparison")
    print("  5. Generated code inspection")
    print("  6. Vector function gradients")
    print("  7. Matrix operations")
    print("\nAll visualizations will be saved to /tmp/ and displayed.")

    try:
        demo_computation_graph()
        demo_gradient_flow()
        demo_multivariate_flow()
        demo_gradient_comparison()
        demo_code_inspection()
        demo_vector_gradients()
        demo_matrix_operations()

        print("\n" + "=" * 80)
        print(" " * 25 + "ALL DEMOS COMPLETE!")
        print("=" * 80)
        print("\n📊 Visualizations saved to:")
        print("   - /tmp/tangent_comp_graph.png")
        print("   - /tmp/tangent_grad_flow.png")
        print("   - /tmp/tangent_multivar_flow.png")
        print("   - /tmp/tangent_grad_comparison.png")
        print("   - /tmp/tangent_vector_grad.png")
        print("   - /tmp/tangent_matrix_grad.png")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
