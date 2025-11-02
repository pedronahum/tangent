"""Unit tests for visualization tools.

This module tests Tangent's visualization and debugging features.
"""
import pytest
import numpy as np
import io
import sys

# Check if visualization dependencies are available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    import networkx as nx
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

if VIZ_AVAILABLE:
    import tangent
    from tangent.visualization import (
        ComputationGraphVisualizer,
        GradientFlowVisualizer,
        visualize,
        compare_gradients,
        show_gradient_code
    )

pytestmark = pytest.mark.skipif(
    not VIZ_AVAILABLE,
    reason="Visualization requires matplotlib and networkx"
)


class TestComputationGraphVisualizer:
    """Test computation graph visualization."""

    def test_visualizer_creation(self):
        """Test creating visualizer instance."""
        def f(x):
            return x * x

        viz = ComputationGraphVisualizer(f)
        assert viz.func == f
        assert viz.wrt == (0,)
        assert viz.graph is None

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        def f(x):
            y = x * x
            return y

        viz = ComputationGraphVisualizer(f)
        viz._parse_function()

        assert viz.graph is not None
        assert isinstance(viz.graph, nx.DiGraph)
        assert len(viz.graph.nodes()) > 0

    def test_parse_complex_function(self):
        """Test parsing a more complex function."""
        def f(x):
            y = x * x
            z = y + x
            return z

        viz = ComputationGraphVisualizer(f)
        viz._parse_function()

        assert viz.graph is not None
        # Should have nodes for x, y, z, and output
        assert len(viz.graph.nodes()) >= 3

    def test_visualize_creates_figure(self):
        """Test that visualize creates a matplotlib figure."""
        def f(x):
            return x * x

        viz = ComputationGraphVisualizer(f)
        fig = viz.visualize()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGradientFlowVisualizer:
    """Test gradient flow visualization."""

    def test_visualizer_creation(self):
        """Test creating gradient flow visualizer."""
        def f(x):
            return x * x

        viz = GradientFlowVisualizer(f)
        assert viz.func == f
        assert viz.wrt == (0,)

    def test_trace_execution(self):
        """Test tracing function execution."""
        def f(x):
            return x * x

        viz = GradientFlowVisualizer(f)
        result = viz.trace_execution(3.0)

        assert result == 9.0
        assert 'output' in viz.forward_values
        assert viz.forward_values['output'] == 9.0

    def test_visualize_flow_scalar(self):
        """Test visualizing gradient flow for scalar input."""
        def f(x):
            return x * x

        viz = GradientFlowVisualizer(f)
        fig = viz.visualize_flow(3.0)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_flow_vector(self):
        """Test visualizing gradient flow for vector input."""
        def f(x):
            return np.sum(x * x)

        viz = GradientFlowVisualizer(f)
        x = np.array([1.0, 2.0, 3.0])
        fig = viz.visualize_flow(x)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_flow_multivariate(self):
        """Test visualizing gradient flow for multiple inputs."""
        def f(x, y):
            return x * y

        viz = GradientFlowVisualizer(f, wrt=(0, 1))
        fig = viz.visualize_flow(2.0, 3.0)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizeFunction:
    """Test the main visualize() function."""

    def test_visualize_graph_mode(self):
        """Test visualize in graph mode."""
        def f(x):
            y = x * x
            return y

        fig = visualize(f, mode='graph')
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_flow_mode(self):
        """Test visualize in flow mode."""
        def f(x):
            return x * x

        fig = visualize(f, mode='flow', inputs=(3.0,))
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_invalid_mode(self):
        """Test visualize with invalid mode."""
        def f(x):
            return x * x

        with pytest.raises(ValueError, match="Unknown mode"):
            visualize(f, mode='invalid')

    def test_visualize_flow_without_inputs(self):
        """Test visualize flow mode without inputs raises error."""
        def f(x):
            return x * x

        with pytest.raises(ValueError, match="inputs argument required"):
            visualize(f, mode='flow')

    def test_visualize_custom_figsize(self):
        """Test visualize with custom figure size."""
        def f(x):
            return x * x

        fig = visualize(f, mode='graph', figsize=(10, 6))
        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)


class TestCompareGradients:
    """Test gradient comparison function."""

    def test_compare_scalar_function(self):
        """Test comparing gradients for scalar function."""
        def f(x):
            return x * x

        fig = compare_gradients(f, (3.0,))
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_vector_function(self):
        """Test comparing gradients for vector function."""
        def f(x):
            return np.sum(x ** 2)

        x = np.array([1.0, 2.0, 3.0])
        fig = compare_gradients(f, (x,))

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_multivariate(self):
        """Test comparing gradients for multivariate function."""
        def f(x, y):
            return x * y

        fig = compare_gradients(f, (2.0, 3.0), wrt=(0, 1))

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_polynomial(self):
        """Test gradient comparison for polynomial."""
        def f(x):
            return np.sum(x**3 - 2*x**2 + x)

        x = np.array([1.0, 2.0, 3.0])
        fig = compare_gradients(f, (x,), eps=1e-7)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_custom_figsize(self):
        """Test gradient comparison with custom figure size."""
        def f(x):
            return x * x

        fig = compare_gradients(f, (3.0,), figsize=(10, 4))

        assert fig is not None
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 4
        plt.close(fig)


class TestShowGradientCode:
    """Test show_gradient_code function."""

    def test_show_gradient_code_basic(self):
        """Test showing gradient code for basic function."""
        def f(x):
            return x * x

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            show_gradient_code(f)
            output = captured_output.getvalue()

            # Check that output contains expected sections
            assert "GRADIENT CODE FOR" in output
            assert "ORIGINAL FUNCTION" in output
            assert "GENERATED GRADIENT FUNCTION" in output
            assert "def f(x):" in output or "return x * x" in output

        finally:
            sys.stdout = sys.__stdout__

    def test_show_gradient_code_multivar(self):
        """Test showing gradient code for multivariate function."""
        def f(x, y):
            return x * y

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            show_gradient_code(f, wrt=(0, 1))
            output = captured_output.getvalue()

            assert "GRADIENT CODE FOR" in output
            assert "f" in output

        finally:
            sys.stdout = sys.__stdout__

    def test_show_gradient_code_complex(self):
        """Test showing gradient code for complex function."""
        def f(x):
            y = x * x
            z = y + x
            return z

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            show_gradient_code(f)
            output = captured_output.getvalue()

            assert "GRADIENT CODE FOR" in output
            assert len(output) > 100  # Should have substantial output

        finally:
            sys.stdout = sys.__stdout__


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skip(reason="Constant functions are an edge case in Tangent core")
    def test_constant_function(self):
        """Test gradient of constant function."""
        def f(x):
            return 5.0

        # This should work without error
        fig = visualize(f, mode='flow', inputs=(3.0,))
        assert fig is not None
        plt.close(fig)

    def test_linear_function(self):
        """Test gradient of linear function."""
        def f(x):
            return 2.0 * x

        fig = compare_gradients(f, (3.0,))
        assert fig is not None
        plt.close(fig)

    def test_nested_operations(self):
        """Test function with nested operations."""
        def f(x):
            y = x * x
            z = y * y
            return z

        fig = visualize(f, mode='graph')
        assert fig is not None
        plt.close(fig)

    def test_multiple_returns_intermediate(self):
        """Test function with multiple intermediate values."""
        def f(x):
            a = x + 1
            b = a * 2
            c = b - 3
            return c * c

        fig = visualize(f, mode='graph')
        assert fig is not None
        plt.close(fig)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_scalar(self):
        """Test complete workflow for scalar function."""
        def f(x):
            return x ** 3

        # Graph visualization
        fig1 = visualize(f, mode='graph')
        assert fig1 is not None
        plt.close(fig1)

        # Flow visualization
        fig2 = visualize(f, mode='flow', inputs=(2.0,))
        assert fig2 is not None
        plt.close(fig2)

        # Gradient comparison
        fig3 = compare_gradients(f, (2.0,))
        assert fig3 is not None
        plt.close(fig3)

    def test_full_workflow_vector(self):
        """Test complete workflow for vector function."""
        def f(x):
            return np.sum(x ** 2)

        x = np.array([1.0, 2.0, 3.0])

        # Graph visualization
        fig1 = visualize(f, mode='graph')
        assert fig1 is not None
        plt.close(fig1)

        # Flow visualization
        fig2 = visualize(f, mode='flow', inputs=(x,))
        assert fig2 is not None
        plt.close(fig2)

        # Gradient comparison
        fig3 = compare_gradients(f, (x,))
        assert fig3 is not None
        plt.close(fig3)

    def test_educational_example(self):
        """Test a typical educational example."""
        def neural_layer(x):
            """Simple neural layer computation."""
            z = x * x + x
            a = np.tanh(z)
            return np.sum(a)

        x = np.array([0.5, 1.0, 1.5])

        # Show code
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            show_gradient_code(neural_layer)
            output = captured_output.getvalue()
            assert len(output) > 0
        finally:
            sys.stdout = sys.__stdout__

        # Visualize
        fig = compare_gradients(neural_layer, (x,))
        assert fig is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
