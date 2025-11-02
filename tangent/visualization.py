"""Visualization tools for Tangent - Educational and debugging aids.

This module provides tools to visualize:
- Computation graphs
- Intermediate gradients
- Forward and backward pass flow
- Variable dependencies
"""

import gast
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Install with: pip install networkx")

from tangent import quoting
from tangent import naming
from tangent import grad_util


class ComputationGraphVisualizer:
    """Visualize the computation graph of a function."""

    def __init__(self, func: Callable, wrt: Union[int, Tuple[int, ...]] = (0,)):
        """Initialize visualizer.

        Args:
            func: Function to visualize
            wrt: Arguments to differentiate with respect to
        """
        self.func = func
        self.wrt = wrt if isinstance(wrt, tuple) else (wrt,)
        self.graph = None
        self.node_info = {}
        self.edge_info = {}

    def _parse_function(self):
        """Parse function and build computation graph."""
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for graph visualization. Install with: pip install networkx")

        # Get function AST
        source = inspect.getsource(self.func)
        source = textwrap.dedent(source)
        tree = gast.parse(source)

        # Build graph
        self.graph = nx.DiGraph()
        self._build_graph_from_ast(tree)

    def _build_graph_from_ast(self, tree):
        """Build computation graph from AST."""
        for node in gast.walk(tree):
            if isinstance(node, gast.FunctionDef):
                # Add function inputs
                for i, arg in enumerate(node.args.args):
                    arg_name = arg.arg if hasattr(arg, 'arg') else arg.id
                    self.graph.add_node(arg_name,
                                       node_type='input',
                                       color='lightblue',
                                       shape='box')
                    self.node_info[arg_name] = {'type': 'input', 'index': i}

            elif isinstance(node, gast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], gast.Name):
                    target = node.targets[0].id

                    # Determine operation type
                    if isinstance(node.value, gast.BinOp):
                        op_type = self._get_binop_name(node.value.op)
                        color = 'lightgreen'
                    elif isinstance(node.value, gast.Call):
                        op_type = self._get_call_name(node.value)
                        color = 'lightyellow'
                    else:
                        op_type = 'assign'
                        color = 'lightgray'

                    self.graph.add_node(target,
                                       node_type='operation',
                                       operation=op_type,
                                       color=color,
                                       shape='ellipse')
                    self.node_info[target] = {'type': 'operation', 'op': op_type}

                    # Add edges from dependencies
                    deps = self._get_dependencies(node.value)
                    for dep in deps:
                        if dep in self.graph:
                            self.graph.add_edge(dep, target, label=op_type)

            elif isinstance(node, gast.Return):
                # Add output node
                self.graph.add_node('output',
                                   node_type='output',
                                   color='lightcoral',
                                   shape='box')
                self.node_info['output'] = {'type': 'output'}

                # Connect return value to output
                deps = self._get_dependencies(node.value)
                for dep in deps:
                    if dep in self.graph:
                        self.graph.add_edge(dep, 'output', label='return')

    def _get_binop_name(self, op):
        """Get name of binary operation."""
        op_map = {
            gast.Add: '+',
            gast.Sub: '-',
            gast.Mult: '*',
            gast.Div: '/',
            gast.Pow: '**',
            gast.Mod: '%',
        }
        return op_map.get(type(op), 'binop')

    def _get_call_name(self, call_node):
        """Get name of function call."""
        if isinstance(call_node.func, gast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, gast.Attribute):
            return call_node.func.attr
        return 'call'

    def _get_dependencies(self, node):
        """Get variable dependencies of a node."""
        deps = []
        for child in gast.walk(node):
            if isinstance(child, gast.Name) and isinstance(child.ctx, gast.Load):
                deps.append(child.id)
        return list(set(deps))

    def visualize(self, show_gradients: bool = False, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the computation graph.

        Args:
            show_gradients: If True, show gradient flow (requires computing gradients)
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
            raise ImportError("matplotlib and networkx required. Install with: pip install matplotlib networkx")

        # Parse function if not done yet
        if self.graph is None:
            self._parse_function()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f'Computation Graph: {self.func.__name__}',
                    fontsize=16, fontweight='bold', pad=20)

        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Draw nodes by type
        for node_type, color in [('input', 'lightblue'),
                                 ('operation', 'lightgreen'),
                                 ('output', 'lightcoral')]:
            nodes = [n for n, d in self.graph.nodes(data=True)
                    if d.get('node_type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(self.graph, pos,
                                      nodelist=nodes,
                                      node_color=color,
                                      node_size=2000,
                                      alpha=0.9,
                                      ax=ax)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax)

        # Draw labels
        labels = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'operation':
                labels[node] = f"{node}\n({data.get('operation', '')})"
            else:
                labels[node] = node

        nx.draw_networkx_labels(self.graph, pos, labels,
                               font_size=10,
                               font_weight='bold',
                               ax=ax)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels,
                                     font_size=8,
                                     ax=ax)

        # Legend
        input_patch = mpatches.Patch(color='lightblue', label='Inputs')
        op_patch = mpatches.Patch(color='lightgreen', label='Operations')
        output_patch = mpatches.Patch(color='lightcoral', label='Output')
        ax.legend(handles=[input_patch, op_patch, output_patch],
                 loc='upper left', fontsize=12)

        ax.axis('off')
        plt.tight_layout()
        return fig


class GradientFlowVisualizer:
    """Visualize gradient flow through a function."""

    def __init__(self, func: Callable, wrt: Union[int, Tuple[int, ...]] = (0,)):
        """Initialize gradient flow visualizer.

        Args:
            func: Function to visualize
            wrt: Arguments to differentiate with respect to
        """
        self.func = func
        self.wrt = wrt if isinstance(wrt, tuple) else (wrt,)
        self.forward_values = {}
        self.gradient_values = {}

    def trace_execution(self, *args):
        """Trace function execution and record intermediate values.

        Args:
            *args: Arguments to pass to function

        Returns:
            (result, forward_trace): Function result and forward pass trace
        """
        # For now, we'll capture basic execution
        # A full implementation would use Python's sys.settrace
        result = self.func(*args)
        self.forward_values['output'] = result
        self.forward_values['inputs'] = args
        return result

    def visualize_flow(self, *args, figsize: Tuple[int, int] = (14, 10)):
        """Visualize forward and backward pass with actual values.

        Args:
            *args: Arguments to pass to function
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")

        # Import here to avoid circular dependency
        from tangent import grad

        # Compute forward pass
        result = self.trace_execution(*args)

        # Compute gradients
        df = grad(self.func, wrt=self.wrt, preserve_result=True)
        if len(self.wrt) == 1:
            gradient, _ = df(*args)
            gradients = [gradient]
        else:
            grad_results = df(*args)
            gradients = grad_results[:-1]  # All but last (result)

        # Create visualization
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Gradient Flow: {self.func.__name__}',
                    fontsize=16, fontweight='bold')

        # Forward pass
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_forward_pass(ax1, args, result)

        # Backward pass
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_backward_pass(ax2, args, gradients)

        plt.tight_layout()
        return fig

    def _plot_forward_pass(self, ax, inputs, output):
        """Plot forward pass flow."""
        ax.set_title('Forward Pass', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Create boxes for inputs and output
        n_inputs = len(inputs)
        box_width = 0.8 / (n_inputs + 1)

        # Draw input boxes
        for i, inp in enumerate(inputs):
            x = 0.1 + i * box_width
            box = FancyBboxPatch((x, 0.6), box_width * 0.9, 0.25,
                                boxstyle="round,pad=0.01",
                                facecolor='lightblue',
                                edgecolor='black',
                                linewidth=2,
                                transform=ax.transAxes)
            ax.add_patch(box)

            # Add text
            text = f"x[{i}]\n{inp}" if hasattr(inp, '__iter__') and len(str(inp)) < 50 else f"x[{i}]"
            ax.text(x + box_width * 0.45, 0.725, text,
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)

        # Draw function box
        func_box = FancyBboxPatch((0.35, 0.35), 0.3, 0.15,
                                 boxstyle="round,pad=0.01",
                                 facecolor='lightgreen',
                                 edgecolor='black',
                                 linewidth=2,
                                 transform=ax.transAxes)
        ax.add_patch(func_box)
        ax.text(0.5, 0.425, self.func.__name__,
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes)

        # Draw output box
        out_box = FancyBboxPatch((0.4, 0.05), 0.2, 0.2,
                                boxstyle="round,pad=0.01",
                                facecolor='lightcoral',
                                edgecolor='black',
                                linewidth=2,
                                transform=ax.transAxes)
        ax.add_patch(out_box)

        output_text = f"output\n{output}" if len(str(output)) < 50 else "output"
        ax.text(0.5, 0.15, output_text,
               ha='center', va='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)

        # Draw arrows
        for i in range(n_inputs):
            x = 0.1 + i * box_width + box_width * 0.45
            arrow = FancyArrowPatch((x, 0.6), (0.45, 0.5),
                                   arrowstyle='->,head_width=0.4,head_length=0.4',
                                   color='black', linewidth=2,
                                   transform=ax.transAxes)
            ax.add_patch(arrow)

        arrow = FancyArrowPatch((0.5, 0.35), (0.5, 0.25),
                               arrowstyle='->,head_width=0.4,head_length=0.4',
                               color='black', linewidth=2,
                               transform=ax.transAxes)
        ax.add_patch(arrow)

    def _plot_backward_pass(self, ax, inputs, gradients):
        """Plot backward pass (gradient flow)."""
        ax.set_title('Backward Pass (Gradients)', fontsize=14, fontweight='bold')
        ax.axis('off')

        n_inputs = len(inputs)
        box_width = 0.8 / (n_inputs + 1)

        # Draw gradient boxes (bottom to top)
        # Output gradient (seed = 1)
        out_grad_box = FancyBboxPatch((0.4, 0.65), 0.2, 0.2,
                                     boxstyle="round,pad=0.01",
                                     facecolor='#ffcccc',
                                     edgecolor='red',
                                     linewidth=2,
                                     transform=ax.transAxes)
        ax.add_patch(out_grad_box)
        ax.text(0.5, 0.75, "d(output)\n= 1.0",
               ha='center', va='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)

        # Function gradient box
        func_grad_box = FancyBboxPatch((0.35, 0.35), 0.3, 0.15,
                                      boxstyle="round,pad=0.01",
                                      facecolor='#ccffcc',
                                      edgecolor='green',
                                      linewidth=2,
                                      transform=ax.transAxes)
        ax.add_patch(func_grad_box)
        ax.text(0.5, 0.425, "∇" + self.func.__name__,
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes)

        # Input gradient boxes
        for i, grad in enumerate(gradients):
            x = 0.1 + i * box_width
            grad_box = FancyBboxPatch((x, 0.05), box_width * 0.9, 0.25,
                                     boxstyle="round,pad=0.01",
                                     facecolor='#ccccff',
                                     edgecolor='blue',
                                     linewidth=2,
                                     transform=ax.transAxes)
            ax.add_patch(grad_box)

            # Format gradient value
            if hasattr(grad, 'shape') and len(str(grad)) < 50:
                grad_text = f"∇x[{i}]\n{grad}"
            else:
                grad_text = f"∇x[{i}]"

            ax.text(x + box_width * 0.45, 0.175, grad_text,
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)

        # Draw backward arrows (red)
        arrow = FancyArrowPatch((0.5, 0.65), (0.5, 0.5),
                               arrowstyle='->,head_width=0.4,head_length=0.4',
                               color='red', linewidth=2,
                               transform=ax.transAxes)
        ax.add_patch(arrow)

        for i in range(len(gradients)):
            x = 0.1 + i * box_width + box_width * 0.45
            arrow = FancyArrowPatch((0.47, 0.35), (x, 0.3),
                                   arrowstyle='->,head_width=0.4,head_length=0.4',
                                   color='red', linewidth=2,
                                   transform=ax.transAxes)
            ax.add_patch(arrow)


def visualize(func: Callable,
              mode: str = 'graph',
              wrt: Union[int, Tuple[int, ...]] = (0,),
              inputs: Optional[Tuple] = None,
              figsize: Tuple[int, int] = (12, 8)):
    """Visualize a function's computation or gradient flow.

    This is the main entry point for Tangent's visualization tools.

    Args:
        func: Function to visualize
        mode: Visualization mode - 'graph' for computation graph,
              'flow' for gradient flow with values
        wrt: Which arguments to compute gradients for (for 'flow' mode)
        inputs: Input values (required for 'flow' mode)
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> def f(x):
        ...     y = x * x
        ...     z = y + x
        ...     return z
        >>>
        >>> # Visualize computation graph
        >>> fig = tangent.visualize(f, mode='graph')
        >>> plt.show()
        >>>
        >>> # Visualize gradient flow with actual values
        >>> fig = tangent.visualize(f, mode='flow', inputs=(3.0,))
        >>> plt.show()
    """
    if mode == 'graph':
        viz = ComputationGraphVisualizer(func, wrt=wrt)
        return viz.visualize(figsize=figsize)

    elif mode == 'flow':
        if inputs is None:
            raise ValueError("inputs argument required for 'flow' mode visualization")
        viz = GradientFlowVisualizer(func, wrt=wrt)
        return viz.visualize_flow(*inputs, figsize=figsize)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'graph' or 'flow'")


def compare_gradients(func: Callable,
                     inputs: Tuple,
                     wrt: Union[int, Tuple[int, ...]] = (0,),
                     eps: float = 1e-7,
                     figsize: Tuple[int, int] = (12, 5)):
    """Compare automatic differentiation gradients with numerical gradients.

    This is useful for debugging and educational purposes.

    Args:
        func: Function to differentiate
        inputs: Input values
        wrt: Which arguments to compute gradients for
        eps: Epsilon for numerical differentiation
        figsize: Figure size

    Returns:
        matplotlib Figure object showing comparison
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    import numpy as np
    from tangent import grad

    # Compute autodiff gradients
    df = grad(func, wrt=wrt)
    if isinstance(wrt, int) or len(wrt) == 1:
        auto_grad = df(*inputs)
        auto_grads = [auto_grad]
    else:
        auto_grads = list(df(*inputs))

    # Compute numerical gradients
    num_grads = []
    wrt_tuple = (wrt,) if isinstance(wrt, int) else wrt

    for idx in wrt_tuple:
        x = inputs[idx]
        if hasattr(x, '__iter__'):
            # Vector input
            x = np.array(x)
            grad_num = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                inputs_plus = list(inputs)
                inputs_minus = list(inputs)
                inputs_plus[idx] = x_plus
                inputs_minus[idx] = x_minus

                grad_num[i] = (func(*inputs_plus) - func(*inputs_minus)) / (2 * eps)
            num_grads.append(grad_num)
        else:
            # Scalar input
            inputs_plus = list(inputs)
            inputs_minus = list(inputs)
            inputs_plus[idx] = x + eps
            inputs_minus[idx] = x - eps
            grad_num = (func(*inputs_plus) - func(*inputs_minus)) / (2 * eps)
            num_grads.append(grad_num)

    # Create comparison plots
    n_grads = len(auto_grads)
    fig, axes = plt.subplots(1, n_grads, figsize=figsize)
    if n_grads == 1:
        axes = [axes]

    fig.suptitle(f'Gradient Comparison: {func.__name__}',
                fontsize=16, fontweight='bold')

    for i, (auto_g, num_g, ax) in enumerate(zip(auto_grads, num_grads, axes)):
        # Convert to numpy if needed
        if hasattr(auto_g, 'numpy'):
            auto_g = auto_g.numpy()
        auto_g = np.atleast_1d(auto_g)
        num_g = np.atleast_1d(num_g)

        # Plot
        x_vals = np.arange(len(auto_g))
        width = 0.35

        ax.bar(x_vals - width/2, auto_g, width, label='Autodiff', alpha=0.8, color='blue')
        ax.bar(x_vals + width/2, num_g, width, label='Numerical', alpha=0.8, color='red')

        ax.set_xlabel('Element', fontsize=12)
        ax.set_ylabel('Gradient Value', fontsize=12)
        ax.set_title(f'∇x[{wrt_tuple[i]}]', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add error text
        error = np.max(np.abs(auto_g - num_g))
        ax.text(0.95, 0.95, f'Max error: {error:.2e}',
               transform=ax.transAxes,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)

    plt.tight_layout()
    return fig


# Convenience function for quick inspection
def show_gradient_code(func: Callable, wrt: Union[int, Tuple[int, ...]] = (0,)):
    """Display the generated gradient code in a formatted way.

    Args:
        func: Function to show gradient code for
        wrt: Which arguments to compute gradients for

    Returns:
        None (prints to stdout)
    """
    from tangent import grad

    df = grad(func, wrt=wrt)

    print("=" * 80)
    print(f"GRADIENT CODE FOR: {func.__name__}")
    print("=" * 80)
    print()

    print("ORIGINAL FUNCTION:")
    print("-" * 80)
    print(inspect.getsource(func))
    print()

    print("GENERATED GRADIENT FUNCTION:")
    print("-" * 80)
    print(inspect.getsource(df))
    print()

    print("=" * 80)
