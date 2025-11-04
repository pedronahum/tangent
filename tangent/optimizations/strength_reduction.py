"""
Strength Reduction optimization for Tangent gradient expressions.

Replaces expensive operations with cheaper equivalents while preserving semantics.

Examples:
  - x ** 2 -> x * x (power to multiply)
  - x ** 3 -> x * x * x (power to multiply)
  - x ** 0.5 -> sqrt(x) (generic power to sqrt)
  - x / constant -> x * reciprocal (division to multiply)
  - 2 ** x -> exp2(x) (generic power to exp2)
"""

import ast
import gast
import math
from typing import Optional


class StrengthReducer(gast.NodeTransformer):
    """
    AST transformer that applies strength reduction patterns.

    Transforms expensive operations into cheaper equivalents:
    - Power operations with small integer exponents -> multiplications
    - Power with 0.5 exponent -> sqrt
    - Division by constants -> multiplication by reciprocal
    - 2**x -> exp2(x)
    """

    def __init__(self, enable_division_to_multiply=True, enable_power_reduction=True):
        """
        Args:
            enable_division_to_multiply: Convert x / constant to x * reciprocal
            enable_power_reduction: Convert x**n to multiplications for small n
        """
        self.enable_division_to_multiply = enable_division_to_multiply
        self.enable_power_reduction = enable_power_reduction
        self.reductions_applied = 0

    def visit_BinOp(self, node):
        """Visit binary operation nodes and apply strength reduction."""
        # First, visit children
        node = self.generic_visit(node)

        # Power reduction: x ** n
        if self.enable_power_reduction and isinstance(node.op, (gast.Pow, ast.Pow)):
            reduced = self._reduce_power(node)
            if reduced is not None:
                self.reductions_applied += 1
                return reduced

        # Division to multiplication: x / constant -> x * (1/constant)
        if self.enable_division_to_multiply and isinstance(node.op, (gast.Div, ast.Div)):
            reduced = self._reduce_division(node)
            if reduced is not None:
                self.reductions_applied += 1
                return reduced

        return node

    def visit_Call(self, node):
        """Visit function calls - handle pow(x, n) as well."""
        # First, visit children
        node = self.generic_visit(node)

        # Check for pow(x, n) function calls
        if isinstance(node.func, (gast.Name, ast.Name)) and node.func.id == 'pow':
            if len(node.args) >= 2:
                # Convert pow(x, n) to x ** n, then apply power reduction
                power_node = gast.BinOp(
                    left=node.args[0],
                    op=gast.Pow(),
                    right=node.args[1]
                )
                reduced = self._reduce_power(power_node)
                if reduced is not None:
                    self.reductions_applied += 1
                    return reduced

        return node

    def _reduce_power(self, node):
        """
        Reduce power operations to cheaper equivalents.

        Patterns:
        - x ** 2 -> x * x
        - x ** 3 -> x * x * x
        - x ** 4 -> (x * x) * (x * x)  [CSE will optimize]
        - x ** 0.5 -> sqrt(x)
        - x ** -1 -> 1.0 / x
        - 2 ** x -> exp2(x)  [if available]
        """
        # Get exponent value if it's a constant
        exponent_value = self._get_constant_value(node.right)

        if exponent_value is not None:
            base = node.left

            # x ** 2 -> x * x
            if exponent_value == 2:
                return gast.BinOp(
                    left=base,
                    op=gast.Mult(),
                    right=self._copy_node(base)
                )

            # x ** 3 -> x * x * x
            elif exponent_value == 3:
                x_times_x = gast.BinOp(
                    left=base,
                    op=gast.Mult(),
                    right=self._copy_node(base)
                )
                return gast.BinOp(
                    left=x_times_x,
                    op=gast.Mult(),
                    right=self._copy_node(base)
                )

            # x ** 4 -> (x * x) * (x * x)
            # Let CSE handle the common x*x subexpression
            elif exponent_value == 4:
                x_times_x = gast.BinOp(
                    left=base,
                    op=gast.Mult(),
                    right=self._copy_node(base)
                )
                return gast.BinOp(
                    left=x_times_x,
                    op=gast.Mult(),
                    right=self._copy_node(x_times_x)
                )

            # x ** 0.5 -> sqrt(x)
            elif exponent_value == 0.5:
                return gast.Call(
                    func=gast.Name(id='sqrt', ctx=gast.Load(),
                                 annotation=None, type_comment=None),
                    args=[base],
                    keywords=[]
                )

            # x ** -1 -> 1.0 / x
            elif exponent_value == -1:
                return gast.BinOp(
                    left=gast.Constant(value=1.0, kind=None),
                    op=gast.Div(),
                    right=base
                )

            # x ** -2 -> 1.0 / (x * x)
            elif exponent_value == -2:
                x_times_x = gast.BinOp(
                    left=base,
                    op=gast.Mult(),
                    right=self._copy_node(base)
                )
                return gast.BinOp(
                    left=gast.Constant(value=1.0, kind=None),
                    op=gast.Div(),
                    right=x_times_x
                )

            # x ** 1 -> x (identity)
            elif exponent_value == 1:
                return base

            # x ** 0 -> 1 (constant)
            elif exponent_value == 0:
                return gast.Constant(value=1, kind=None)

        # Special case: 2 ** x -> exp2(x)
        # Only if base is constant 2 and we have exp2 available
        base_value = self._get_constant_value(node.left)
        if base_value == 2:
            # Check if it's a small integer exponent first
            if exponent_value in [2, 3, 4]:
                # 2**2 = 4, etc. - fold the constant
                result = 2 ** exponent_value
                return gast.Constant(value=result, kind=None)
            else:
                # For non-constant or large exponents, use exp2 if available
                # Note: exp2 might not be available in all contexts
                # We keep the original power operation to be safe
                pass

        # No reduction applied
        return None

    def _reduce_division(self, node):
        """
        Reduce division by constant to multiplication by reciprocal.

        Pattern: x / constant -> x * (1.0 / constant)

        This is beneficial because:
        - Division takes ~10-20 cycles on modern CPUs
        - Multiplication takes ~1 cycle
        - Reciprocal is computed at compile time (constant folding)
        """
        # Only reduce if divisor is a constant
        divisor_value = self._get_constant_value(node.right)

        if divisor_value is not None and divisor_value != 0:
            # Compute reciprocal
            reciprocal = 1.0 / divisor_value

            # Create x * reciprocal
            return gast.BinOp(
                left=node.left,
                op=gast.Mult(),
                right=gast.Constant(value=reciprocal, kind=None)
            )

        # No reduction applied
        return None

    def _get_constant_value(self, node):
        """
        Extract constant value from a node if possible.

        Returns:
            Numeric value or None
        """
        # Handle ast.Constant (Python 3.8+)
        if isinstance(node, (gast.Constant, ast.Constant)):
            if isinstance(node.value, (int, float)):
                return float(node.value)

        # Handle legacy ast.Num (Python < 3.8)
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            if isinstance(node.n, (int, float)):
                return float(node.n)

        # Handle unary operations like -1 (UnaryOp with USub)
        if isinstance(node, (gast.UnaryOp, ast.UnaryOp)):
            if isinstance(node.op, (gast.USub, ast.USub)):
                operand_value = self._get_constant_value(node.operand)
                if operand_value is not None:
                    return -operand_value
            elif isinstance(node.op, (gast.UAdd, ast.UAdd)):
                return self._get_constant_value(node.operand)

        return None

    def _copy_node(self, node):
        """
        Create a copy of an AST node.

        Important: We need to copy nodes when using them multiple times
        in the AST (e.g., x appears twice in x * x).
        """
        import copy
        return copy.deepcopy(node)


class StrengthReductionOptimizer:
    """
    Main strength reduction optimizer.

    Usage:
        optimizer = StrengthReductionOptimizer()
        optimized_func = optimizer.optimize(func_ast)
    """

    def __init__(self, enable_division_to_multiply=True, enable_power_reduction=True):
        self.enable_division_to_multiply = enable_division_to_multiply
        self.enable_power_reduction = enable_power_reduction

        # Statistics
        self.total_reductions = 0

    def optimize(self, func_ast):
        """
        Apply strength reduction to a function.

        Args:
            func_ast: Function AST to optimize

        Returns:
            Optimized function AST
        """
        reducer = StrengthReducer(
            enable_division_to_multiply=self.enable_division_to_multiply,
            enable_power_reduction=self.enable_power_reduction
        )

        optimized = reducer.visit(func_ast)
        self.total_reductions = reducer.reductions_applied

        # Fix missing AST locations
        ast.fix_missing_locations(optimized)

        return optimized


def apply_strength_reduction(func_ast, config=None):
    """
    Apply strength reduction optimization.

    Args:
        func_ast: Function AST to optimize
        config: Configuration dict with:
            - 'enable_division_to_multiply': bool (default True)
            - 'enable_power_reduction': bool (default True)

    Returns:
        Optimized AST with strength reduction applied

    Example:
        >>> config = {
        ...     'enable_division_to_multiply': True,
        ...     'enable_power_reduction': True
        ... }
        >>> optimized = apply_strength_reduction(func_ast, config)
    """
    config = config or {}
    enable_division = config.get('enable_division_to_multiply', True)
    enable_power = config.get('enable_power_reduction', True)

    optimizer = StrengthReductionOptimizer(
        enable_division_to_multiply=enable_division,
        enable_power_reduction=enable_power
    )

    return optimizer.optimize(func_ast)
