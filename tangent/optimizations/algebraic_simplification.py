"""
Algebraic Simplification for Tangent gradient expressions.

Uses SymPy to apply mathematical identities and simplify expressions.
Examples:
  - sin(x)^2 + cos(x)^2 -> 1
  - log(exp(x)) -> x
  - x * 1 -> x
  - x + 0 -> x
"""

import ast
import gast
from typing import Dict, Set, Any, Optional
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


class ASTToSymPyConverter:
    """
    Convert Python AST expressions to SymPy symbolic expressions.

    Strategy:
    1. Traverse AST
    2. Build SymPy expression tree
    3. Handle common math operations and functions
    """

    def __init__(self):
        # Track SymPy symbols for variable names
        self.symbols = {}  # name -> sympy.Symbol

    def convert(self, node):
        """
        Convert AST node to SymPy expression.

        Args:
            node: AST node representing an expression

        Returns:
            SymPy expression, or None if conversion fails
        """
        try:
            return self._convert_node(node)
        except Exception as e:
            # Conversion can fail for complex expressions
            # Return None to indicate we can't simplify this expression
            return None

    def _convert_node(self, node):
        """Recursively convert AST node to SymPy."""
        if node is None:
            return None

        # Names/variables
        if isinstance(node, (ast.Name, gast.Name)):
            name = node.id
            if name not in self.symbols:
                self.symbols[name] = sp.Symbol(name)
            return self.symbols[name]

        # Constants
        if isinstance(node, (ast.Constant, gast.Constant)):
            return sp.sympify(node.value)

        # Legacy Num node (Python < 3.8)
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return sp.sympify(node.n)

        # Binary operations
        if isinstance(node, (ast.BinOp, gast.BinOp)):
            left = self._convert_node(node.left)
            right = self._convert_node(node.right)

            if left is None or right is None:
                return None

            op = node.op
            if isinstance(op, (ast.Add, gast.Add)):
                return left + right
            elif isinstance(op, (ast.Sub, gast.Sub)):
                return left - right
            elif isinstance(op, (ast.Mult, gast.Mult)):
                return left * right
            elif isinstance(op, (ast.Div, gast.Div)):
                return left / right
            elif isinstance(op, (ast.Pow, gast.Pow)):
                return left ** right
            elif isinstance(op, (ast.FloorDiv, gast.FloorDiv)):
                return sp.floor(left / right)
            elif isinstance(op, (ast.Mod, gast.Mod)):
                return sp.Mod(left, right)
            else:
                return None

        # Unary operations
        if isinstance(node, (ast.UnaryOp, gast.UnaryOp)):
            operand = self._convert_node(node.operand)
            if operand is None:
                return None

            op = node.op
            if isinstance(op, (ast.UAdd, gast.UAdd)):
                return operand
            elif isinstance(op, (ast.USub, gast.USub)):
                return -operand
            else:
                return None

        # Function calls
        if isinstance(node, (ast.Call, gast.Call)):
            func_name = None
            if isinstance(node.func, (ast.Name, gast.Name)):
                func_name = node.func.id
            elif isinstance(node.func, (ast.Attribute, gast.Attribute)):
                # Handle np.sin, math.cos, etc.
                func_name = node.func.attr

            if not func_name:
                return None

            # Convert arguments
            if len(node.args) == 0:
                return None

            arg = self._convert_node(node.args[0])
            if arg is None:
                return None

            # Map function names to SymPy functions
            func_map = {
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'exp': sp.exp,
                'log': sp.log,
                'sqrt': sp.sqrt,
                'abs': sp.Abs,
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'asin': sp.asin,
                'acos': sp.acos,
                'atan': sp.atan,
            }

            if func_name in func_map:
                return func_map[func_name](arg)
            else:
                return None

        # Power (x**2)
        if isinstance(node, (ast.Pow, gast.Pow)):
            base = self._convert_node(node.left)
            exp = self._convert_node(node.right)
            if base is None or exp is None:
                return None
            return base ** exp

        # Unsupported node type
        return None


class SymPyToASTConverter:
    """
    Convert SymPy expressions back to Python AST.

    Strategy:
    1. Traverse SymPy expression tree
    2. Build equivalent AST
    3. Preserve variable names and structure
    """

    def convert(self, expr):
        """
        Convert SymPy expression to AST node.

        Args:
            expr: SymPy expression

        Returns:
            AST node, or None if conversion fails
        """
        try:
            return self._convert_expr(expr)
        except Exception:
            return None

    def _convert_expr(self, expr):
        """Recursively convert SymPy expression to AST."""
        # Numbers
        if expr.is_Number:
            value = float(expr)
            return gast.Constant(value=value, kind=None)

        # Symbols (variables)
        if expr.is_Symbol:
            return gast.Name(id=str(expr), ctx=gast.Load(),
                           annotation=None, type_comment=None)

        # Addition
        if expr.is_Add:
            args = expr.args
            if len(args) == 0:
                return gast.Constant(value=0, kind=None)

            result = self._convert_expr(args[0])
            for arg in args[1:]:
                right = self._convert_expr(arg)
                result = gast.BinOp(left=result, op=gast.Add(), right=right)
            return result

        # Multiplication
        if expr.is_Mul:
            args = expr.args
            if len(args) == 0:
                return gast.Constant(value=1, kind=None)

            result = self._convert_expr(args[0])
            for arg in args[1:]:
                right = self._convert_expr(arg)
                result = gast.BinOp(left=result, op=gast.Mult(), right=right)
            return result

        # Power
        if expr.is_Pow:
            base = self._convert_expr(expr.args[0])
            exp = self._convert_expr(expr.args[1])
            return gast.BinOp(left=base, op=gast.Pow(), right=exp)

        # Division (represented as Mul with Pow(-1))
        if expr.is_Mul:
            # Check for division pattern: a * b^(-1)
            numer_args = []
            denom_args = []

            for arg in expr.args:
                if arg.is_Pow and arg.args[1] == -1:
                    denom_args.append(arg.args[0])
                else:
                    numer_args.append(arg)

            if denom_args:
                # Build numerator
                if numer_args:
                    numer = self._convert_expr(sp.Mul(*numer_args))
                else:
                    numer = gast.Constant(value=1, kind=None)

                # Build denominator
                if len(denom_args) == 1:
                    denom = self._convert_expr(denom_args[0])
                else:
                    denom = self._convert_expr(sp.Mul(*denom_args))

                return gast.BinOp(left=numer, op=gast.Div(), right=denom)

        # Negation
        if expr.is_Mul and len(expr.args) >= 1 and expr.args[0] == -1:
            # -x pattern
            if len(expr.args) == 2:
                operand = self._convert_expr(expr.args[1])
                return gast.UnaryOp(op=gast.USub(), operand=operand)

        # Functions
        if isinstance(expr, sp.sin):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='sin', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        if isinstance(expr, sp.cos):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='cos', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        if isinstance(expr, sp.exp):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='exp', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        if isinstance(expr, sp.log):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='log', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        if isinstance(expr, sp.sqrt):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='sqrt', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        if isinstance(expr, sp.tan):
            arg = self._convert_expr(expr.args[0])
            return gast.Call(
                func=gast.Name(id='tan', ctx=gast.Load(),
                             annotation=None, type_comment=None),
                args=[arg],
                keywords=[]
            )

        # Fallback: can't convert
        return None


class AlgebraicSimplifier:
    """
    Main algebraic simplification optimizer.

    Usage:
        simplifier = AlgebraicSimplifier()
        optimized_func = simplifier.simplify(gradient_func_ast)
    """

    def __init__(self, aggressive=False):
        """
        Args:
            aggressive: If True, apply more aggressive simplifications
                       that might change numerical behavior slightly
        """
        self.aggressive = aggressive
        self.ast_to_sympy = ASTToSymPyConverter()
        self.sympy_to_ast = SymPyToASTConverter()

        # Statistics
        self.simplifications_applied = 0
        self.expressions_simplified = 0

    def simplify(self, func_ast):
        """
        Apply algebraic simplification to a function.

        Args:
            func_ast: Function AST (from Tangent or after CSE)

        Returns:
            Simplified function AST
        """
        self.simplifications_applied = 0
        self.expressions_simplified = 0

        new_body = []

        for stmt in func_ast.body:
            if isinstance(stmt, (ast.Assign, gast.Assign)):
                # Try to simplify RHS
                simplified_value = self._simplify_expression(stmt.value)

                if simplified_value is not None:
                    # Create new assignment with simplified RHS
                    new_stmt = gast.Assign(
                        targets=stmt.targets,
                        value=simplified_value,
                        type_comment=None
                    )
                    ast.fix_missing_locations(new_stmt)
                    new_body.append(new_stmt)
                    self.expressions_simplified += 1
                else:
                    # Keep original
                    new_body.append(stmt)
            else:
                # Non-assignment statements, keep as-is
                new_body.append(stmt)

        func_ast.body = new_body
        ast.fix_missing_locations(func_ast)
        return func_ast

    def _simplify_expression(self, expr_node):
        """
        Simplify a single expression.

        Args:
            expr_node: AST node representing an expression

        Returns:
            Simplified AST node, or None if simplification failed/not beneficial
        """
        # Convert AST to SymPy
        sympy_expr = self.ast_to_sympy.convert(expr_node)

        if sympy_expr is None:
            # Conversion failed, can't simplify
            return None

        # Apply SymPy simplification
        try:
            # Try multiple simplification strategies
            candidates = []

            # Strategy 1: Basic simplify
            simplified = sp.simplify(sympy_expr)
            candidates.append(simplified)

            # Strategy 2: More aggressive if requested
            if self.aggressive:
                candidates.append(sp.expand(sympy_expr))
                candidates.append(sp.factor(sympy_expr))

            # Strategy 3: Trigonometric simplifications
            candidates.append(sp.trigsimp(sympy_expr))

            # Strategy 4: Logarithm simplifications
            candidates.append(sp.logcombine(sympy_expr, force=True))

            # Choose best candidate based on operation count
            best = sympy_expr
            best_ops = self._count_operations(sympy_expr)

            for candidate in candidates:
                ops = self._count_operations(candidate)
                if ops < best_ops:
                    best = candidate
                    best_ops = ops

            # Check if we improved
            original_ops = self._count_operations(sympy_expr)

            # Also consider string representation length as a tie-breaker
            if best_ops < original_ops or (best_ops == original_ops and
                                           len(str(best)) < len(str(sympy_expr))):
                # Beneficial simplification
                self.simplifications_applied += 1

                # Convert back to AST
                simplified_ast = self.sympy_to_ast.convert(best)

                if simplified_ast is not None:
                    return simplified_ast
        except Exception:
            # Simplification failed
            pass

        # No beneficial simplification
        return None

    def _count_operations(self, expr):
        """
        Count operations in a SymPy expression.

        Used to determine if simplification is beneficial.
        """
        if expr.is_Number or expr.is_Symbol:
            return 0

        count = 1  # This operation

        # Count operations in children
        for arg in expr.args:
            count += self._count_operations(arg)

        return count


def apply_algebraic_simplification(func_ast, config=None):
    """
    Apply algebraic simplification to a function.

    Args:
        func_ast: Function AST to simplify
        config: Configuration dict with 'aggressive' flag

    Returns:
        Simplified AST
    """
    config = config or {}
    aggressive = config.get('aggressive', False)

    simplifier = AlgebraicSimplifier(aggressive=aggressive)
    return simplifier.simplify(func_ast)
