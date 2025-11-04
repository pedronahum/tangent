"""
Common Subexpression Elimination for Tangent symbolic expressions.

Identifies and eliminates redundant computations in generated gradient code.
"""

import ast
import gast
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import hashlib


class SubexpressionAnalyzer:
    """
    Analyze expressions to find common subexpressions.

    Strategy:
    1. Traverse expression AST
    2. Hash each subexpression for quick comparison
    3. Count occurrences of each unique subexpression
    4. Track locations for replacement
    """

    def __init__(self, min_occurrences=2, min_cost=2):
        self.min_occurrences = min_occurrences
        self.min_cost = min_cost  # Minimum computational cost to be worth CSE

        # Map: expression_hash -> (ast_node, cost, occurrences, locations)
        self.subexpressions = {}

        # Cache for expression hashes
        self.expr_hash_cache = {}

    def analyze(self, expr_ast):
        """
        Analyze expression and return CSE candidates.

        Returns:
            List of (subexpression_ast, cost, count, locations)
        """
        self.subexpressions = {}
        self.expr_hash_cache = {}

        # Walk the AST and collect subexpressions
        self._collect_subexpressions(expr_ast, [])

        # Filter candidates
        candidates = []
        for expr_hash, (node, cost, count, locations) in self.subexpressions.items():
            if count >= self.min_occurrences and cost >= self.min_cost:
                candidates.append((node, cost, count, locations))

        # Sort by benefit (cost * count) descending
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

        return candidates

    def _collect_subexpressions(self, node, path):
        """Recursively collect subexpressions."""
        if node is None:
            return

        # Skip leaf nodes (names, constants)
        # Note: ast.Num deprecated in Python 3.8+, use ast.Constant
        if isinstance(node, (ast.Name, gast.Name, ast.Constant, gast.Constant)):
            return
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return

        # Compute hash for this subexpression
        expr_hash = self._hash_expression(node)

        # Compute cost
        cost = self._compute_cost(node)

        # Record this subexpression
        if expr_hash in self.subexpressions:
            # Already seen, increment count
            existing_node, existing_cost, count, locations = self.subexpressions[expr_hash]
            self.subexpressions[expr_hash] = (existing_node, existing_cost, count + 1,
                                             locations + [path[:]])
        else:
            # New subexpression
            self.subexpressions[expr_hash] = (node, cost, 1, [path[:]])

        # Recurse on children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, (ast.AST, gast.AST)):
                        self._collect_subexpressions(item, path + [(field, i)])
            elif isinstance(value, (ast.AST, gast.AST)):
                self._collect_subexpressions(value, path + [(field, None)])

    def _hash_expression(self, node):
        """
        Create a hash for an expression.

        Two expressions are the same if they have the same structure
        and use the same variables/constants.
        """
        if id(node) in self.expr_hash_cache:
            return self.expr_hash_cache[id(node)]

        # Convert AST to canonical string representation
        try:
            if hasattr(gast, 'unparse'):
                expr_str = gast.unparse(node)
            elif hasattr(ast, 'unparse'):
                expr_str = ast.unparse(node)
            else:
                # Fallback: use ast.dump
                expr_str = ast.dump(node)
        except Exception:
            expr_str = ast.dump(node)

        # Hash the string
        expr_hash = hashlib.md5(expr_str.encode()).hexdigest()
        self.expr_hash_cache[id(node)] = expr_hash

        return expr_hash

    def _compute_cost(self, node):
        """
        Estimate computational cost of an expression.

        Cost heuristics:
        - Add/Sub: 1
        - Mult: 2
        - Div/Mod: 5
        - Power: 10
        - Function call (sin, cos, exp, etc.): 20
        """
        # Note: ast.Num deprecated in Python 3.8+, use ast.Constant
        if isinstance(node, (ast.Name, gast.Name, ast.Constant, gast.Constant)):
            return 0  # Free (just a load)
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return 0

        cost = 0

        # Operation costs
        if isinstance(node, (ast.BinOp, gast.BinOp)):
            op = node.op
            if isinstance(op, (ast.Add, gast.Add, ast.Sub, gast.Sub)):
                cost = 1
            elif isinstance(op, (ast.Mult, gast.Mult)):
                cost = 2
            elif isinstance(op, (ast.Div, gast.Div, ast.FloorDiv, gast.FloorDiv,
                              ast.Mod, gast.Mod)):
                cost = 5
            elif isinstance(op, (ast.Pow, gast.Pow)):
                cost = 10
            else:
                cost = 2  # Default

            # Add costs of operands
            cost += self._compute_cost(node.left)
            cost += self._compute_cost(node.right)

        elif isinstance(node, (ast.UnaryOp, gast.UnaryOp)):
            cost = 1 + self._compute_cost(node.operand)

        elif isinstance(node, (ast.Call, gast.Call)):
            # Function calls are expensive
            func_name = ''
            if isinstance(node.func, (ast.Name, gast.Name)):
                func_name = node.func.id

            # Different costs for different functions
            expensive_funcs = {'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'pow'}
            if func_name in expensive_funcs:
                cost = 20
            else:
                cost = 10  # Generic function call

            # Add argument costs
            for arg in node.args:
                cost += self._compute_cost(arg)

        else:
            # Recurse on children for other node types
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, (ast.AST, gast.AST)):
                            cost += self._compute_cost(item)
                elif isinstance(value, (ast.AST, gast.AST)):
                    cost += self._compute_cost(value)

        return cost


class CSETransformer(ast.NodeTransformer):
    """
    Transform AST to use temporary variables for common subexpressions.
    """

    def __init__(self, cse_map: Dict[str, str]):
        """
        Args:
            cse_map: Map from expression_hash to temporary variable name
        """
        self.cse_map = cse_map
        self.expr_to_hash = {}  # Cache: expr_id -> hash

    def visit(self, node):
        """Visit node and replace with temp var if it's a CSE candidate."""
        # Compute hash
        expr_hash = self._hash_expr(node)

        # Check if this is a common subexpression
        if expr_hash in self.cse_map:
            # Replace with temporary variable
            temp_var_name = self.cse_map[expr_hash]
            return gast.Name(id=temp_var_name, ctx=gast.Load(), annotation=None, type_comment=None)

        # Otherwise, continue normal traversal
        return self.generic_visit(node)

    def _hash_expr(self, node):
        """Hash expression (same as SubexpressionAnalyzer)."""
        if id(node) in self.expr_to_hash:
            return self.expr_to_hash[id(node)]

        try:
            if hasattr(gast, 'unparse'):
                expr_str = gast.unparse(node)
            elif hasattr(ast, 'unparse'):
                expr_str = ast.unparse(node)
            else:
                expr_str = ast.dump(node)
        except Exception:
            expr_str = ast.dump(node)

        expr_hash = hashlib.md5(expr_str.encode()).hexdigest()
        self.expr_to_hash[id(node)] = expr_hash
        return expr_hash


class CommonSubexpressionEliminator:
    """
    Main CSE optimizer for Tangent.

    Usage:
        eliminator = CommonSubexpressionEliminator()
        optimized_func = eliminator.optimize(grad_func_ast)
    """

    def __init__(self, min_occurrences=2, min_cost=2):
        self.analyzer = SubexpressionAnalyzer(min_occurrences, min_cost)
        self.temp_var_counter = 0

    def optimize(self, func_ast):
        """
        Apply CSE to a function.

        Args:
            func_ast: Function AST (from Tangent gradient generation)

        Returns:
            Optimized function AST with CSE applied
        """
        # First, perform global analysis across all statements
        global_candidates = self._global_analysis(func_ast)

        if global_candidates:
            # Apply global CSE
            return self._apply_global_cse(func_ast, global_candidates)
        else:
            # Fall back to per-statement CSE
            return self._optimize_per_statement(func_ast)

    def _global_analysis(self, func_ast):
        """
        Analyze all RHS expressions in the function to find common subexpressions
        that appear across multiple statements.

        Returns:
            List of (expression_hash, ast_node, cost, count, statement_indices)
        """
        # Collect all RHS expressions
        expr_map = {}  # expr_hash -> (node, cost, [stmt_indices])

        for i, stmt in enumerate(func_ast.body):
            if isinstance(stmt, (ast.Assign, gast.Assign)):
                # Collect all subexpressions from this RHS
                self._collect_from_expression(stmt.value, i, expr_map)

        # Filter candidates that appear multiple times
        candidates = []
        for expr_hash, (node, cost, stmt_indices) in expr_map.items():
            count = len(stmt_indices)
            if count >= self.analyzer.min_occurrences and cost >= self.analyzer.min_cost:
                candidates.append((expr_hash, node, cost, count, stmt_indices))

        # Sort by benefit (cost * count)
        candidates.sort(key=lambda x: x[2] * x[3], reverse=True)

        return candidates

    def _collect_from_expression(self, node, stmt_index, expr_map):
        """
        Recursively collect all subexpressions from an expression tree.
        """
        if node is None:
            return

        # Skip leaf nodes
        if isinstance(node, (ast.Name, gast.Name, ast.Constant, gast.Constant)):
            return
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return

        # Hash this subexpression
        try:
            if hasattr(gast, 'unparse'):
                expr_str = gast.unparse(node)
            elif hasattr(ast, 'unparse'):
                expr_str = ast.unparse(node)
            else:
                expr_str = ast.dump(node)
        except Exception:
            expr_str = ast.dump(node)

        expr_hash = hashlib.md5(expr_str.encode()).hexdigest()

        # Compute cost
        cost = self.analyzer._compute_cost(node)

        # Record this expression
        if expr_hash in expr_map:
            existing_node, existing_cost, stmt_indices = expr_map[expr_hash]
            # Add this statement index if not already present
            if stmt_index not in stmt_indices:
                stmt_indices.append(stmt_index)
        else:
            expr_map[expr_hash] = (node, cost, [stmt_index])

        # Recurse on children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (ast.AST, gast.AST)):
                        self._collect_from_expression(item, stmt_index, expr_map)
            elif isinstance(value, (ast.AST, gast.AST)):
                self._collect_from_expression(value, stmt_index, expr_map)

    def _get_used_vars(self, node):
        """Get all variable names used in an expression."""
        vars_used = set()
        for child in gast.walk(node):
            if isinstance(child, (gast.Name, ast.Name)) and isinstance(child.ctx, (gast.Load, ast.Load)):
                vars_used.add(child.id)
        return vars_used

    def _apply_global_cse(self, func_ast, candidates):
        """
        Apply global CSE by creating temporary variables at appropriate locations
        (after dependencies are satisfied) and replacing expressions.

        IMPORTANT: We must respect data dependencies - only place a CSE temp
        after all its required variables are defined.
        """
        # Build a map of where each variable is defined
        var_def_positions = {}  # var_name -> statement_index
        for i, stmt in enumerate(func_ast.body):
            if isinstance(stmt, (gast.Assign, ast.Assign)):
                for target in stmt.targets:
                    if isinstance(target, (gast.Name, ast.Name)):
                        var_def_positions[target.id] = i

        # For each candidate, determine the earliest safe position
        safe_candidates = []
        for expr_hash, node, cost, count, stmt_indices in candidates:
            # Get all variables used in this expression
            vars_used = self._get_used_vars(node)

            # Find the latest definition of any used variable
            latest_def = -1
            all_vars_defined = True
            for var in vars_used:
                if var in var_def_positions:
                    latest_def = max(latest_def, var_def_positions[var])
                else:
                    # Variable not defined in this function (might be parameter)
                    # This is OK - parameters are available from the start
                    pass

            # Earliest statement that uses this expression
            first_use = min(stmt_indices)

            # CSE temp must be placed after all dependencies and before first use
            if latest_def < first_use:
                # Safe to create CSE temp
                safe_pos = latest_def + 1  # Place right after last dependency
                safe_candidates.append((expr_hash, node, safe_pos, stmt_indices))

        if not safe_candidates:
            # No safe CSE opportunities, keep function as-is
            return func_ast

        # Sort by position where temp should be inserted
        safe_candidates.sort(key=lambda x: x[2])

        # Create CSE map and insert temps at appropriate positions
        cse_map = {}  # expr_hash -> temp_var_name
        temp_insertions = {}  # position -> [temp_assignments]

        for expr_hash, node, safe_pos, stmt_indices in safe_candidates:
            temp_var_name = f'_cse_temp_{self.temp_var_counter}'
            self.temp_var_counter += 1

            # Create assignment: temp_var = subexpression
            temp_assign = gast.Assign(
                targets=[gast.Name(id=temp_var_name, ctx=gast.Store(),
                                 annotation=None, type_comment=None)],
                value=node,
                type_comment=None
            )
            ast.fix_missing_locations(temp_assign)

            # Record where to insert this temp
            if safe_pos not in temp_insertions:
                temp_insertions[safe_pos] = []
            temp_insertions[safe_pos].append(temp_assign)

            # Map expression to temp var
            cse_map[expr_hash] = temp_var_name

        # Transform all statements and insert temps at appropriate positions
        transformer = CSETransformer(cse_map)
        new_body = []

        for i, stmt in enumerate(func_ast.body):
            # Insert any temps that should go before this statement
            if i in temp_insertions:
                new_body.extend(temp_insertions[i])

            # Transform and add the original statement
            new_stmt = transformer.visit(stmt)
            ast.fix_missing_locations(new_stmt)
            new_body.append(new_stmt)

        func_ast.body = new_body
        ast.fix_missing_locations(func_ast)
        return func_ast

    def _optimize_per_statement(self, func_ast):
        """
        Original per-statement CSE (fallback when no global opportunities found).
        """
        new_body = []

        for stmt in func_ast.body:
            if isinstance(stmt, (ast.Assign, gast.Assign)):
                # Analyze RHS for common subexpressions
                candidates = self.analyzer.analyze(stmt.value)

                if candidates:
                    # Create temporary variables for CSE candidates
                    cse_map = {}  # expr_hash -> temp_var_name
                    cse_assignments = []

                    for node, cost, count, locations in candidates:
                        temp_var_name = f'_cse_temp_{self.temp_var_counter}'
                        self.temp_var_counter += 1

                        # Create assignment: temp_var = subexpression
                        temp_assign = gast.Assign(
                            targets=[gast.Name(id=temp_var_name, ctx=gast.Store(),
                                             annotation=None, type_comment=None)],
                            value=node,
                            type_comment=None
                        )
                        cse_assignments.append(temp_assign)

                        # Map expression to temp var
                        expr_hash = self.analyzer._hash_expression(node)
                        cse_map[expr_hash] = temp_var_name

                    # Transform original statement to use temp vars
                    transformer = CSETransformer(cse_map)
                    new_stmt = transformer.visit(stmt)

                    # Add CSE assignments before this statement
                    new_body.extend(cse_assignments)
                    new_body.append(new_stmt)
                else:
                    # No CSE opportunities, keep as-is
                    new_body.append(stmt)
            else:
                # Not an assignment, keep as-is
                new_body.append(stmt)

        func_ast.body = new_body
        return func_ast


def apply_cse(func_ast, config=None):
    """
    Apply Common Subexpression Elimination.

    Args:
        func_ast: Function AST to optimize
        config: Configuration dict with 'min_occurrences', 'min_cost'

    Returns:
        Optimized AST
    """
    config = config or {}
    min_occurrences = config.get('min_occurrences', 2)
    min_cost = config.get('min_cost', 2)

    eliminator = CommonSubexpressionEliminator(min_occurrences, min_cost)
    return eliminator.optimize(func_ast)
