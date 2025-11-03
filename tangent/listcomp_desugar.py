# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Desugar list comprehensions into explicit for loops.

List comprehensions are syntactic sugar that Tangent can't differentiate directly.
This module transforms them into equivalent for loops that can be handled by the
existing AD machinery.

Example transformation:
    result = [x * i for i in range(3) if i > 0]

becomes:
    result = []
    for i in range(3):
        if i > 0:
            result.append(x * i)
"""

from __future__ import absolute_import

import gast

from tangent import quoting
from tangent import template


class ListCompDesugarer(gast.NodeTransformer):
    """Transform list comprehensions into explicit for loops."""

    def __init__(self):
        self.temp_counter = 0

    def _get_temp_name(self):
        """Generate a unique temporary variable name."""
        name = f'_listcomp_{self.temp_counter}'
        self.temp_counter += 1
        return name

    def visit_FunctionDef(self, node):
        """Visit function and transform body statements."""
        # Process the function body
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, list):
                # If a statement was expanded into multiple statements
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)

        node.body = new_body
        return node

    def visit_For(self, node):
        """Visit for loop and transform body."""
        # First visit iter and target
        node.iter = self.visit(node.iter)
        node.target = self.visit(node.target)

        # Transform body
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)
        node.body = new_body

        # Transform orelse if present
        if node.orelse:
            new_orelse = []
            for stmt in node.orelse:
                result = self.visit(stmt)
                if isinstance(result, list):
                    new_orelse.extend(result)
                elif result is not None:
                    new_orelse.append(result)
            node.orelse = new_orelse

        return node

    def visit_If(self, node):
        """Visit if statement and transform body."""
        # Visit test
        node.test = self.visit(node.test)

        # Transform body
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)
        node.body = new_body

        # Transform orelse if present
        if node.orelse:
            new_orelse = []
            for stmt in node.orelse:
                result = self.visit(stmt)
                if isinstance(result, list):
                    new_orelse.extend(result)
                elif result is not None:
                    new_orelse.append(result)
            node.orelse = new_orelse

        return node

    def visit_While(self, node):
        """Visit while loop and transform body."""
        node.test = self.visit(node.test)

        # Transform body
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)
        node.body = new_body

        # Transform orelse if present
        if node.orelse:
            new_orelse = []
            for stmt in node.orelse:
                result = self.visit(stmt)
                if isinstance(result, list):
                    new_orelse.extend(result)
                elif result is not None:
                    new_orelse.append(result)
            node.orelse = new_orelse

        return node

    def visit_Assign(self, node):
        """Transform assignments containing list comprehensions."""
        # Visit RHS (may contain nested list comps)
        node.value = self.visit(node.value)

        # Check if RHS is a ListComp
        if isinstance(node.value, gast.ListComp):
            return self._desugar_listcomp_assign(node)

        return node

    def visit_Expr(self, node):
        """Handle standalone list comprehensions (though rare)."""
        node.value = self.visit(node.value)

        if isinstance(node.value, gast.ListComp):
            # Standalone list comp - just desugar it
            temp_var = self._get_temp_name()
            stmts = self._desugar_listcomp(node.value, temp_var)
            return stmts

        return node

    def _desugar_listcomp_assign(self, assign_node):
        """Transform: target = [expr for ...] into for loop."""
        listcomp = assign_node.value
        target = assign_node.targets[0]

        # Get the target variable name
        if isinstance(target, gast.Name):
            result_var = target.id
        else:
            # For complex targets like subscripts, use a temp variable
            result_var = self._get_temp_name()

        # Generate the for loop statements
        stmts = self._desugar_listcomp(listcomp, result_var)

        # If we used a temp variable, add final assignment
        if not isinstance(target, gast.Name) or target.id != result_var:
            final_assign = gast.Assign(
                targets=[target],
                value=gast.Name(id=result_var, ctx=gast.Load(), annotation=None))
            stmts.append(final_assign)

        return stmts

    def _desugar_listcomp(self, listcomp, result_var):
        """
        Transform a ListComp into a sequence of statements.

        Args:
            listcomp: The gast.ListComp node
            result_var: Name of the variable to store the result

        Returns:
            List of statement nodes (initialization + for loops)
        """
        # Initialize the result list: result_var = []
        init_list = gast.Assign(
            targets=[gast.Name(id=result_var, ctx=gast.Store(), annotation=None)],
            value=gast.List(elts=[], ctx=gast.Load()))

        # Build the for loop(s) from the generators
        # A list comp can have multiple generators: [x+y for x in a for y in b]
        loop_body = self._build_append_stmt(listcomp.elt, result_var)

        # Build nested for loops from the generators (innermost first)
        for generator in reversed(listcomp.generators):
            # Start with the body (append statement or existing loop)
            body = loop_body if not isinstance(loop_body, list) else loop_body

            # Add any filter conditions (if clauses)
            for if_clause in reversed(generator.ifs):
                body = [gast.If(test=if_clause, body=[body] if not isinstance(body, list) else body, orelse=[])]
                body = body[0]  # Unwrap from list

            # Create the for loop
            for_loop = gast.For(
                target=generator.target,
                iter=generator.iter,
                body=[body] if not isinstance(body, list) else body,
                orelse=[])

            loop_body = for_loop

        return [init_list, loop_body]

    def _build_append_stmt(self, expr, result_var):
        """Build: temp = expr; result_var.append(temp)

        We create a temporary variable to ensure the expression is differentiated.
        If we did result_var.append(expr) directly, the expr would be hidden inside
        a non-differentiable append() call.
        """
        # Create temporary variable for the element
        temp_var = self._get_temp_name()

        # Assignment: temp = expr
        temp_assign = gast.Assign(
            targets=[gast.Name(id=temp_var, ctx=gast.Store(), annotation=None)],
            value=expr)

        # Append call: result_var.append(temp)
        append_call = gast.Expr(value=gast.Call(
            func=gast.Attribute(
                value=gast.Name(id=result_var, ctx=gast.Load(), annotation=None),
                attr='append',
                ctx=gast.Load()),
            args=[gast.Name(id=temp_var, ctx=gast.Load(), annotation=None)],
            keywords=[]))

        # Return both statements
        return [temp_assign, append_call]


def desugar_listcomps(node):
    """
    Desugar all list comprehensions in the given AST.

    Args:
        node: A gast.FunctionDef node

    Returns:
        The transformed AST with list comprehensions replaced by for loops
    """
    desugarer = ListCompDesugarer()
    return desugarer.visit(node)
