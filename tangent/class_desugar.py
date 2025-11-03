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
"""Class method inlining for automatic differentiation.

This module transforms class method calls into inlined code, allowing Tangent
to differentiate through user-defined classes without needing OOP support.

Strategy:
---------
1. Access class definitions from the function's __globals__ namespace
2. Track instance variable assignments (e.g., calc = Calculator())
3. When encountering method calls (e.g., calc.square(x)), inline the method body
4. Substitute 'self' and method parameters with actual values

Example:
--------
Before transformation:
    class Calculator:
        def square(self, x):
            return x ** 2

    def f(x):
        calc = Calculator()
        return calc.square(x)

After transformation:
    def f(x):
        # calc = Calculator() removed
        return x ** 2  # Method body inlined
"""
from __future__ import absolute_import

import copy
import gast
import inspect
import textwrap
import types


class ClassMethodInliner(gast.NodeTransformer):
    """Inlines class method calls by substituting method bodies.

    This transformer:
    1. Resolves class objects from func.__globals__
    2. Tracks instance variable assignments
    3. Inlines method calls by parsing and substituting method bodies
    4. Handles 'self' parameter substitution
    """

    def __init__(self, func):
        """Initialize the class method inliner.

        Args:
            func: The function being transformed (used to access __globals__)
        """
        self.func = func

        # Map variable name -> class info
        # e.g., 'calc' -> {'class': Calculator, 'init_args': [...]}
        self.instance_vars = {}

        # Map variable name -> instance attributes
        # e.g., 'scaler' -> {'factor': gast.Constant(2.5)}
        self.instance_attrs = {}

    def visit_Assign(self, node):
        """Track instance variable assignments (e.g., calc = Calculator()).

        Args:
            node: Assign AST node

        Returns:
            Modified node or None (if we remove the assignment)
        """
        # Check for pattern: var = ClassName(args)
        if (isinstance(node.value, gast.Call) and
            isinstance(node.value.func, gast.Name) and
            len(node.targets) == 1 and
            isinstance(node.targets[0], gast.Name)):

            class_name = node.value.func.id
            var_name = node.targets[0].id

            # Try to resolve the class from function's globals
            if class_name in self.func.__globals__:
                potential_class = self.func.__globals__[class_name]

                # Check if it's actually a class
                if inspect.isclass(potential_class):
                    # Store instance variable mapping
                    self.instance_vars[var_name] = {
                        'class': potential_class,
                        'init_args': node.value.args,
                        'init_keywords': node.value.keywords
                    }

                    # Parse __init__ to extract instance attributes
                    if hasattr(potential_class, '__init__'):
                        self._extract_instance_attrs(var_name, potential_class,
                                                     node.value.args,
                                                     node.value.keywords)

                    # Remove the instantiation - we'll inline methods instead
                    return None

        # Not an instance creation, visit normally
        self.generic_visit(node)
        return node

    def _extract_instance_attrs(self, var_name, class_obj, init_args, init_keywords):
        """Extract instance attributes from __init__ method.

        Args:
            var_name: Name of the instance variable
            class_obj: The class object
            init_args: Arguments passed to __init__
            init_keywords: Keyword arguments passed to __init__
        """
        try:
            # Extract attributes from entire class hierarchy
            attrs = self._extract_attrs_from_hierarchy(class_obj, init_args, init_keywords)
            self.instance_attrs[var_name] = attrs

        except (TypeError, OSError, IOError):
            # Can't get source or parse it - that's okay, continue without attrs
            self.instance_attrs[var_name] = {}

    def _extract_attrs_from_hierarchy(self, class_obj, init_args, init_keywords):
        """Extract attributes from class and its parents.

        Args:
            class_obj: The class object
            init_args: Arguments passed to __init__
            init_keywords: Keyword arguments passed to __init__

        Returns:
            Dictionary of attribute name -> value AST nodes
        """
        # Build parameter map for current class __init__
        init_method = class_obj.__init__
        init_source = inspect.getsource(init_method)
        init_source = textwrap.dedent(init_source)
        init_ast = gast.parse(init_source).body[0]

        params = init_ast.args.args[1:]  # Skip 'self'
        param_map = {}

        # Map positional arguments
        for i, param in enumerate(params):
            if i < len(init_args):
                param_map[param.id] = init_args[i]

        # Map keyword arguments
        for keyword in init_keywords:
            param_map[keyword.arg] = keyword.value

        # Collect attributes, starting with parent class attributes
        attrs = {}

        # Check for super().__init__() calls to process parent attributes
        for stmt in init_ast.body:
            parent_attrs = self._process_super_init_call(stmt, class_obj, param_map)
            if parent_attrs:
                attrs.update(parent_attrs)

        # Now process current class's attribute assignments
        for stmt in init_ast.body:
            if (isinstance(stmt, gast.Assign) and
                len(stmt.targets) == 1 and
                isinstance(stmt.targets[0], gast.Attribute) and
                isinstance(stmt.targets[0].value, gast.Name) and
                stmt.targets[0].value.id == 'self'):

                attr_name = stmt.targets[0].attr

                # Substitute parameters in the RHS
                attr_value = self._substitute_params(stmt.value, param_map)
                attrs[attr_name] = attr_value

        return attrs

    def _process_super_init_call(self, stmt, class_obj, param_map):
        """Process super().__init__() calls to extract parent attributes.

        Args:
            stmt: AST statement to check
            class_obj: Current class object
            param_map: Parameter substitution map for current __init__

        Returns:
            Dictionary of parent attributes, or empty dict if not a super() call
        """
        # Check for super().__init__(...) pattern
        # This can appear as either:
        # 1. Expr node containing the call: super().__init__(args)
        # 2. Assign node: result = super().__init__(args) (rare but possible)

        call_node = None

        if isinstance(stmt, gast.Expr) and isinstance(stmt.value, gast.Call):
            call_node = stmt.value
        elif (isinstance(stmt, gast.Assign) and
              isinstance(stmt.value, gast.Call)):
            call_node = stmt.value

        if call_node is None:
            return {}

        # Check if this is super().__init__(...)
        if not (isinstance(call_node.func, gast.Attribute) and
                call_node.func.attr == '__init__'):
            return {}

        if not (isinstance(call_node.func.value, gast.Call) and
                isinstance(call_node.func.value.func, gast.Name) and
                call_node.func.value.func.id == 'super'):
            return {}

        # Found super().__init__() call!
        # Get the parent class using MRO
        mro = inspect.getmro(class_obj)
        if len(mro) <= 1:
            return {}

        parent_class = mro[1]  # Next class in MRO after current class

        # Skip 'object' class
        if parent_class is object:
            return {}

        # Substitute parameters in the super().__init__ arguments
        super_args = [self._substitute_params(arg, param_map) for arg in call_node.args]
        super_keywords = []
        for kw in call_node.keywords:
            super_keywords.append(
                gast.keyword(
                    arg=kw.arg,
                    value=self._substitute_params(kw.value, param_map)
                )
            )

        # Recursively extract attributes from parent class
        try:
            parent_attrs = self._extract_attrs_from_hierarchy(
                parent_class, super_args, super_keywords
            )
            return parent_attrs
        except (TypeError, OSError, IOError):
            # Parent __init__ not available
            return {}

    def visit_Call(self, node):
        """Inline method calls (e.g., calc.square(x)).

        Args:
            node: Call AST node

        Returns:
            Inlined method body or original node
        """
        # Check for pattern: obj.method(args)
        if (isinstance(node.func, gast.Attribute) and
            isinstance(node.func.value, gast.Name)):

            obj_name = node.func.value.id
            method_name = node.func.attr

            # Check if this is a tracked instance
            if obj_name in self.instance_vars:
                class_obj = self.instance_vars[obj_name]['class']

                # Check if the method exists
                if hasattr(class_obj, method_name):
                    method = getattr(class_obj, method_name)

                    # Only inline if it's an instance method
                    if inspect.ismethod(method) or inspect.isfunction(method):
                        # Inline the method!
                        return self._inline_method(method, obj_name, node.args,
                                                   node.keywords)

        # Not a tracked method call, visit normally
        self.generic_visit(node)
        return node

    def _inline_method(self, method, instance_var, args, keywords):
        """Inline a method call by parsing and substituting its body.

        Args:
            method: The method object to inline
            instance_var: Name of the instance variable (e.g., 'calc')
            args: List of positional argument AST nodes
            keywords: List of keyword argument AST nodes

        Returns:
            Inlined expression (method body with substitutions)
        """
        try:
            # Get method source code
            source = inspect.getsource(method)
            source = textwrap.dedent(source)

            # Parse the method
            method_ast = gast.parse(source).body[0]

            # Find the return statement
            return_stmt = None
            for stmt in method_ast.body:
                if isinstance(stmt, gast.Return):
                    return_stmt = stmt
                    break

            if return_stmt is None or return_stmt.value is None:
                # Method doesn't return anything useful
                return gast.Constant(value=None, kind=None)

            # Build parameter substitution map
            params = method_ast.args.args[1:]  # Skip 'self'
            param_map = {}

            # Map positional arguments
            for i, param in enumerate(params):
                if i < len(args):
                    param_map[param.id] = args[i]

            # Map keyword arguments
            for keyword in keywords:
                param_map[keyword.arg] = keyword.value

            # Substitute parameters and 'self' in the return expression
            inlined_expr = self._substitute_in_expr(
                return_stmt.value, param_map, instance_var)

            return inlined_expr

        except (TypeError, OSError, IOError):
            # Can't get source or parse - return original call
            # This will likely fail later, but we tried
            return gast.Call(
                func=gast.Attribute(
                    value=gast.Name(id=instance_var, ctx=gast.Load(), annotation=None),
                    attr=method.__name__,
                    ctx=gast.Load()),
                args=args,
                keywords=keywords)

    def _substitute_in_expr(self, expr, param_map, instance_var):
        """Recursively substitute parameters and 'self' in an expression.

        Args:
            expr: Expression AST node
            param_map: Dict mapping parameter names to argument AST nodes
            instance_var: Name of instance variable (replaces 'self')

        Returns:
            Expression with substitutions applied
        """
        expr = copy.deepcopy(expr)

        class Substitutor(gast.NodeTransformer):
            """Recursively substitute names and attributes."""

            def __init__(self, param_map, instance_var, instance_attrs):
                self.param_map = param_map
                self.instance_var = instance_var
                self.instance_attrs = instance_attrs

            def visit_Name(self, node):
                # Substitute parameter names with their argument values
                if node.id in self.param_map:
                    return copy.deepcopy(self.param_map[node.id])
                return node

            def visit_Attribute(self, node):
                # Substitute self.attr
                if (isinstance(node.value, gast.Name) and
                    node.value.id == 'self'):

                    attr_name = node.attr

                    # If we know the attribute value from __init__, use it
                    if (self.instance_var in self.instance_attrs and
                        attr_name in self.instance_attrs[self.instance_var]):
                        return copy.deepcopy(
                            self.instance_attrs[self.instance_var][attr_name])

                    # Otherwise, replace self with instance variable name
                    # This handles cases where attributes aren't set in __init__
                    return gast.Attribute(
                        value=gast.Name(id=self.instance_var, ctx=gast.Load(),
                                       annotation=None),
                        attr=attr_name,
                        ctx=node.ctx)

                # Recursively visit the value
                self.generic_visit(node)
                return node

            def visit_Call(self, node):
                # Handle self.method() calls (method chaining)
                if (isinstance(node.func, gast.Attribute) and
                    isinstance(node.func.value, gast.Name) and
                    node.func.value.id == 'self'):

                    # This is a self.method() call - we'll handle it in the outer transformer
                    # For now, just replace self with the instance variable
                    node.func.value = gast.Name(id=self.instance_var,
                                                ctx=gast.Load(),
                                                annotation=None)

                # Recursively visit arguments
                self.generic_visit(node)
                return node

        substitutor = Substitutor(param_map, instance_var, self.instance_attrs)
        return substitutor.visit(expr)

    def _substitute_params(self, expr, param_map):
        """Substitute parameter names with their values.

        Args:
            expr: Expression AST node
            param_map: Dict mapping parameter names to values

        Returns:
            Expression with parameters substituted
        """
        expr = copy.deepcopy(expr)

        class ParamSubstitutor(gast.NodeTransformer):
            def __init__(self, param_map):
                self.param_map = param_map

            def visit_Name(self, node):
                if node.id in self.param_map:
                    return copy.deepcopy(self.param_map[node.id])
                return node

        substitutor = ParamSubstitutor(param_map)
        return substitutor.visit(expr)


def inline_class_methods(node, func):
    """Inline class method calls in an AST.

    Args:
        node: AST node (Module or FunctionDef)
        func: The function object being transformed (for __globals__ access)

    Returns:
        Transformed AST with class methods inlined
    """
    # Create a single inliner and run it multiple times to handle method chaining
    # The inliner maintains state about instance variables across passes
    inliner = ClassMethodInliner(func)

    # First pass: Track instance variables and do initial inlining
    node = inliner.visit(node)

    # Additional passes: Handle method chaining (methods calling other methods)
    # We need to keep the instance_vars tracking but visit again
    max_iterations = 10
    for iteration in range(max_iterations):
        prev_dump = gast.dump(node)

        # Create a new transformer but reuse the instance tracking
        chained_inliner = ClassMethodInliner(func)
        chained_inliner.instance_vars = inliner.instance_vars
        chained_inliner.instance_attrs = inliner.instance_attrs

        # Don't remove instance assignments on subsequent passes
        # Override visit_Assign to only track, not remove
        original_visit_assign = chained_inliner.visit_Assign

        def visit_assign_no_remove(node):
            # Track but don't remove
            original_visit_assign(node)
            return node

        chained_inliner.visit_Assign = visit_assign_no_remove

        node = chained_inliner.visit(node)

        # If nothing changed, we're done
        if gast.dump(node) == prev_dump:
            break

    return node
