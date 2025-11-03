# Copyright 2025 Google Inc.
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
"""Desugar lambda functions by inlining them.

This module transforms lambda functions into inlined expressions so that Tangent
can differentiate them. Since Tangent doesn't support nested function definitions,
we inline lambdas at their call sites.

Transformation Examples:
    # Before:
    g = lambda x: x ** 2
    return g(5)

    # After:
    return 5 ** 2

    # Before:
    f = lambda a, b: a * b + a ** 2
    return f(x, y)

    # After:
    return x * y + x ** 2
"""
from __future__ import absolute_import

import gast
import copy

from tangent import annotations as anno


class LambdaInliner(gast.NodeTransformer):
    """Inline lambda functions at their call sites.

    This approach works because:
    1. Tangent doesn't support nested functions
    2. Lambdas are typically used for simple expressions
    3. Inlining preserves the mathematical structure for AD

    Limitations:
    - Only handles lambdas assigned to variables and immediately called
    - Doesn't handle lambdas passed as arguments to other functions (yet)
    """

    def __init__(self):
        super(LambdaInliner, self).__init__()
        # Maps variable names to lambda nodes
        self.lambda_assignments = {}

    def visit_Assign(self, node):
        """Track assignments of lambdas to variables."""
        # Check if this is assigning a lambda to a simple name
        if (len(node.targets) == 1 and
                isinstance(node.targets[0], gast.Name) and
                isinstance(node.value, gast.Lambda)):

            var_name = node.targets[0].id
            lambda_node = node.value

            # Store the lambda for later inlining
            self.lambda_assignments[var_name] = lambda_node

            # Remove this assignment by returning None
            # (the lambda will be inlined at call sites)
            return None

        # Recursively process other assignments
        return self.generic_visit(node)

    def visit_Call(self, node):
        """Inline lambda calls if the function is a known lambda."""
        # First, recursively process arguments
        node = self.generic_visit(node)

        # Check if this is calling a lambda we've seen
        if isinstance(node.func, gast.Name) and node.func.id in self.lambda_assignments:
            lambda_node = self.lambda_assignments[node.func.id]

            # Create a substitution map from lambda params to call args
            param_names = [arg.id for arg in lambda_node.args.args]

            if len(param_names) != len(node.args):
                # Argument count mismatch - can't inline safely
                # Fall back to not inlining
                return node

            # Build substitution mapping
            substitutions = dict(zip(param_names, node.args))

            # Inline the lambda body with substituted arguments
            inlined_body = self._substitute_args(
                copy.deepcopy(lambda_node.body),
                substitutions
            )

            # Return the inlined expression
            return inlined_body

        return node

    def _substitute_args(self, node, substitutions):
        """Recursively substitute parameter names with argument expressions.

        Args:
            node: AST node (expression from lambda body)
            substitutions: Dict mapping parameter names to argument nodes

        Returns:
            Modified AST node with substitutions applied
        """
        class ArgSubstituter(gast.NodeTransformer):
            def __init__(self, subs):
                self.subs = subs

            def visit_Name(self, node):
                if isinstance(node.ctx, gast.Load) and node.id in self.subs:
                    # Replace parameter with argument
                    return copy.deepcopy(self.subs[node.id])
                return node

        return ArgSubstituter(substitutions).visit(node)


def desugar_lambdas(node):
    """Main entry point for lambda desugaring.

    Args:
        node: A gast Module or FunctionDef node

    Returns:
        The transformed AST with lambdas inlined at call sites
    """
    return LambdaInliner().visit(node)
