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
"""Validation of control flow constructs for automatic differentiation.

This module validates that control flow patterns in user code are compatible
with Tangent's AD implementation. It detects known limitations and provides
helpful error messages with workarounds.
"""
from __future__ import absolute_import

import gast
from tangent import quoting


class ControlFlowError(ValueError):
    """Exception raised when unsupported control flow patterns are detected."""
    pass


class ControlFlowValidator(gast.NodeVisitor):
    """Validates control flow patterns for AD compatibility.

    This visitor checks for known limitations in Tangent's handling of
    control flow statements and provides helpful error messages.
    """

    def __init__(self, source_code=''):
        """Initialize the validator.

        Args:
            source_code: Original source code for better error messages.
        """
        self.source_code = source_code
        self.errors = []
        self.warnings = []
        self.in_while = False
        self.while_stack = []  # Track nested while loops
        self.continue_vars = set()  # Variables assigned before continue

    def visit_While(self, node):
        """Check while loops for compatibility issues."""
        # Enter while loop context
        old_in_while = self.in_while
        self.in_while = True
        self.while_stack.append({
            'assigned_before_continue': set(),
            'used_after_assignment': set(),
            'has_continue': False
        })

        # Visit the while body
        for stmt in node.body:
            self.visit(stmt)

        # Check for problematic pattern: assignment before continue, used in computation
        loop_info = self.while_stack.pop()
        if loop_info['has_continue']:
            problematic_vars = (loop_info['assigned_before_continue'] &
                               loop_info['used_after_assignment'])
            if problematic_vars:
                self.warnings.append({
                    'type': 'while_continue_variable',
                    'vars': problematic_vars,
                    'message': (
                        f"While loop with continue statement uses variables "
                        f"({', '.join(sorted(problematic_vars))}) that are assigned "
                        f"before the continue check. This may cause gradient computation "
                        f"errors if these variables are used in differentiable operations.\n\n"
                        f"Workaround: Move variable assignments after the continue check, "
                        f"or avoid using these variables in computations that affect the gradient."
                    ),
                    'node': node
                })

        # Restore context
        self.in_while = old_in_while
        self.generic_visit(node)

    def visit_Continue(self, node):
        """Mark that we've seen a continue statement."""
        if self.while_stack:
            self.while_stack[-1]['has_continue'] = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Track variable assignments in while loops."""
        if self.while_stack:
            # Get variable names being assigned
            for target in node.targets:
                if isinstance(target, gast.Name):
                    # This variable is being assigned
                    self.while_stack[-1]['assigned_before_continue'].add(target.id)

            # Check if the RHS uses any tracked variables
            class VarUseFinder(gast.NodeVisitor):
                def __init__(self):
                    self.used_vars = set()
                def visit_Name(self, n):
                    if isinstance(n.ctx, gast.Load):
                        self.used_vars.add(n.id)

            finder = VarUseFinder()
            finder.visit(node.value)
            self.while_stack[-1]['used_after_assignment'].update(finder.used_vars)

        self.generic_visit(node)

    def visit_If(self, node):
        """Check if statements for conditional control flow issues."""
        # Check for the "Node has no annotation 'active_out'" issue
        # This happens with certain conditional patterns
        # For now, we don't have a good way to detect this statically
        # since it depends on the control flow analysis in fence.py
        self.generic_visit(node)

    def validate(self):
        """Run validation and return errors/warnings.

        Returns:
            A tuple of (errors, warnings) where each is a list of dicts
            containing validation messages.
        """
        return self.errors, self.warnings

    def format_message(self, item):
        """Format an error or warning message for display.

        Args:
            item: A dict containing error/warning information.

        Returns:
            A formatted string with the message and context.
        """
        msg = f"\n{item['type'].upper()}: {item['message']}"
        if 'node' in item and self.source_code:
            try:
                line_num = item['node'].lineno if hasattr(item['node'], 'lineno') else '?'
                msg += f"\n  at line {line_num}"
            except:
                pass
        return msg


def validate_control_flow(node, source_code='', verbose=False):
    """Validate control flow patterns in an AST.

    Args:
        node: The AST node to validate (typically a Module or FunctionDef).
        source_code: Optional source code for better error messages.
        verbose: If True, print warnings even if there are no errors.

    Returns:
        The original node (unchanged).

    Raises:
        ControlFlowError: If unsupported control flow patterns are detected.
    """
    validator = ControlFlowValidator(source_code)
    validator.visit(node)
    errors, warnings = validator.validate()

    # Errors are fatal
    if errors:
        error_msgs = [validator.format_message(e) for e in errors]
        raise ControlFlowError(
            "Unsupported control flow patterns detected:\n" +
            "\n".join(error_msgs)
        )

    # Warnings are informational
    if warnings and verbose:
        print("=" * 70)
        print("CONTROL FLOW WARNINGS")
        print("=" * 70)
        for warning in warnings:
            print(validator.format_message(warning))
        print("=" * 70)
        print("Gradient computation may fail for the patterns above.")
        print("See warnings for suggested workarounds.")
        print("=" * 70)

    return node
