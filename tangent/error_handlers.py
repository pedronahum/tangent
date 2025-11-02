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
"""Enhanced error handling for Tangent with source code context and helpful suggestions.

This module provides rich error messages that include:
- Source code context with line numbers
- Highlighted error locations
- Helpful suggestions for common issues
- Links to documentation
"""
from __future__ import absolute_import

import ast
import inspect
import sys
import textwrap
from typing import Optional, List, Tuple


class TangentError(Exception):
    """Base class for all Tangent errors with enhanced formatting."""

    def __init__(self, message, func=None, node=None, suggestion=None,
                 doc_link=None, original_error=None):
        """Create a Tangent error with rich context.

        Args:
            message: The error message
            func: The function being differentiated (if available)
            node: The AST node where the error occurred (if available)
            suggestion: A helpful suggestion for fixing the error
            doc_link: URL to relevant documentation
            original_error: The original exception that was caught (if any)
        """
        self.message = message
        self.func = func
        self.node = node
        self.suggestion = suggestion
        self.doc_link = doc_link
        self.original_error = original_error

        # Build the formatted error message
        formatted_msg = self._format_error()
        super(TangentError, self).__init__(formatted_msg)

    def _format_error(self):
        """Format the error with source context and suggestions."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Tangent Error")
        lines.append("=" * 80)
        lines.append("")
        lines.append(self.message)
        lines.append("")

        # Add source code context if available
        if self.func is not None:
            source_context = self._get_source_context()
            if source_context:
                lines.append("Source Code Context:")
                lines.append("-" * 80)
                lines.extend(source_context)
                lines.append("")

        # Add AST node information if available
        if self.node is not None and hasattr(self.node, 'lineno'):
            lines.append(f"Error at line {self.node.lineno}")
            if hasattr(self.node, 'col_offset'):
                lines.append(f"Column {self.node.col_offset}")
            lines.append("")

        # Add original error if available
        if self.original_error:
            lines.append("Original Error:")
            lines.append("-" * 80)
            lines.append(f"{type(self.original_error).__name__}: {str(self.original_error)}")
            lines.append("")

        # Add suggestion if available
        if self.suggestion:
            lines.append("💡 Suggestion:")
            lines.append("-" * 80)
            for suggestion_line in self.suggestion.split('\n'):
                lines.append(f"  {suggestion_line}")
            lines.append("")

        # Add documentation link if available
        if self.doc_link:
            lines.append(f"📖 Documentation: {self.doc_link}")
            lines.append("")

        lines.append("=" * 80)
        return '\n'.join(lines)

    def _get_source_context(self, context_lines=3):
        """Get source code context around the error location.

        Args:
            context_lines: Number of lines to show before and after error

        Returns:
            List of formatted source lines with line numbers
        """
        if self.func is None:
            return None

        try:
            source_lines = inspect.getsource(self.func).split('\n')
        except (OSError, TypeError):
            return None

        # If we have a node with line number, highlight that line
        if self.node and hasattr(self.node, 'lineno'):
            error_line = self.node.lineno - 1  # Convert to 0-indexed
        else:
            error_line = None

        formatted_lines = []
        total_lines = len(source_lines)

        if error_line is not None:
            # Show context around the error line
            start = max(0, error_line - context_lines)
            end = min(total_lines, error_line + context_lines + 1)
        else:
            # Show first few lines
            start = 0
            end = min(total_lines, 10)

        for i in range(start, end):
            line_num = i + 1
            line = source_lines[i]

            # Highlight the error line
            if i == error_line:
                marker = ">>> "
                formatted_lines.append(f"{marker}{line_num:4d} | {line}")
                # Add column marker if available
                if hasattr(self.node, 'col_offset'):
                    col = self.node.col_offset
                    formatted_lines.append(" " * (len(marker) + 7 + col) + "^" * max(1, getattr(self.node, 'end_col_offset', col + 1) - col))
            else:
                formatted_lines.append(f"    {line_num:4d} | {line}")

        return formatted_lines


class UnsupportedSyntaxError(TangentError):
    """Error for unsupported Python syntax."""

    def __init__(self, syntax_feature, func=None, node=None):
        message = f"Unsupported Python syntax: {syntax_feature}"

        # Common suggestions for unsupported syntax
        suggestions = {
            'global': 'Try passing values as function arguments instead of using global variables.',
            'nonlocal': 'Try restructuring your code to avoid nonlocal variables.',
            'yield': 'Generator functions are not yet supported. Try using regular functions.',
            'async': 'Async functions are not yet supported. Use synchronous functions.',
            'with': 'Some context managers may not be supported. Try simplifying your code.',
        }

        suggestion = suggestions.get(syntax_feature,
                                    'This Python feature is not yet supported in Tangent. '
                                    'Try rewriting your code without this feature.')

        super(UnsupportedSyntaxError, self).__init__(
            message=message,
            func=func,
            node=node,
            suggestion=suggestion,
            doc_link='https://github.com/google/tangent#limitations'
        )


class GradientNotFoundError(TangentError):
    """Error when gradient definition is not found for a function."""

    def __init__(self, func_name, func=None, node=None):
        message = f"No gradient definition found for function: {func_name}"

        suggestion = f'''To define a custom gradient for "{func_name}", use:

@tangent.adjoint(original_function)
def d_original_function(df, *args):
    # Compute gradients here
    return gradients

Example:
    def my_function(x):
        return x ** 3

    @tangent.adjoint(my_function)
    def d_my_function(df, x):
        return df * 3 * x ** 2

For more details, see the custom gradients documentation.'''

        super(GradientNotFoundError, self).__init__(
            message=message,
            func=func,
            node=node,
            suggestion=suggestion,
            doc_link='https://github.com/google/tangent#custom-gradients'
        )


class SourceCodeNotAvailableError(TangentError):
    """Error when function source code cannot be retrieved."""

    def __init__(self, func_name, func=None):
        message = f"Cannot retrieve source code for function: {func_name}"

        suggestion = '''Tangent requires access to function source code for differentiation.

Common causes:
1. Function defined in Python REPL/interactive session
   → Define functions in .py files and import them

2. Function is a built-in or C extension
   → Define a Python wrapper or custom gradient

3. Function is dynamically generated (exec, eval)
   → Define functions normally using def

4. Function is imported from compiled bytecode (.pyc)
   → Ensure source .py files are available'''

        super(SourceCodeNotAvailableError, self).__init__(
            message=message,
            func=func,
            suggestion=suggestion
        )


class NonScalarOutputError(TangentError):
    """Error when trying to take gradient of non-scalar function."""

    def __init__(self, output_shape, func=None):
        if hasattr(output_shape, '__len__'):
            shape_str = f"shape {output_shape}"
        else:
            shape_str = f"type {type(output_shape).__name__}"

        message = f"Function output must be scalar, but got {shape_str}"

        suggestion = '''tangent.grad() requires a scalar output (single number).

For non-scalar outputs, you have several options:

1. Use tangent.autodiff() for vector outputs:
   df = tangent.autodiff(f, mode='reverse')

2. Reduce to scalar with sum:
   def scalar_f(x):
       return np.sum(vector_f(x))

   df = tangent.grad(scalar_f)

3. Use a loss function:
   def loss(params):
       pred = model(params, x)
       return np.sum((pred - y) ** 2)

   d_loss = tangent.grad(loss)'''

        super(NonScalarOutputError, self).__init__(
            message=message,
            func=func,
            suggestion=suggestion
        )


class TypeMismatchError(TangentError):
    """Error for type mismatches in gradient computation."""

    def __init__(self, expected_type, actual_type, context, func=None, node=None):
        message = f"Type mismatch in {context}: expected {expected_type}, got {actual_type}"

        suggestion = f'''Type mismatch detected during gradient computation.

Common fixes:
1. Ensure consistent data types throughout your function
2. Cast arrays explicitly: x = x.astype(np.float64)
3. For TensorFlow: use tf.cast(x, tf.float32)
4. Check that input types match expected types

Context: {context}'''

        super(TypeMismatchError, self).__init__(
            message=message,
            func=func,
            node=node,
            suggestion=suggestion
        )


class InplaceModificationError(TangentError):
    """Error for in-place modifications that break autodiff."""

    def __init__(self, variable_name, func=None, node=None):
        message = f"In-place modification detected for variable: {variable_name}"

        suggestion = f'''In-place modifications can break automatic differentiation.

Instead of modifying arrays in-place:
    ❌ x[0] = 5
    ❌ x += 1
    ❌ x *= 2

Create new arrays:
    ✅ x = x.at[0].set(5)  # JAX
    ✅ x = x + 1
    ✅ x = x * 2

Variable: {variable_name}'''

        super(InplaceModificationError, self).__init__(
            message=message,
            func=func,
            node=node,
            suggestion=suggestion
        )


def format_error_with_context(error, func=None, node=None):
    """Wrap a standard Python error with Tangent context.

    Args:
        error: The original error
        func: The function being differentiated
        node: The AST node where error occurred

    Returns:
        A TangentError with enhanced formatting
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Detect common error patterns and provide specific suggestions
    if isinstance(error, ValueError):
        if 'could not get source code' in error_msg:
            return SourceCodeNotAvailableError(
                func_name=func.__name__ if func else 'unknown',
                func=func
            )

    elif isinstance(error, TypeError):
        if 'unsupported operand type' in error_msg:
            return TypeMismatchError(
                expected_type='numeric',
                actual_type='mixed',
                context='arithmetic operation',
                func=func,
                node=node
            )

    # Generic wrapper for other errors
    return TangentError(
        message=f"{error_type}: {error_msg}",
        func=func,
        node=node,
        original_error=error
    )


def enhance_traceback(exc_info):
    """Enhance exception traceback with Tangent-specific information.

    Args:
        exc_info: Exception info tuple from sys.exc_info()

    Returns:
        Enhanced exception with better formatting
    """
    exc_type, exc_value, exc_tb = exc_info

    # If already a TangentError, return as-is
    if isinstance(exc_value, TangentError):
        return exc_value

    # Try to extract function and node information from traceback
    func = None
    node = None

    # Wrap with generic TangentError
    return format_error_with_context(exc_value, func=func, node=node)
