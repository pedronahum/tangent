# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Tangent-specific errors."""
from __future__ import absolute_import

# Import enhanced error handling
from tangent.error_handlers import (
    TangentError,
    UnsupportedSyntaxError,
    GradientNotFoundError,
    SourceCodeNotAvailableError,
    NonScalarOutputError,
    TypeMismatchError,
    InplaceModificationError,
    format_error_with_context,
    enhance_traceback
)


class TangentParseError(SyntaxError):
  """Error generated when encountering an unsupported feature."""
  pass


class ForwardNotImplementedError(NotImplementedError):
  """Error generated when encountering a @tangent_ yet to be implemented."""

  def __init__(self, func):
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    message = f'''Forward mode for function "{func_name}" is not yet implemented.

💡 Suggestion:
  Use reverse mode (default) instead:
    df = tangent.grad(f)           # Uses reverse mode
    df = tangent.autodiff(f, mode='reverse')

  Or define a custom forward-mode gradient:
    @tangent.tangent(original_function)
    def tangent_original_function(x, dx):
        # Compute forward-mode derivative
        return result, dresult

📖 Documentation: https://github.com/google/tangent#forward-and-reverse-mode
'''
    NotImplementedError.__init__(self, message)


class ReverseNotImplementedError(NotImplementedError):
  """Error generated when encountering an @adjoint yet to be implemented."""

  def __init__(self, func):
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    message = f'''Reverse mode for function "{func_name}" is not yet implemented.

💡 Suggestion:
  To define a custom reverse-mode gradient:

    @tangent.adjoint(original_function)
    def adjoint_original_function(df, *args):
        # Compute reverse-mode gradient
        # df is the gradient of output
        # return gradients with respect to inputs
        return dx

  Example:
    def my_function(x):
        return x ** 3

    @tangent.adjoint(my_function)
    def d_my_function(df, x):
        return df * 3 * x ** 2

📖 Documentation: https://github.com/google/tangent#custom-gradients
'''
    NotImplementedError.__init__(self, message)
