# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License');
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
"""The fence allows placing language feature restrictions on the AST.

The fence works by walking an AST and raising an error if the tree contains any
of the unsupported features. Only the first encountered feature is flagged.

For a detailed documentation of AST nodes, see
http://greentreesnakes.readthedocs.io/en/latest/nodes.html
"""
from __future__ import absolute_import
from __future__ import division

import gast as ast

from tangent.errors import TangentParseError
from tangent import error_suggestions


def validate(node, source):
  """Call this function to validate an AST."""
  # TODO: leaving strict checking off to support insert_grad_of
  lf = LanguageFence(source, strict=False)
  lf.visit(node)
  return node


class _NestedFunctionFinder(ast.NodeVisitor):
  """Detects a function definition nested inside another function body.

  Tangent differentiates a single function with exactly one (normalized)
  return. A nested def introduces a second FunctionDef whose return breaks the
  reverse transform, and closures/recursion cannot be represented at all. This
  must run before any desugaring pass, because passes such as
  explicit_loop_indexes also choke on nested defs with an opaque IndexError.
  """

  def __init__(self, source):
    self._source = source
    self._function_depth = 0

  def _visit_def(self, node):
    if self._function_depth >= 1:
      feature = 'Nested function definitions'
      message = '%s are not supported' % feature
      suggestion = error_suggestions.get_suggestion(feature)
      if suggestion:
        message = error_suggestions.format_error_with_suggestion(
            feature, message)
      lineno = getattr(node, 'lineno', 1)
      offset = getattr(node, 'col_offset', 0)
      lines = self._source.splitlines()
      line = lines[lineno - 1] if 0 < lineno <= len(lines) else ''
      # TangentParseError is a SyntaxError: (msg, (filename, lineno, offset,
      # text)), matching LanguageFence._raise_error.
      raise TangentParseError(message, ('<stdin>', lineno, offset + 1, line))
    self._function_depth += 1
    self.generic_visit(node)
    self._function_depth -= 1

  def visit_FunctionDef(self, node):
    self._visit_def(node)

  def visit_AsyncFunctionDef(self, node):
    self._visit_def(node)


def validate_no_nested_functions(node, source):
  """Reject nested function definitions early, before any desugaring pass."""
  _NestedFunctionFinder(source).visit(node)
  return node


class LanguageFence(ast.NodeVisitor):
  """An AST visitor that raises an error for unsupported language features.

  This implementation is not thread-safe.

  LanguageFence instances are lightweight and tied to the AST they validate.
  In general, you should not attempt to reuse them.
  """

  def __init__(self, source, strict=True):
    """Creates a LanguageFence.

    Args:
      source: String, the source code of the AST that will be verified.
      strict: Boolean, set to False to allow unsafe constructs.
    Raises:
      ValueError: if source code has not been supplied.
    """
    self._visited_top_module = False
    if not source:
      raise ValueError('The source code of the tree is required.')
    self._source = source
    self._strict = strict

    # Location information is used to locate the offending elements
    # in the source code.
    self._current_lineno = None  # Only consistent during a visit.
    self._current_offset = None  # Only consistent during a visit.

    super(LanguageFence, self).__init__()

  def _raise_error(self, msg):
    assert self._source
    lineno = self._current_lineno
    offset = self._current_offset
    line = self._source.splitlines()[lineno - 1]
    raise TangentParseError(msg, ('<stdin>', lineno, offset + 1, line))

  def _track_location(self, node):
    # TODO: Add tests that cover all nodes.
    exposed_symbols = dir(node)
    # Not all nodes have source information. This is a generic way to collect
    # whenever available.
    if 'lineno' in exposed_symbols and 'col_offset' in exposed_symbols:
      self._current_lineno = node.lineno
      self._current_offset = node.col_offset

  def _allow_and_continue(self, node):
    self._track_location(node)
    self.generic_visit(node)

  def _reject(self, node, msg):
    """Reject a node with an enhanced error message including suggestions."""
    self._track_location(node)

    # Extract feature name from message (e.g., "F-Strings are not supported" -> "F-Strings")
    feature_name = msg.split(' are not supported')[0].split(' is not supported')[0]

    # Get suggestion if available
    suggestion = error_suggestions.get_suggestion(feature_name)

    # Enhance the error message with suggestion
    if suggestion:
      enhanced_msg = f"{msg}\n\n💡 Suggestion:\n{suggestion}"
    else:
      enhanced_msg = msg

    self._raise_error(enhanced_msg)

  def visit_Module(self, node):
    self._visited_top_module = True
    self._allow_and_continue(node)

  def visit_Constant(self, node):
    # Bytes literals are not differentiable; on Python 3.8+ they arrive as
    # Constant nodes (the old ast.Bytes class is a deprecated alias).
    if isinstance(node.value, bytes):
      self._reject(node, 'Byte Literals are not supported')
    self._allow_and_continue(node)

  def visit_FormattedValue(self, node):
    # f-strings produce a (non-differentiable) string, typically used for
    # debug or assert messages, so they are safe to allow.
    self._allow_and_continue(node)

  def visit_JoinedStr(self, node):
    self._allow_and_continue(node)

  def visit_List(self, node):
    self._allow_and_continue(node)

  def visit_Tuple(self, node):
    # TODO: Make sure none of the original functionality was lost.
    self._allow_and_continue(node)

  def visit_Set(self, node):
    # A set literal is a non-differentiable collection, most commonly used as a
    # membership guard (e.g. `if x in {1, 2, 3}`). Allow it like tuples/lists.
    self._allow_and_continue(node)

  def visit_Dict(self, node):
    self._allow_and_continue(node)

  def visit_Ellipsis(self, node):
    self._allow_and_continue(node)

  def visit_NameConstant(self, node):
    self._allow_and_continue(node)

  def visit_Name(self, node):
    self._allow_and_continue(node)

  def visit_Load(self, node):
    self._allow_and_continue(node)

  def visit_Store(self, node):
    self._allow_and_continue(node)

  def visit_Del(self, node):
    self._reject(node, 'Deleting variables is not supported')

  def visit_Starred(self, node):
    self._reject(node, 'Unpackings are not supported')

  def visit_Expr(self, node):
    self._allow_and_continue(node)

  def visit_UnaryOp(self, node):
    self._allow_and_continue(node)

  def visit_UAdd(self, node):
    self._reject(node, 'Unary Add operator is not supported')

  def visit_USub(self, node):
    self._allow_and_continue(node)

  def visit_Not(self, node):
    self._allow_and_continue(node)

  def visit_Invert(self, node):
    self._reject(node, 'Invert operator is not supported')

  def visit_BinOp(self, node):
    self._allow_and_continue(node)

  def visit_Add(self, node):
    self._allow_and_continue(node)

  def visit_Sub(self, node):
    self._allow_and_continue(node)

  def visit_Mult(self, node):
    self._allow_and_continue(node)

  def visit_Div(self, node):
    self._allow_and_continue(node)

  def visit_FloorDiv(self, node):
    self._reject(node, 'Floor Div operator is not supported')

  def visit_Mod(self, node):
    self._allow_and_continue(node)

  def visit_Pow(self, node):
    self._allow_and_continue(node)

  def visit_LShift(self, node):
    self._reject(node, 'Left Shift operator is not supported')

  def visit_RShift(self, node):
    self._reject(node, 'Right Shift operator is not supported')

  def visit_BitOr(self, node):
    self._reject(node, 'Bitwise Or operator is not supported')

  def visit_BitXor(self, node):
    self._reject(node, 'Bitwise Xor operator is not supported')

  def visit_BitAnd(self, node):
    self._reject(node, 'Bitwise And operator is not supported')

  def visit_MatMult(self, node):
    # TODO: Add support for this.
    self._reject(node, 'MatMult operator is not supported')

  def visit_BoolOp(self, node):
    self._allow_and_continue(node)

  def visit_And(self, node):
    self._allow_and_continue(node)

  def visit_Or(self, node):
    self._allow_and_continue(node)

  def visit_Compare(self, node):
    self._allow_and_continue(node)

  def visit_Eq(self, node):
    self._allow_and_continue(node)

  def visit_NotEq(self, node):
    self._allow_and_continue(node)

  def visit_Lt(self, node):
    self._allow_and_continue(node)

  def visit_LtE(self, node):
    self._allow_and_continue(node)

  def visit_Gt(self, node):
    self._allow_and_continue(node)

  def visit_GtE(self, node):
    self._allow_and_continue(node)

  def visit_Is(self, node):
    self._allow_and_continue(node)

  def visit_IsNot(self, node):
    self._allow_and_continue(node)

  def visit_In(self, node):
    # Membership tests produce a (non-differentiable) boolean, just like the
    # other comparison operators, so they are safe to allow in conditions.
    self._allow_and_continue(node)

  def visit_NotIn(self, node):
    self._allow_and_continue(node)

  def visit_Call(self, node):
    self._allow_and_continue(node)

  def visit_keyword(self, node):
    self._allow_and_continue(node)

  def visit_IfExp(self, node):
    self._allow_and_continue(node)

  def visit_NamedExpr(self, node):
    # The walrus operator (y := expr) binds a name in expression position. The
    # bound variable is not tracked as an active intermediate, so the adjoint of
    # any branch that uses it is dropped and the gradient comes back silently
    # wrong (usually zero). Reject it rather than return an incorrect result.
    self._reject(node, 'Assignment expressions (the walrus operator ":=") are '
                        'not supported')

  def visit_Attribute(self, node):
    self._allow_and_continue(node)

  def visit_Subscript(self, node):
    self._allow_and_continue(node)

  def visit_Index(self, node):
    self._allow_and_continue(node)

  def visit_Slice(self, node):
    self._allow_and_continue(node)

  def visit_ExtSlice(self, node):
    self._allow_and_continue(node)

  def visit_ListComp(self, node):
    self._allow_and_continue(node)

  def visit_SetComp(self, node):
    self._reject(node, 'Set Comprehensions are not supported')

  def visit_GeneratorExp(self, node):
    self._reject(node, 'Generator Expressions are not supported')

  def visit_DictComp(self, node):
    self._reject(node, 'Dictionary Comprehensions are not supported')

  def visit_comprehension(self, node):
    self._allow_and_continue(node)

  def visit_Assign(self, node):
    self._allow_and_continue(node)

  def visit_AnnAssign(self, node):
    self._reject(node, 'Type-annotated assignment are not supported')

  def visit_AugAssign(self, node):
    # Augmented assignment to a subscript or attribute (e.g. `a[i] += x`) is not
    # supported: the in-place update is lost during normalization, which
    # corrupts both the primal value and the gradient. Reject it rather than
    # return a silently wrong result. Plain `x += y` on a name is fine.
    if isinstance(node.target, (ast.Subscript, ast.Attribute)):
      self._reject(node,
                   'Augmented assignment to a subscript or attribute is not '
                   'supported')
    else:
      self._allow_and_continue(node)

  def visit_Print(self, node):
    self._allow_and_continue(node)

  def visit_Raise(self, node):
    # Raise statements cannot be differentiated: the reverse pass has no way to
    # propagate adjoints across an exception edge. Reject up front rather than
    # fail later with an opaque "Unknown node type" error.
    self._reject(node, 'Raise statements are not supported')

  def visit_Assert(self, node):
    if __debug__:
      self._allow_and_continue(node)
    else:
      assert False, 'Assert statements should not appear in optimized code'

  def visit_Delete(self, node):
    self._reject(node, 'Delete statements are not supported')

  def visit_Pass(self, node):
    self._allow_and_continue(node)

  def visit_Import(self, node):
    self._reject(node, 'Import statements are not supported')

  def visit_ImportFrom(self, node):
    self._reject(node, 'Import/From statements are not supported')

  def visit_alias(self, node):
    self._reject(node, 'Aliases are not supported')

  def visit_If(self, node):
    self._allow_and_continue(node)

  def visit_For(self, node):
    if node.orelse:
      self._reject(node, 'For/Else block is not supported')
    else:
      self._allow_and_continue(node)

  def visit_While(self, node):
    self._allow_and_continue(node)

  def visit_Break(self, node):
    # Break is never supported. The reverse-mode loop tape records one entry per
    # completed iteration, but `break` exits mid-iteration before the iteration
    # counter is advanced, so the backward pass replays the wrong number of
    # iterations and silently produces incorrect gradients. Reject it outright
    # (regardless of strict mode) rather than emit a plausible-but-wrong result.
    self._reject(node, 'Break statements are not supported')

  def visit_Continue(self, node):
    # Continue has the same tape/iteration-counter problem as break (see
    # visit_Break): it only happens to give correct gradients when no active
    # computation precedes it, and otherwise fails or is silently wrong.
    self._reject(node, 'Continue statements are not supported')

  def visit_Try(self, node):
    # On Python 3.8+ there is a single `Try` node (the old TryExcept /
    # TryFinally nodes are gone). Try blocks are not differentiable: multiple
    # returns across handler blocks also defeat the single-exit transform.
    # Reject here instead of crashing in reverse mode with "function must have
    # exactly one return statement".
    self._reject(node, 'Try/Except blocks are not supported')

  def visit_TryFinally(self, node):
    self._reject(node, 'Try/Finally blocks are not supported')

  def visit_TryExcept(self, node):
    self._reject(node, 'Try/Except blocks are not supported')

  def visit_ExceptHandler(self, node):
    self._allow_and_continue(node)

  def visit_With(self, node):
    self._allow_and_continue(node)

  def visit_withitem(self, node):
    self._allow_and_continue(node)

  def visit_FunctionDef(self, node):
    self._allow_and_continue(node)

  def visit_Lambda(self, node):
    self._reject(node, 'Lambda functions are not supported')

  def visit_arguments(self, node):
    # *args / **kwargs cannot be differentiated: the activity analysis indexes
    # positional arguments by position (wrt), which is ill-defined for a
    # variable-length argument pack and crashes with an IndexError. Plain
    # positional and keyword arguments with defaults are fine.
    if node.vararg is not None:
      self._reject(node, 'Variadic positional arguments (*args) are not '
                          'supported')
    if node.kwarg is not None:
      self._reject(node, 'Variadic keyword arguments (**kwargs) are not '
                          'supported')
    self._allow_and_continue(node)

  def visit_arg(self, node):
    self._allow_and_continue(node)

  def visit_Return(self, node):
    # TODO: Make sure none of the original functionality was lost.
    self._allow_and_continue(node)

  def visit_Yield(self, node):
    self._reject(node, 'Yield statements are not supported')

  def visit_YieldFrom(self, node):
    self._reject(node, 'Yield/From statements are not supported')

  def visit_Global(self, node):
    self._reject(node, 'Global statements are not supported')

  def visit_Nonlocal(self, node):
    self._reject(node, 'Nonlocal statements are not supported')

  def visit_ClassDef(self, node):
    self._reject(node, 'Classes are not supported')

  def visit_AsyncFunctionDef(self, node):
    self._reject(node, 'Async function definitions are not supported')

  def visit_Await(self, node):
    self._reject(node, 'Await statements are not supported')

  def visit_AsyncFor(self, node):
    self._reject(node, 'Async For loops are not supported')

  def visit_AsyncWith(self, node):
    self._reject(node, 'Async With statements are not supported')
