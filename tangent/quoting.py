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
"""Moving between source code and AST."""
from __future__ import absolute_import
import inspect
import textwrap

import astor
import gast

from tangent import annotations as anno


class TangentParseError(SyntaxError):
  pass


class SourceWithCommentGenerator(astor.codegen.SourceGenerator):
  """Source code generator that outputs comments."""

  def __init__(self, *args, **kwargs):
    super(SourceWithCommentGenerator, self).__init__(*args, **kwargs)
    self.new_indentation = True

  def body(self, statements):
    self.new_indentation = True
    super(SourceWithCommentGenerator, self).body(statements)

  def visit(self, node, abort=astor.codegen.SourceGenerator.abort_visit):
    if anno.hasanno(node, 'comment'):
      comment = anno.getanno(node, 'comment')
      # Preprocess the comment to fit to maximum line width of 80 characters
      linewidth = 78
      if comment['location'] in ('above', 'below'):
        comment['text'] = comment['text'][:linewidth]
      n_newlines = 1 if self.new_indentation else 2
      if comment['location'] == 'above':
        self.result.append('\n' * n_newlines)
        self.result.append(self.indent_with * self.indentation)
        self.result.append('# %s' % comment['text'])
        super(SourceWithCommentGenerator, self).visit(node)
      elif comment['location'] == 'below':
        super(SourceWithCommentGenerator, self).visit(node)
        self.result.append('\n')
        self.result.append(self.indent_with * self.indentation)
        self.result.append('# %s' % comment['text'])
        self.result.append('\n' * (n_newlines - 1))
      elif comment['location'] == 'right':
        super(SourceWithCommentGenerator, self).visit(node)
        self.result.append(' # %s' % comment['text'])
      else:
        raise TangentParseError('Only valid comment locations are '
                                'above, below, right')
    else:
      self.new_indentation = False
      super(SourceWithCommentGenerator, self).visit(node)


def _ensure_type_comments(node):
  """Recursively ensure all gast nodes have required attributes for AST conversion.

  This is needed for Python 3.8+ compatibility where gast_to_ast expects
  certain attributes on nodes (type_comment, type_params, type_ignores, etc).
  """
  if isinstance(node, gast.AST):
    # Attributes that may be missing and need default values
    missing_attrs = {
        'type_comment': None,
        'type_params': [],
        'type_ignores': [],
    }

    for attr, default in missing_attrs.items():
      if not hasattr(node, attr):
        try:
          setattr(node, attr, default)
        except (AttributeError, TypeError):
          pass  # Some nodes don't support setting attributes

    # Recursively process all child nodes
    for field, value in gast.iter_fields(node):
      if isinstance(value, list):
        for item in value:
          _ensure_type_comments(item)
      elif isinstance(value, gast.AST):
        _ensure_type_comments(value)

  return node


def _copy_annotations(gast_node, ast_node, annotation_map):
  """Copy annotations from gast nodes to converted AST nodes using a mapping.

  Args:
    gast_node: Original gast node (before conversion)
    ast_node: Converted standard AST node
    annotation_map: Dictionary mapping gast node IDs to annotations
  """
  import ast

  # Copy annotations from the mapping to the converted node
  gast_id = id(gast_node)
  if gast_id in annotation_map:
    annotations = annotation_map[gast_id]
    if annotations:
      setattr(ast_node, anno.ANNOTATION_FIELD, annotations)

  # Recursively process child nodes
  for gast_field, gast_value in gast.iter_fields(gast_node):
    if hasattr(ast_node, gast_field):
      ast_value = getattr(ast_node, gast_field)

      if isinstance(gast_value, list) and isinstance(ast_value, list):
        # Process lists of nodes
        for gast_child, ast_child in zip(gast_value, ast_value):
          if isinstance(gast_child, gast.AST) and isinstance(ast_child, ast.AST):
            _copy_annotations(gast_child, ast_child, annotation_map)
      elif isinstance(gast_value, gast.AST) and isinstance(ast_value, ast.AST):
        # Process single nodes
        _copy_annotations(gast_value, ast_value, annotation_map)


def _collect_annotations(node):
  """Collect all annotations from a gast tree into a dictionary.

  Args:
    node: A gast node

  Returns:
    Dictionary mapping node ID to its annotations dictionary
  """
  annotation_map = {}

  for child in gast.walk(node):
    if hasattr(child, anno.ANNOTATION_FIELD):
      annotations = getattr(child, anno.ANNOTATION_FIELD)
      if annotations:
        annotation_map[id(child)] = annotations.copy()

  return annotation_map


def to_source(node, indentation=' ' * 4):
  """Return source code of a given AST."""
  if isinstance(node, gast.AST):
    # Collect annotations before conversion
    annotation_map = _collect_annotations(node)

    # Add missing type_comment attributes before conversion (Python 3.8+ compat)
    node_gast = _ensure_type_comments(node)
    node_ast = gast.gast_to_ast(node_gast)

    # Copy annotations to the converted AST
    _copy_annotations(node_gast, node_ast, annotation_map)

    node = node_ast
  generator = SourceWithCommentGenerator(indentation, False,
                                         astor.string_repr.pretty_string)
  generator.visit(node)
  generator.result.append('\n')
  return astor.source_repr.pretty_source(generator.result).lstrip()


def parse_function(fn):
  """Get the source of a function and return its AST."""
  try:
    return parse_string(inspect.getsource(fn))
  except (IOError, OSError) as e:
    # Use enhanced error handler
    from tangent.error_handlers import SourceCodeNotAvailableError
    func_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
    raise SourceCodeNotAvailableError(func_name=func_name, func=fn)


def parse_string(src):
  """Parse a string into an AST."""
  return gast.parse(textwrap.dedent(src))


def quote(src_string, return_expr=False):
  """Go from source code to AST nodes.

  This function returns a tree without enclosing `Module` or `Expr` nodes.

  Args:
    src_string: The source code to parse.
    return_expr: Whether or not to return a containing expression. This can be
        set to `True` if the result is to be part of a series of statements.

  Returns:
    An AST of the given source code.

  """
  node = parse_string(src_string)
  body = node.body
  if len(body) == 1:
    if isinstance(body[0], gast.Expr) and not return_expr:
      out = body[0].value
    else:
      out = body[0]
  else:
    out = node
  return out


def unquote(node):
  """Go from an AST to source code."""
  return to_source(node).strip()
