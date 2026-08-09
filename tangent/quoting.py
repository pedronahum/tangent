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
import ast
import copy
import inspect
import textwrap

import gast

from tangent import annotations as anno


class TangentParseError(SyntaxError):
  pass


def _has_comments(node):
  """True if any node in the tree carries a Tangent comment annotation."""
  return any(anno.hasanno(n, 'comment') for n in ast.walk(node))


def _comment(node):
  """Return (location, text) for a node's comment, or (None, None)."""
  if not anno.hasanno(node, 'comment'):
    return None, None
  comment = anno.getanno(node, 'comment')
  if comment['location'] not in ('above', 'below', 'right'):
    raise TangentParseError('Only valid comment locations are '
                            'above, below, right')
  return comment['location'], comment['text']


def _emit_stmts(stmts, level, ind):
  lines = []
  for stmt in stmts:
    lines.extend(_emit_stmt(stmt, level, ind))
  return lines


def _emit_stmt(stmt, level, ind):
  """Emit one statement as source lines, injecting any comment annotation."""
  pad = ind * level
  loc, text = _comment(stmt)
  out = []
  if loc == 'above':
    out.append(pad + '# ' + text[:78])
  body = _emit_stmt_body(stmt, level, ind)
  if loc == 'right' and body:
    body[-1] = body[-1] + ' # ' + text
  out.extend(body)
  if loc == 'below':
    out.append(pad + '# ' + text[:78])
  return out


def _emit_stmt_body(stmt, level, ind):
  """Emit a statement without its comment. Compound statements are recursed
  into so that comments nested in their bodies are emitted too; everything
  else is delegated to ast.unparse."""
  pad = ind * level
  if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    return _emit_func_or_class(stmt, level, ind)
  if isinstance(stmt, ast.If):
    return _emit_if(stmt, level, ind, is_elif=False)
  if isinstance(stmt, (ast.For, ast.AsyncFor)):
    return _emit_for(stmt, level, ind)
  if isinstance(stmt, ast.While):
    return _emit_while(stmt, level, ind)
  if isinstance(stmt, (ast.With, ast.AsyncWith)):
    return _emit_with(stmt, level, ind)
  if isinstance(stmt, ast.Try):
    return _emit_try(stmt, level, ind)
  # Simple statements (and any exotic compound types Tangent does not
  # generate, e.g. match/except*) are emitted verbatim.
  return [pad + line for line in ast.unparse(stmt).split('\n')]


def _emit_func_or_class(stmt, level, ind):
  pad = ind * level
  lines = []
  for decorator in stmt.decorator_list:
    lines.append(pad + '@' + ast.unparse(decorator))
  # Reuse ast.unparse for the signature: unparse a copy with a trivial body
  # and no decorators, then take the header line.
  dummy = copy.copy(stmt)
  dummy.body = [ast.Pass()]
  dummy.decorator_list = []
  header = ast.unparse(dummy).split('\n')[0]
  lines.append(pad + header)
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  return lines


def _emit_if(stmt, level, ind, is_elif):
  pad = ind * level
  keyword = 'elif' if is_elif else 'if'
  lines = [pad + keyword + ' ' + ast.unparse(stmt.test) + ':']
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  if stmt.orelse:
    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
      elif_node = stmt.orelse[0]
      loc, text = _comment(elif_node)
      if loc == 'above':
        lines.append(pad + '# ' + text[:78])
      sub = _emit_if(elif_node, level, ind, is_elif=True)
      if loc == 'right' and sub:
        sub[-1] = sub[-1] + ' # ' + text
      lines.extend(sub)
      if loc == 'below':
        lines.append(pad + '# ' + text[:78])
    else:
      lines.append(pad + 'else:')
      lines.extend(_emit_stmts(stmt.orelse, level + 1, ind))
  return lines


def _emit_for(stmt, level, ind):
  pad = ind * level
  keyword = 'async for' if isinstance(stmt, ast.AsyncFor) else 'for'
  lines = [pad + keyword + ' ' + ast.unparse(stmt.target) + ' in ' +
           ast.unparse(stmt.iter) + ':']
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  if stmt.orelse:
    lines.append(pad + 'else:')
    lines.extend(_emit_stmts(stmt.orelse, level + 1, ind))
  return lines


def _emit_while(stmt, level, ind):
  pad = ind * level
  lines = [pad + 'while ' + ast.unparse(stmt.test) + ':']
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  if stmt.orelse:
    lines.append(pad + 'else:')
    lines.extend(_emit_stmts(stmt.orelse, level + 1, ind))
  return lines


def _emit_with(stmt, level, ind):
  pad = ind * level
  items = []
  for item in stmt.items:
    piece = ast.unparse(item.context_expr)
    if item.optional_vars is not None:
      piece += ' as ' + ast.unparse(item.optional_vars)
    items.append(piece)
  keyword = 'async with' if isinstance(stmt, ast.AsyncWith) else 'with'
  lines = [pad + keyword + ' ' + ', '.join(items) + ':']
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  return lines


def _emit_try(stmt, level, ind):
  pad = ind * level
  lines = [pad + 'try:']
  lines.extend(_emit_stmts(stmt.body, level + 1, ind))
  for handler in stmt.handlers:
    header = pad + 'except'
    if handler.type is not None:
      header += ' ' + ast.unparse(handler.type)
    if handler.name is not None:
      header += ' as ' + handler.name
    lines.append(header + ':')
    lines.extend(_emit_stmts(handler.body, level + 1, ind))
  if stmt.orelse:
    lines.append(pad + 'else:')
    lines.extend(_emit_stmts(stmt.orelse, level + 1, ind))
  if stmt.finalbody:
    lines.append(pad + 'finally:')
    lines.extend(_emit_stmts(stmt.finalbody, level + 1, ind))
  return lines


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

    # ast.unparse reads location info (e.g. for type comments); many Tangent
    # generated nodes carry none, so fill in defaults before unparsing.
    node = ast.fix_missing_locations(node_ast)

  # Fast path: no preserved comments, so ast.unparse can emit the whole tree.
  if not _has_comments(node):
    return ast.unparse(node)

  # Comment path: emit statement-by-statement so the preserved comments land
  # at the right place and indentation.
  if isinstance(node, ast.Module):
    lines = _emit_stmts(node.body, 0, indentation)
  elif isinstance(node, ast.stmt):
    lines = _emit_stmt(node, 0, indentation)
  else:
    # An expression cannot carry a statement comment; emit it directly.
    return ast.unparse(node)
  return '\n'.join(lines) + '\n'


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
