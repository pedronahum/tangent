"""
Straight-line coarsening for Tangent (prototype).

Background
----------
Tangent's reverse mode differentiates the primal one primitive operation at a
time: every ``+``, ``*``, ``sin`` ... produces its own adjoint statement and,
for values needed later, a tape push/pop pair.  For a long straight-line
stretch of arithmetic this per-op expansion is the dominant cost, and it also
hides simplification opportunities that only become visible across several
operations (e.g. ``sin(x)**2 + cos(x)**2`` collapsing to ``1``).

The coarsening idea (Shen et al., "Integrating symbolic and algorithmic
automatic differentiation") is to treat a whole straight-line segment as a
single symbolic function, differentiate it once symbolically, and emit the
resulting vector-Jacobian product directly.  This skips the intermediate
adjoints and the tape traffic for the segment's internal values, and exposes
cross-operation simplification to the symbolic engine.

This module is a *prototype* of that idea, built as an extension of
``tangent.optimizations.algebraic_simplification``: it reuses that module's
``ASTToSymPyConverter`` / ``SymPyToASTConverter`` round-trip and its SymPy
based simplification.  Given a straight-line primal function it produces the
coarsened reverse-mode adjoint.  It is not yet wired into ``tangent.grad``;
integration into the reverse-mode pipeline is the natural follow-up.

Scope (deliberately small for a prototype)
-------------------------------------------
- Only straight-line segments: a body made of plain single-target assignments
  followed by a single ``return``.  No control flow, no augmented assignment,
  no subscript/attribute targets, no calls other than the elementwise
  primitives the SymPy converters understand.
- Scalar symbolic math (SymPy).  The converters operate element-wise, which is
  exactly the regime the prototype targets.
"""

import ast
import gast
import sympy as sp

from tangent.optimizations.algebraic_simplification import (
    ASTToSymPyConverter,
    SymPyToASTConverter,
)


class NotStraightLineError(Exception):
  """Raised when a function body cannot be treated as a straight-line segment."""


# Elementwise primitives coarsening can handle end-to-end. The constraint is
# tighter than "the SymPy converters know the function": both the forward lift
# (ASTToSymPyConverter) and the lowering of the *derivative*
# (SymPyToASTConverter) must succeed. That rules out ops whose derivative
# reintroduces a function the lowering cannot emit - e.g. tanh' = 1 - tanh^2,
# sinh' = cosh, abs' = sign (and SymPy's derivative of Abs over a complex
# symbol is not lowerable) - so those are excluded even though they lift fine.
# The inverse-trig ops use NumPy's spelling (np.arcsin, ...); their
# derivatives lower cleanly (arcsin' = 1/sqrt(1-x^2), arctan' = 1/(1+x^2)).
# Empirically pinned by
# tests/test_coarsening.py::test_elementwise_support_set_is_exact.
_SUPPORTED_ELEMENTWISE = frozenset((
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt',
    'arcsin', 'arccos', 'arctan',
))


def _callee_name(node):
  """Return the simple name of a call target, or None."""
  func = node.func
  if isinstance(func, (ast.Name, gast.Name)):
    return func.id
  if isinstance(func, (ast.Attribute, gast.Attribute)):
    return func.attr
  return None


class StraightLineCoarsener:
  """Coarsen a straight-line primal segment into a symbolic adjoint.

  Usage:
      coarsener = StraightLineCoarsener()
      adjoint = coarsener.coarsen(primal_func_ast)  # None if not coarsenable
  """

  def __init__(self, simplify=True):
    # One converter instance is reused across the whole segment so that a
    # given variable name always maps to the same SymPy symbol.
    self.ast_to_sympy = ASTToSymPyConverter()
    self.sympy_to_ast = SymPyToASTConverter()
    self.simplify = simplify
    # Diagnostics
    self.inputs = []
    self.output_expr = None

  # -- segment validation / inlining ---------------------------------------

  def _validate_and_collect(self, func_ast):
    """Check the body is a straight-line segment and return (assigns, return)."""
    body = func_ast.body
    if not body:
      raise NotStraightLineError('empty body')

    if not isinstance(body[-1], (ast.Return, gast.Return)):
      raise NotStraightLineError('segment must end with a return')
    if body[-1].value is None:
      raise NotStraightLineError('segment must return a value')

    assigns = body[:-1]
    for stmt in assigns:
      if not isinstance(stmt, (ast.Assign, gast.Assign)):
        raise NotStraightLineError(
            'non-assignment statement in segment: %s' % type(stmt).__name__)
      if len(stmt.targets) != 1:
        raise NotStraightLineError('multiple assignment targets')
      target = stmt.targets[0]
      if not isinstance(target, (ast.Name, gast.Name)):
        raise NotStraightLineError(
            'non-Name assignment target (subscript/attribute/tuple writes are '
            'not pure)')
      self._validate_expr(stmt.value)

    self._validate_expr(body[-1].value)
    return assigns, body[-1]

  def _validate_expr(self, node):
    """Reject expressions that are not pure elementwise arithmetic."""
    # Attribute nodes are only legal as the callee of a supported call
    # (e.g. ``np.sin``); attribute access on data (``x.attr``) is rejected.
    call_callees = set()
    for sub in gast.walk(node):
      if isinstance(sub, (ast.Call, gast.Call)):
        call_callees.add(id(sub.func))

    for sub in gast.walk(node):
      if isinstance(sub, (ast.Call, gast.Call)):
        name = _callee_name(sub)
        if name is None or name not in _SUPPORTED_ELEMENTWISE:
          raise NotStraightLineError(
              'unsupported call in segment: %s' % name)
        if sub.keywords:
          raise NotStraightLineError('keyword arguments in segment call')
      elif isinstance(sub, (ast.Attribute, gast.Attribute)):
        if id(sub) not in call_callees:
          raise NotStraightLineError('attribute access in segment')
      elif isinstance(sub, (ast.IfExp, gast.IfExp, ast.Lambda, gast.Lambda,
                            ast.ListComp, gast.ListComp, ast.List, gast.List,
                            ast.Tuple, gast.Tuple, ast.Dict, gast.Dict,
                            ast.Subscript, gast.Subscript)):
        raise NotStraightLineError(
            'unsupported construct in segment: %s' % type(sub).__name__)

  def _input_names(self, func_ast):
    names = []
    for arg in func_ast.args.args:
      name = getattr(arg, 'id', None) or getattr(arg, 'arg', None)
      if name is None:
        raise NotStraightLineError('cannot read parameter name')
      names.append(name)
    if func_ast.args.vararg or func_ast.args.kwarg or func_ast.args.kwonlyargs:
      raise NotStraightLineError('varargs/kwargs not supported')
    return names

  def _inline(self, assigns, return_stmt):
    """Symbolically inline the assignments, returning the output expression.

    The result is a SymPy expression over the segment's input symbols only;
    every intermediate has been substituted away.
    """
    env = {}  # intermediate name -> sympy expression
    for stmt in assigns:
      target = stmt.targets[0].id
      expr = self.ast_to_sympy.convert(stmt.value)
      if expr is None:
        raise NotStraightLineError(
            'cannot lift RHS of "%s" to SymPy' % target)
      for name, value in env.items():
        expr = expr.subs(sp.Symbol(name), value)
      env[target] = expr

    out = self.ast_to_sympy.convert(return_stmt.value)
    if out is None:
      raise NotStraightLineError('cannot lift return value to SymPy')
    for name, value in env.items():
      out = out.subs(sp.Symbol(name), value)
    return out

  # -- adjoint construction -------------------------------------------------

  def _seed_name(self, func_ast, return_stmt, input_names):
    """Pick a name for the incoming adjoint of the returned value."""
    ret = return_stmt.value
    base = ret.id if isinstance(ret, (ast.Name, gast.Name)) else 'ret'
    candidate = 'b' + base
    reserved = set(input_names) | {'b' + n for n in input_names}
    if candidate in reserved:
      candidate = '_bret'
    return candidate

  def coarsen(self, func_ast, seed_name=None):
    """Return a coarsened adjoint FunctionDef, or None if not coarsenable.

    The adjoint has signature
        d<name>(<inputs...>, <seed>) -> (b<input> , ...)
    where ``<seed>`` is the adjoint of the primal's returned value and each
    ``b<input> = seed * d(output)/d(input)`` is computed symbolically.
    """
    try:
      assigns, return_stmt = self._validate_and_collect(func_ast)
      input_names = self._input_names(func_ast)
      output_expr = self._inline(assigns, return_stmt)
    except NotStraightLineError:
      return None

    self.inputs = list(input_names)
    self.output_expr = output_expr

    if seed_name is None:
      seed_name = self._seed_name(func_ast, return_stmt, input_names)
    seed = sp.Symbol(seed_name)

    # Differentiate the whole segment symbolically w.r.t. each input and lower
    # the resulting VJP term back to an AST expression.
    lines = []
    adjoint_names = []
    for name in input_names:
      deriv = sp.diff(output_expr, sp.Symbol(name))
      if self.simplify:
        deriv = sp.simplify(deriv)
      term = seed * deriv
      term_ast = self.sympy_to_ast.convert(term)
      if term_ast is None:
        # Could not lower this derivative; bail out on the whole segment so we
        # never emit a partially-coarsened (and thus incorrect) adjoint.
        return None
      adj_name = 'b' + name
      adjoint_names.append(adj_name)
      lines.append('  %s = %s' % (adj_name, gast.unparse(term_ast)))

    if len(input_names) == 1:
      ret_src = '  return %s' % adjoint_names[0]
    else:
      ret_src = '  return %s' % ', '.join(adjoint_names)

    params = ', '.join(input_names + [seed_name])
    src = 'def d%s(%s):\n%s\n%s' % (func_ast.name, params,
                                     '\n'.join(lines), ret_src)
    adjoint_ast = gast.parse(src).body[0]
    gast.fix_missing_locations(adjoint_ast)
    return adjoint_ast


def apply_coarsening(func_ast, config=None):
  """Entry point: coarsen a straight-line primal into its symbolic adjoint.

  Args:
    func_ast: A straight-line primal FunctionDef.
    config: Optional dict; 'simplify' (default True) toggles SymPy
      simplification of the symbolic derivatives.

  Returns:
    The coarsened adjoint FunctionDef, or None if the function is not a
    coarsenable straight-line segment.
  """
  config = config or {}
  coarsener = StraightLineCoarsener(simplify=config.get('simplify', True))
  return coarsener.coarsen(func_ast, seed_name=config.get('seed_name'))
