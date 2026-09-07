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
"""Functions which perform compiler-style optimizations on the AST."""
from __future__ import absolute_import
from collections import defaultdict
import gast

from tangent import annotate
from tangent import annotations as anno
from tangent import cfg
from tangent import transformers
from tangent import utils


def fixed_point(f):

  def _fp(node):
    while True:
      # Use gast.dump instead of to_source to avoid gast_to_ast conversion issues
      # This is faster and avoids Python 3.8+ compatibility issues with type_comment
      import gast
      before = gast.dump(node)
      node = f(node)
      after = gast.dump(node)
      if before == after:
        break
    return node

  return _fp


@fixed_point
def optimize(node):
  """Perform a series of optimization passes.

  This function performs a series of optimizations (dead code elimination,
  constant folding, variable folding) on the given AST.
  It optimizes the code repeatedly until reaching a fixed point. The fixed
  point is determine roughly by checking whether the number of lines of
  generated source code changed after the latest pass.

  Args:
    node: The AST to optimize.
  Returns:
    The optimized AST.
  """
  # Phase 1: Basic optimizations
  node = constant_folding(node)  # Fold constants first (may enable more DCE)
  node = dead_code_elimination(node)  # Remove simple dead code
  node = assignment_propagation(node)  # Propagate single-use assignments

  # These create a positive feedback loop:
  # - constant_folding may create dead assignments
  # - dead_code_elimination removes them
  # - assignment_propagation creates more opportunities
  # - repeat until fixed point

  return node


def optimize_with_advanced_dce(node, requested_grads=None, verbose=0):
  """Enhanced optimization pipeline with advanced DCE.

  This combines Tangent's standard optimizations with the advanced DCE
  (activity analysis + control flow) for multiplicative benefits.

  Optimization order:
  1. Standard optimizations (constant folding, basic DCE, assignment propagation)
  2. Advanced DCE (activity analysis + control flow-aware)
  3. Standard optimizations again (to clean up after advanced DCE)

  Args:
    node: The AST to optimize
    requested_grads: List of parameter names for gradient computation (optional)
    verbose: Verbosity level

  Returns:
    The optimized AST
  """
  # Phase 1: Standard optimizations (fixed-point)
  if verbose >= 2:
    print("[Optimization] Phase 1: Standard optimizations")
  node = optimize(node)

  # Phase 2: Advanced DCE (if requested gradients provided)
  if requested_grads is not None:
    if verbose >= 2:
      print(f"[Optimization] Phase 2: Advanced DCE for {requested_grads}")
    try:
      from tangent.optimizations.dce import apply_dce
      # Apply advanced DCE to the gradient function. In split motion the
      # module holds [forward, backward] so the gradient code lives in the
      # LAST function; in joint motion there is a single combined function,
      # which is also the last one. body[0] would hit the primal in split
      # mode and strip the tape pushes the adjoint pops rely on.
      if hasattr(node, 'body') and len(node.body) > 0:
        node.body[-1] = apply_dce(node.body[-1], requested_grads, verbose)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: Advanced DCE failed: {e}")

  # Phase 3: Standard optimizations again (fixed-point)
  # Advanced DCE may create new opportunities for basic optimizations
  if verbose >= 2:
    print("[Optimization] Phase 3: Post-DCE cleanup")
  node = optimize(node)

  return node


def optimize_with_symbolic(node, requested_grads=None, enable_cse=True,
                           enable_algebraic=True, enable_strength_reduction=True,
                           verbose=0):
  """Enhanced optimization pipeline with symbolic optimizations.

  This combines Tangent's standard optimizations with:
  - Strength Reduction (expensive ops → cheap ops)
  - Common Subexpression Elimination (CSE)
  - Algebraic Simplification (using SymPy)
  - Advanced DCE

  Optimization order:
  1. Standard optimizations (constant folding, basic DCE, assignment propagation)
  2. Strength Reduction (x**2 → x*x, x/const → x*(1/const))
  3. CSE (reduces redundant computations, benefits from strength reduction)
  4. Algebraic Simplification (applies mathematical identities)
  5. Advanced DCE (removes unused code)
  6. Standard optimizations again (final cleanup)

  Args:
    node: The AST to optimize
    requested_grads: List of parameter names for gradient computation (optional)
    enable_strength_reduction: Whether to enable Strength Reduction
    enable_cse: Whether to enable Common Subexpression Elimination
    enable_algebraic: Whether to enable Algebraic Simplification
    verbose: Verbosity level

  Returns:
    The optimized AST
  """
  # Phase 1: Standard optimizations (fixed-point)
  if verbose >= 2:
    print("[Optimization] Phase 1: Standard optimizations")
  node = optimize(node)

  # Phase 2: Strength Reduction (before CSE so CSE can optimize the results)
  if enable_strength_reduction:
    if verbose >= 2:
      print("[Optimization] Phase 2: Strength Reduction")
    try:
      from tangent.optimizations.strength_reduction import apply_strength_reduction
      # Apply strength reduction to each function
      if hasattr(node, 'body') and len(node.body) > 0:
        for i, func in enumerate(node.body):
          if isinstance(func, gast.FunctionDef):
            if verbose >= 3:
              print(f"[Optimization]   - Applying strength reduction to {func.name}")
            node.body[i] = apply_strength_reduction(func)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: Strength reduction failed: {e}")

  # Phase 3: Common Subexpression Elimination
  if enable_cse:
    if verbose >= 2:
      print("[Optimization] Phase 3: Common Subexpression Elimination")
    try:
      from tangent.optimizations.cse import apply_cse
      # Apply CSE to each function in the module
      if hasattr(node, 'body') and len(node.body) > 0:
        for i, func in enumerate(node.body):
          if isinstance(func, gast.FunctionDef):
            if verbose >= 3:
              print(f"[Optimization]   - Applying CSE to {func.name}")
            node.body[i] = apply_cse(func)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: CSE failed: {e}")

  # Phase 4: Algebraic Simplification
  if enable_algebraic:
    if verbose >= 2:
      print("[Optimization] Phase 4: Algebraic Simplification")
    try:
      from tangent.optimizations.algebraic_simplification import apply_algebraic_simplification
      # Apply algebraic simplification to each function
      if hasattr(node, 'body') and len(node.body) > 0:
        for i, func in enumerate(node.body):
          if isinstance(func, gast.FunctionDef):
            if verbose >= 3:
              print(f"[Optimization]   - Applying algebraic simplification to {func.name}")
            node.body[i] = apply_algebraic_simplification(func)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: Algebraic simplification failed: {e}")

  # Phase 5: Advanced DCE (if requested gradients provided)
  if requested_grads is not None:
    if verbose >= 2:
      print(f"[Optimization] Phase 5: Advanced DCE for {requested_grads}")
    try:
      from tangent.optimizations.dce import apply_dce
      # Apply advanced DCE to the gradient function (the last one in the
      # module - see optimize_with_advanced_dce for why body[0] is wrong in
      # split motion).
      if hasattr(node, 'body') and len(node.body) > 0:
        node.body[-1] = apply_dce(node.body[-1], requested_grads, verbose)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: Advanced DCE failed: {e}")

  # Phase 6: Standard optimizations again (fixed-point)
  # Symbolic optimizations may create new opportunities for basic optimizations
  if verbose >= 2:
    print("[Optimization] Phase 6: Post-symbolic cleanup")
  node = optimize(node)

  return node


# Names of the tape primitives. A `push`-kind call records a value on the
# tape; a `pop`-kind call consumes it. `push_stack`/`pop_stack` do the same
# for sub-stacks (function calls); their pairing rules are identical.
_TAPE_PUSH_NAMES = frozenset(('push', 'push_stack'))
_TAPE_POP_NAMES = frozenset(('pop', 'pop_stack'))


def _tape_call_kind(call):
  """Classify a Call node as a tape operation.

  Recognizes both the textual form Tangent generates (``tangent.push(...)``
  etc.) and calls whose resolved `func` annotation is one of the tape
  primitives (the same information `annotate.find_stacks` uses).

  Args:
    call: A `gast.Call` node.

  Returns:
    'push', 'pop', or None if the call is not a tape operation.
  """
  name = None
  func = call.func
  if (isinstance(func, gast.Attribute) and
      isinstance(func.value, gast.Name) and func.value.id == 'tangent'):
    name = func.attr
  fn_handle = anno.getanno(call, 'func', False)
  if fn_handle:
    for candidate in ('push', 'pop', 'push_stack', 'pop_stack'):
      if fn_handle is getattr(utils, candidate):
        name = candidate
  if name in _TAPE_PUSH_NAMES:
    return 'push'
  if name in _TAPE_POP_NAMES:
    return 'pop'
  return None


def _tape_op_id(call):
  """Return the string op id of a tape call, or None if it is not static."""
  if not call.args:
    return None
  op_id_node = call.args[-1]
  # gast.Constant replaced gast.Str; keep the `.s` fallback for old gast.
  if isinstance(op_id_node, gast.Constant):
    value = op_id_node.value
  else:
    value = getattr(op_id_node, 's', None)
  return value if isinstance(value, str) else None


class _TapeOpCollector(gast.NodeVisitor):
  """Collect tape operations as (kind, op_id, statement, function) in
  program order. Program order of the statement list is execution order for
  the straight-line joint-motion code the pairing below relies on."""

  def __init__(self):
    self.ops = []
    self._stmts = []
    self._func = None
    self._func_counter = 0

  def visit(self, node):
    is_function = isinstance(node, gast.FunctionDef)
    is_stmt = isinstance(node, gast.stmt)
    if is_function:
      self._func_counter += 1
      enclosing_func = self._func
      self._func = self._func_counter
    if is_stmt:
      self._stmts.append(node)
    if isinstance(node, gast.Call):
      kind = _tape_call_kind(node)
      if kind:
        stmt = self._stmts[-1] if self._stmts else None
        self.ops.append((kind, _tape_op_id(node), stmt, self._func))
    self.generic_visit(node)
    if is_stmt:
      self._stmts.pop()
    if is_function:
      self._func = enclosing_func


def _tape_pairings(node):
  """Pair tape pushes with the pops that consume them.

  Pushes and pops are paired by their `op_id` argument. In first-order
  generated code every op id is unique, so an id appearing once as a push and
  once as a pop identifies a pair unambiguously. Differentiating gradient
  code AGAIN (higher-order derivatives) duplicates op ids: the primal of the
  new gradient re-executes the old push and pop, and the adjoint mirrors them
  (the adjoint of a push is a pop and vice versa), so one op id then names
  several distinct runtime pairs. `annotate.find_stacks` keeps only the
  last-seen push/pop per op id, so relying on its annotations makes dead code
  elimination remove a pop together with the WRONG push, crossing the tape's
  dataflow and corrupting second derivatives.

  Pairing rules, per op id:
    - exactly one push and one pop anywhere in the AST: pair them (ordinary
      first-order code, including split motion, where the push lives in the
      primal function and the pop in the adjoint function);
    - several pushes/pops, all inside one function: pair them like
      parentheses in program order (LIFO). This matches runtime order for
      joint-motion higher-order code, where the primal section pushes and
      pops an id before the adjoint section does so again;
    - anything else (dynamic op ids, unmatched pops, duplicated ids across
      functions): the operations become barriers that must not be removed,
      because balanced removal cannot be established.

  Args:
    node: The AST to analyze.

  Returns:
    A tuple `(pop_to_push, barriers)` where `pop_to_push` maps a pop
    statement to the push statement whose value it consumes, and `barriers`
    is a set of tape statements that must not be removed.
  """
  collector = _TapeOpCollector()
  collector.visit(node)
  pop_to_push = {}
  barriers = set()
  by_id = defaultdict(list)
  for kind, op_id, stmt, func in collector.ops:
    if stmt is None:
      continue
    if op_id is None:
      barriers.add(stmt)
      continue
    by_id[op_id].append((kind, stmt, func))
  for op_id, entries in by_id.items():
    stmts = [entry[1] for entry in entries]
    if len(set(stmts)) != len(stmts):
      # One statement holds several tape calls for this op id; don't touch.
      barriers.update(stmts)
      continue
    pushes = [entry for entry in entries if entry[0] == 'push']
    pops = [entry for entry in entries if entry[0] == 'pop']
    if len(pushes) == 1 and len(pops) == 1:
      pop_to_push[pops[0][1]] = pushes[0][1]
      continue
    if len(set(entry[2] for entry in entries)) != 1:
      # Duplicated op ids spanning several functions: program order across
      # function boundaries need not match execution order, so no confident
      # pairing exists.
      barriers.update(stmts)
      continue
    # LIFO (balanced-parentheses) pairing in program order.
    unmatched_pushes = []
    local_pairs = {}
    consistent = True
    for kind, stmt, _ in entries:
      if kind == 'push':
        unmatched_pushes.append(stmt)
      elif unmatched_pushes:
        local_pairs[stmt] = unmatched_pushes.pop()
      else:
        consistent = False
        break
    if consistent:
      pop_to_push.update(local_pairs)
      # Trailing pushes without a visible pop stay put.
      barriers.update(unmatched_pushes)
    else:
      barriers.update(stmts)
  return pop_to_push, barriers


@fixed_point
def dead_code_elimination(node):
  """Perform a simple form of dead code elimination on a Python AST.

  This method performs reaching definitions analysis on all function
  definitions. It then looks for the definition of variables that are not used
  elsewhere and removes those definitions.

  This function takes into consideration push and pop statements: tape state
  stays consistent only under BALANCED removal, so a pop statement is removed
  together with the push statement it consumes (established by
  `_tape_pairings`), and a tape operation whose counterpart cannot be
  established is never removed. Note that this *requires dead code
  elimination to be performed on the primal and adjoint simultaneously*.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  # Find all statements that are inside exception handlers - these should not be removed
  # because they may execute when an exception is raised
  statements_in_handlers = set()
  for try_node in gast.walk(node):
    if isinstance(try_node, gast.Try):
      for handler in try_node.handlers:
        for stmt in gast.walk(handler):
          if isinstance(stmt, gast.stmt):
            statements_in_handlers.add(stmt)

  # Pair tape pushes with the pops that consume them. Only balanced pairs may
  # be removed; `tape_barriers` holds tape operations with no established
  # counterpart, which must stay.
  pop_to_push, tape_barriers = _tape_pairings(node)
  tape_stmts = (set(pop_to_push) | set(pop_to_push.values()) | tape_barriers)

  to_remove = set(def_[1] for def_ in annotate.unused(node)
                  if not isinstance(def_[1], (gast.arguments, gast.For))
                  and def_[1] not in statements_in_handlers
                  and def_[1] not in tape_barriers
                  and not anno.getanno(def_[1], 'tangent_keep', False))
  for n in list(to_remove):
    if n in pop_to_push:
      push = pop_to_push[n]
      if (push in statements_in_handlers or
          anno.getanno(push, 'tangent_keep', False)):
        # The push must stay, so its pop must stay too: removing only one
        # half of a pair would unbalance the tape.
        to_remove.discard(n)
      else:
        to_remove.add(push)
    elif n not in tape_stmts:
      # Fallback for tape calls the structural analysis did not recognize:
      # chase the pairing annotations left by `annotate.find_stacks`. (For
      # recognized tape statements those annotations may be stale - op ids
      # duplicated by higher-order differentiation make find_stacks keep only
      # the last-seen counterpart - so they are only trusted here.)
      for succ in gast.walk(n):
        if anno.getanno(succ, 'push', False):
          to_remove.add(anno.getanno(succ, 'push'))
  # Never remove statements marked keep-alive (e.g. varargs pack/unpack),
  # even if they were pulled in via a push annotation.
  to_remove = set(n for n in to_remove
                  if not anno.getanno(n, 'tangent_keep', False))
  transformers.Remove(to_remove).visit(node)
  anno.clearanno(node)
  return node


class ReadCounts(gast.NodeVisitor):
  """Find the number of times that each definition is used.

  Requires `ReachingDefinitions` analysis to have been performed.
  """

  def __init__(self):
    self.n_read = defaultdict(int)

  def visit(self, node):
    if anno.hasanno(node, 'definitions_in'):
      self.reaching_definitions = anno.getanno(node, 'definitions_in')
    super(ReadCounts, self).visit(node)
    if anno.hasanno(node, 'definitions_in'):
      self.reaching_definitions = None

  def visit_Name(self, node):
    if isinstance(node.ctx, gast.Load) and self.reaching_definitions is not None:
      for def_ in self.reaching_definitions:
        if def_[0] == node.id:
          self.n_read[def_[1]] += 1


def read_counts(node):
  """Check how many times a variable definition was used.

  Args:
    node: An AST to analyze.

  Returns:
    A dictionary from assignment nodes to the number of times the assigned to
        variable was used.
  """
  cfg.forward(node, cfg.ReachingDefinitions())

  rc = ReadCounts()
  rc.visit(node)
  return rc.n_read


@fixed_point
def assignment_propagation(node):
  """Perform assignment propagation.

  Assignment propagation is not a compiler optimization as much as a
  readability optimization. If a variable name is used only once, it gets
  renamed when possible e.g. `y = x; z = y` will become `z = x`.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  n_reads = read_counts(node)

  to_remove = []
  for succ in gast.walk(node):
    # We found an assignment of the form a = b
    # - Left-hand side is a Name, right-hand side is a Name.
    if (isinstance(succ, gast.Assign) and isinstance(succ.value, gast.Name) and
        len(succ.targets) == 1 and isinstance(succ.targets[0], gast.Name)):
      rhs_name = succ.value.id
      # We now find all the places that b was defined
      rhs_defs = [def_[1] for def_ in anno.getanno(succ, 'definitions_in')
                  if def_[0] == rhs_name]
      # If b was defined in only one place (not an argument), and wasn't used
      # anywhere else but in a == b, and was defined as b = x, then we can fold
      # the statements
      if (len(rhs_defs) == 1 and isinstance(rhs_defs[0], gast.Assign) and
          n_reads[rhs_defs[0]] == 1 and
          isinstance(rhs_defs[0].value, gast.Name) and
          isinstance(rhs_defs[0].targets[0], gast.Name)):
        # Mark rhs_def for deletion
        to_remove.append(rhs_defs[0])
        # Propagate the definition
        succ.value = rhs_defs[0].value

  # Remove the definitions we folded
  transformers.Remove(to_remove).visit(node)
  anno.clearanno(node)
  return node


class ConstantFolding(gast.NodeTransformer):

  def visit_BinOp(self, node):
    self.generic_visit(node)
    left_val = node.left
    right_val = node.right
    # gast.Constant replaces gast.Num in gast >= 0.3.0
    left_is_num = isinstance(left_val, gast.Constant) and isinstance(left_val.value, (int, float))
    right_is_num = isinstance(right_val, gast.Constant) and isinstance(right_val.value, (int, float))

    if isinstance(node.op, gast.Mult):
      if left_is_num and right_is_num:
        return gast.Constant(value=left_val.value * right_val.value, kind=None)
      if left_is_num:
        if left_val.value == 0:
          return gast.Constant(value=0, kind=None)
        elif left_val.value == 1:
          return right_val
      if right_is_num:
        if right_val.value == 0:
          return gast.Constant(value=0, kind=None)
        elif right_val.value == 1:
          return left_val
    elif isinstance(node.op, gast.Add):
      if left_is_num and right_is_num:
        return gast.Constant(value=left_val.value + right_val.value, kind=None)
      if left_is_num and left_val.value == 0:
        return right_val
      if right_is_num and right_val.value == 0:
        return left_val
    elif isinstance(node.op, gast.Sub):
      if left_is_num and right_is_num:
        return gast.Constant(value=left_val.value - right_val.value, kind=None)
      if left_is_num and left_val.value == 0:
        return gast.UnaryOp(op=gast.USub(), operand=right_val)
      if right_is_num and right_val.value == 0:
        return left_val
    elif isinstance(node.op, gast.Div):
      if left_is_num and right_is_num:
        return gast.Constant(value=left_val.value / right_val.value, kind=None)
      if right_is_num and right_val.value == 1:
        return left_val
    elif isinstance(node.op, gast.Pow):
      if left_is_num and right_is_num:
        return gast.Constant(value=left_val.value ** right_val.value, kind=None)
      if left_is_num:
        if left_val.value == 0:
          return gast.Constant(value=0, kind=None)
        elif left_val.value == 1:
          return gast.Constant(value=1, kind=None)
      if right_is_num:
        if right_val.value == 0:
          return gast.Constant(value=1, kind=None)
        elif right_val.value == 1:
          return left_val
    return node


@fixed_point
def constant_folding(node):
  """Perform constant folding.

  This function also uses arithmetic identities (like multiplying with one or
  adding zero) to simplify statements. However, it doesn't inline constants in
  expressions, so the simplifications don't propagate.

  Args:
    node: The AST to optimize.

  Returns:
    The optimized AST.
  """
  f = ConstantFolding()
  return f.visit(node)
