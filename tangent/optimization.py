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
from tangent import quoting
from tangent import transformers


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
      # Apply advanced DCE to the gradient function
      if hasattr(node, 'body') and len(node.body) > 0:
        node.body[0] = apply_dce(node.body[0], requested_grads)
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
      # Apply advanced DCE to the gradient function
      if hasattr(node, 'body') and len(node.body) > 0:
        node.body[0] = apply_dce(node.body[0], requested_grads)
    except Exception as e:
      if verbose >= 1:
        print(f"[Optimization] Warning: Advanced DCE failed: {e}")

  # Phase 6: Standard optimizations again (fixed-point)
  # Symbolic optimizations may create new opportunities for basic optimizations
  if verbose >= 2:
    print("[Optimization] Phase 6: Post-symbolic cleanup")
  node = optimize(node)

  return node


@fixed_point
def dead_code_elimination(node):
  """Perform a simple form of dead code elimination on a Python AST.

  This method performs reaching definitions analysis on all function
  definitions. It then looks for the definition of variables that are not used
  elsewhere and removes those definitions.

  This function takes into consideration push and pop statements; if a pop
  statement is removed, it will also try to remove the accompanying push
  statement. Note that this *requires dead code elimination to be performed on
  the primal and adjoint simultaneously*.

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

  to_remove = set(def_[1] for def_ in annotate.unused(node)
                  if not isinstance(def_[1], (gast.arguments, gast.For))
                  and def_[1] not in statements_in_handlers)
  for n in list(to_remove):
    for succ in gast.walk(n):
      if anno.getanno(succ, 'push', False):
        to_remove.add(anno.getanno(succ, 'push'))
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
