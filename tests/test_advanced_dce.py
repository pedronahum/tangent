# Copyright 2026 Tangent contributors
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
"""Unit tests for the advanced DCE (tangent/optimizations/dce.py).

These tests pin down the correctness invariants that the wide integration
suite exercises only indirectly:

1. Tape push/pop statements are never eliminated, and the variables they
   reference (the stack in particular) survive the backward slice.
2. In split motion the DCE targets the gradient function (the last def in
   the module), not the primal.
3. Def/use analysis of statements nested in control flow merges with the
   enclosing statement instead of clobbering it (e.g. an if-condition stays
   live when its branch defines a requested gradient).
4. Statements kept for their side effects (subscript/attribute writes) pull
   the variables they use into the slice.
"""
import textwrap

import gast
import pytest

from tangent import optimization
from tangent.optimizations.dce import apply_dce


def _parse_function(source):
  return gast.parse(textwrap.dedent(source)).body[0]


def _defined_names(func):
  """All names assigned anywhere in the function (including nested bodies)."""
  names = set()
  for node in gast.walk(func):
    if isinstance(node, gast.Assign):
      for target in node.targets:
        if isinstance(target, gast.Name):
          names.add(target.id)
  return names


def _is_tape_call(stmt, op_name):
  """True if stmt is `tangent.<op_name>(...)` as an expression statement."""
  if not isinstance(stmt, gast.Expr):
    return False
  call = stmt.value
  return (isinstance(call, gast.Call) and
          isinstance(call.func, gast.Attribute) and
          call.func.attr == op_name)


def _uses_tape_call(stmt, op_name):
  """True if stmt contains a call to tangent.<op_name> anywhere."""
  for node in gast.walk(stmt):
    if (isinstance(node, gast.Call) and
        isinstance(node.func, gast.Attribute) and
        node.func.attr == op_name):
      return True
  return False


def test_tape_operations_and_dead_code():
  """Pushes/pops survive; genuinely dead computations do not."""
  func = _parse_function("""
      def _dfdx(x, bx_seed):
          _stack = tangent.Stack()
          y = x * x
          tangent.push(_stack, y, '_abc')
          dead = y * 3.0
          y2 = tangent.pop(_stack, '_abc')
          by = bx_seed * y2
          return by
  """)

  result = apply_dce(func, ['x'])

  defined = _defined_names(result)

  # The dead computation is gone...
  assert 'dead' not in defined
  # ...but the tape operations, the stack and every value they need remain.
  assert any(_is_tape_call(stmt, 'push') for stmt in result.body)
  assert any(_uses_tape_call(stmt, 'pop') for stmt in result.body)
  assert '_stack' in defined
  assert 'y' in defined
  assert 'y2' in defined
  assert 'by' in defined


def test_targets_gradient_function_not_primal():
  """Split-motion modules hold [forward, backward]; the DCE must hit the
  backward function and leave the primal's tape pushes alone."""
  module = gast.parse(textwrap.dedent("""
      def fwd(x):
          _stack = tangent.Stack()
          y = x * x
          tangent.push(_stack, y, '_abc')
          return y

      def bwd(_stack, by):
          y = tangent.pop(_stack, '_abc')
          bx = by * y
          dead = bx + 99.0
          return bx
  """))

  result = optimization.optimize_with_advanced_dce(module, ['x'])

  fwd, bwd = result.body[0], result.body[-1]
  assert fwd.name == 'fwd'
  assert bwd.name == 'bwd'

  # The primal keeps its tape push and its computation...
  assert any(_is_tape_call(stmt, 'push') for stmt in fwd.body)
  assert 'y' in _defined_names(fwd)

  # ...while the adjoint loses dead code but keeps its pop.
  assert 'dead' not in _defined_names(bwd)
  assert any(_uses_tape_call(stmt, 'pop') for stmt in bwd.body)
  assert 'bx' in _defined_names(bwd)


def test_if_condition_survives_control_flow_analysis():
  """A condition whose branches define the requested gradient must not be
  eliminated (regression: nested def/use maps used to clobber it)."""
  func = _parse_function("""
      def _dfdx(bx_seed):
          cond = bx_seed > 0.0
          if cond:
              bx = bx_seed * 2.0
          else:
              bx = bx_seed
          return bx
  """)

  result = apply_dce(func, ['x'])

  defined = _defined_names(result)
  assert 'cond' in defined
  assert 'bx' in defined
  assert any(isinstance(stmt, gast.If) for stmt in result.body)


def test_subscript_write_keeps_its_operands():
  """x[i] = expr writes are side-effecting and kept unconditionally, so the
  definition of expr must survive the slice as well."""
  func = _parse_function("""
      def _dfdx(x, bx_seed):
          a_times_b = x * 2.0
          x[0] = a_times_b
          bx = bx_seed + 1.0
          return bx
  """)

  result = apply_dce(func, ['x'])

  defined = _defined_names(result)
  assert 'a_times_b' in defined
  assert any(
      isinstance(stmt, gast.Assign) and
      isinstance(stmt.targets[0], gast.Subscript)
      for stmt in result.body)
  assert 'bx' in defined


if __name__ == '__main__':
  assert not pytest.main([__file__])
