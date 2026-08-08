# Copyright 2018 Google Inc.
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
"""Desugar conditional expressions (ternaries) into if-statements.

    y = a if test else b
    # becomes
    if test:
        _ifexp0 = a
    else:
        _ifexp0 = b
    y = _ifexp0

Reverse mode differentiates conditional expressions directly, so this pass is
only used for forward mode, which supports if-statements but not the expression
form. Each conditional expression is replaced by a temporary and an if-statement
computing it is prepended to the enclosing statement. The prepended statement is
re-processed, so nested ternaries and ternaries in the test are handled within
the correct branch scope, and only the selected branch is evaluated.
"""
from __future__ import absolute_import

import gast

from tangent import transformers


class IfExpDesugarer(transformers.TreeTransformer):
  """Rewrite ``a if test else b`` expressions into if-statements."""

  def __init__(self):
    super(IfExpDesugarer, self).__init__()
    self._counter = 0

  def _fresh(self):
    name = '_ifexp%d' % self._counter
    self._counter += 1
    return name

  def visit_IfExp(self, node):
    name = self._fresh()

    def _assign(value):
      return gast.Assign(
          targets=[gast.Name(id=name, ctx=gast.Store(), annotation=None)],
          value=value)

    if_stmt = gast.If(test=node.test,
                      body=[_assign(node.body)],
                      orelse=[_assign(node.orelse)])
    # The prepended if-statement is itself re-visited, so a nested conditional in
    # a branch or in the test is desugared within the correct scope.
    self.prepend(if_stmt)
    return gast.Name(id=name, ctx=gast.Load(), annotation=None)


def desugar_ifexps(node):
  """Rewrite conditional expressions into if-statements."""
  node = IfExpDesugarer().visit(node)
  gast.fix_missing_locations(node)
  return node
