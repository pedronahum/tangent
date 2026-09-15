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
"""The lowered frontend IR, named explicitly.

Tangent has no separate IR datatype: the "IR" is the restricted, post-ANF gast
sublanguage that the frontend pass pipeline produces - single-exit, no
classes/lambdas/comprehensions/break/continue, calls resolved, A-normal form
(see `tangent/passes.py`, `tangent/verify.py`, and
`docs/development/explicit-ir-investigation.md`).

`IRModule` is a thin marker that names that boundary. `passes.run_passes`
returns one, so the frontend's *output type is explicit* rather than "a gast
Module that happens to satisfy some invariants," and it records which pass
invariants the payload is known to satisfy. The differentiation transforms
consume `ir.module` (still gast today). This is the first, deliberately
non-invasive step toward pass and AD signatures that read `IRModule -> IRModule`
end to end; the payload staying gast keeps the source round-trip and the
readable-Python codegen unchanged.
"""

from __future__ import absolute_import

import gast


class IRModule(object):
    """A gast Module lowered to the frontend IR contract.

    Attributes:
      module: The lowered gast `Module`.
      mode: 'forward' or 'reverse' - the pass set that produced it.
      satisfied: frozenset of pass names whose IR invariant holds for `module`
        (the passes that ran, per `passes.get_passes(mode)`).
    """

    __slots__ = ('module', 'mode', 'satisfied')

    def __init__(self, module, mode, satisfied=frozenset()):
        if not isinstance(module, gast.Module):
            raise TypeError('IRModule wraps a gast.Module, got %s' % type(module).__name__)
        self.module = module
        self.mode = mode
        self.satisfied = frozenset(satisfied)

    @property
    def function(self):
        """The single lowered FunctionDef the AD transform differentiates."""
        for node in self.module.body:
            if isinstance(node, gast.FunctionDef):
                return node
        raise ValueError('IRModule has no top-level FunctionDef')

    def verify(self):
        """Re-check every registered invariant this IR is expected to satisfy.

        Raises `verify.IRInvariantError` on a violation. `verify_after` is a
        no-op for a pass with no registered invariant, so this checks exactly
        the contracted subset of `satisfied`.
        """
        from tangent import verify

        for pass_name in self.satisfied:
            verify.verify_after(pass_name, self.module)
        return self

    def __repr__(self):
        return 'IRModule(mode=%r, satisfied=%d passes)' % (self.mode, len(self.satisfied))
