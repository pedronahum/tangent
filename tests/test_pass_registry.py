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
"""Tests for the frontend pass registry (tangent/passes.py)."""

import pytest

import tangent
from tangent import passes


# Snapshot of the resolved pass order per mode. If you change the pipeline,
# update this test deliberately: the ordering carries semantic constraints
# (see the pass docstrings in tangent/passes.py).
EXPECTED_REVERSE = [
    'sentinel_rename',
    'class_desugar',
    'lambda_desugar',
    'return_desugar',
    'chained_assign_desugar',
    'enumerate_desugar',
    'zip_desugar',
    # Unrolls constant-iterable comprehensions into literals and lowers
    # dynamic-iterable list comprehensions into indexed loops built on
    # tangent.list_append; unexpressible forms are rejected by the fence.
    'comprehension_desugar',
    'dict_method_desugar',
    'concat_desugar',
    # xs.append(v) / xs.pop() are rewritten into differentiable rebindings
    # (tangent.list_append / list_last / list_init) before call resolution.
    'list_method_desugar',
    # break/continue are lowered into guard flags (break converts the loop to
    # an indexed while) after the iterator desugarings, before resolution.
    'loop_exit_desugar',
    'checkpoint_annotation',
    'resolve_calls',
    'explicit_loop_indexes',
    'fence',
    'anf',
]

# Forward mode additionally lowers ternaries, after all other desugarings
# and before call resolution.
EXPECTED_FORWARD = EXPECTED_REVERSE[:13] + ['ifexp_desugar'] + EXPECTED_REVERSE[13:]


def test_reverse_pass_order():
    assert [p.name for p in passes.get_passes('reverse')] == EXPECTED_REVERSE


def test_forward_pass_order():
    assert [p.name for p in passes.get_passes('forward')] == EXPECTED_FORWARD


def test_ifexp_is_forward_only():
    (ifexp,) = [p for p in passes.get_passes('forward') if p.name == 'ifexp_desugar']
    assert ifexp.modes == frozenset(('forward',))


def _noop(node, ctx):
    """A do-nothing extension pass."""
    assert ctx.func is not None
    return node


def test_register_before_and_after():
    passes.register_pass(passes.Pass('ext_before', _noop), before='anf')
    passes.register_pass(passes.Pass('ext_after', _noop), after='sentinel_rename')
    try:
        names = [p.name for p in passes.get_passes('reverse')]
        assert names.index('ext_before') == names.index('anf') - 1
        assert names.index('ext_after') == names.index('sentinel_rename') + 1

        # The extended pipeline still produces correct gradients.
        def f(x):
            return x * x

        df = tangent.grad(f)
        assert df(3.0) == 6.0
    finally:
        passes.unregister_pass('ext_before')
        passes.unregister_pass('ext_after')
    assert [p.name for p in passes.get_passes('reverse')] == EXPECTED_REVERSE


def test_register_pass_validates_arguments():
    extra = passes.Pass('ext', _noop)
    with pytest.raises(ValueError):
        passes.register_pass(extra)  # neither anchor
    with pytest.raises(ValueError):
        passes.register_pass(extra, before='anf', after='fence')  # both anchors
    with pytest.raises(ValueError):
        passes.register_pass(extra, before='no_such_pass')
    with pytest.raises(ValueError):
        passes.register_pass(passes.Pass('anf', _noop), after='fence')  # collision


if __name__ == '__main__':
    assert not pytest.main([__file__])
