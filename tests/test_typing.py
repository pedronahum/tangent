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
"""The public API is statically typed (PEP 561).

`tangent` ships a `py.typed` marker and inline type annotations for its entry
points, so `df = tangent.grad(f)` is a typed callable rather than `Any` - and,
for a single-argument function, `df`'s result carries the input's type (the
gradient has the input's shape/type). These tests pin that contract: the marker
is present, and mypy (when installed) infers the input-preserving type without
errors.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

import tangent


def test_py_typed_marker_is_shipped():
    """PEP 561 marker sits next to the package so downstream tooling sees types."""
    pkg_dir = os.path.dirname(tangent.__file__)
    assert os.path.exists(os.path.join(pkg_dir, 'py.typed')), 'tangent/py.typed missing'


def test_public_entry_points_are_callable():
    # Runtime smoke: the annotations must not disturb the real objects.
    for name in ('grad', 'autodiff', 'vjp', 'jvp'):
        assert callable(getattr(tangent, name)), name


_MYPY = shutil.which('mypy') or (
    os.path.join(os.path.dirname(sys.executable), 'mypy')
    if os.path.exists(os.path.join(os.path.dirname(sys.executable), 'mypy'))
    else None
)


@pytest.mark.skipif(_MYPY is None, reason='mypy not installed')
def test_grad_result_type_is_inferred_by_mypy(tmp_path):
    """`tangent.grad(f)` returns a callable whose result has the input's type."""
    snippet = tmp_path / 'check_types.py'
    snippet.write_text(
        textwrap.dedent(
            """
            from typing import List
            import tangent

            def f(x: List[float]) -> float:
                return sum(v * v for v in x)

            df = tangent.grad(f)
            reveal_type(df(([1.0])))   # the gradient has the input's type
            """
        )
    )
    # Point mypy at tangent's own source tree (robust to editable installs,
    # which mypy's import resolver does not always follow).
    repo_root = os.path.dirname(os.path.dirname(tangent.__file__))
    env = dict(os.environ, MYPYPATH=repo_root)
    proc = subprocess.run(
        [
            _MYPY,
            '--follow-imports=silent',
            '--ignore-missing-imports',
            '--no-error-summary',
            str(snippet),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    out = proc.stdout + proc.stderr
    # No type errors, and the gradient's result is inferred as the input type.
    assert 'error:' not in out, out
    assert 'list[float]' in out.lower(), out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
