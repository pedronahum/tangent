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
"""Capture a function's source so Tangent can differentiate it anywhere.

Source-to-source AD needs the function's source. `inspect.getsource` finds it
for functions defined in imported modules, but fails in the plain Python REPL,
in `exec`-created functions, and behind some decorators. `@tangent.function`
grabs the source at definition time and stashes it on the function, so a later
`tangent.grad` works regardless of where the function lives.

    @tangent.function
    def f(x):
        return x * x

    tangent.grad(f)(3.0)   # works even in a bare REPL

When even definition-time capture is impossible (a function built by `exec`
with no retrievable source), pass the source explicitly:

    tangent.grad(f, source="def f(x):\\n    return x * x")
"""

from __future__ import absolute_import

import inspect
import textwrap


def _strip_decorators(src):
    """Drop leading decorator lines so the source starts at `def`/`async def`.

    Captured source includes the `@tangent.function` line (and any decorators
    above it); Tangent's frontend expects a bare function definition.
    """
    src = textwrap.dedent(src)
    lines = src.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('async def '):
            start = i
            break
    return '\n'.join(lines[start:]) + '\n'


def _attach_source(fn, source):
    try:
        fn._tangent_source = source
    except (AttributeError, TypeError):
        pass
    return fn


def with_source(fn, source):
    """Attach explicit source to `fn` (used by `tangent.grad(f, source=...)`)."""
    return _attach_source(fn, _strip_decorators(source))


def function(fn=None, *, source=None):
    """Decorator: capture `fn`'s source now so Tangent can differentiate it later.

    Usable bare (`@tangent.function`) or with an explicit source string
    (`@tangent.function(source=...)`) for functions whose source is not
    otherwise retrievable.
    """
    if fn is None:
        # Called with arguments: @tangent.function(source=...)
        def decorator(f):
            return function(f, source=source)

        return decorator

    if source is None:
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            source = None
    if source is not None:
        _attach_source(fn, _strip_decorators(source))
    # If capture failed and no source was given, fall through unchanged: the
    # function still works if inspect.getsource succeeds at grad() time, and
    # otherwise raises the usual clear SourceCodeNotAvailableError.
    return fn
