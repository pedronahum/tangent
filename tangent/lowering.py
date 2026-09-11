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
"""Lower a generated gradient function into a backend compiler.

The generated adjoint is ordinary Python - that is the point - but ordinary
Python is also the performance ceiling. `tangent.grad(f, compile=...)` keeps
the readable source as the debug artifact (`df.__tangent_source__`,
`verbose=1`, `tangent.explain`) while handing the *execution* to a backend
compiler that fuses it:

- ``compile='python'`` - the default: run the generated source as-is.
- ``compile='jax'`` - wrap with `jax.jit`. Works when the primal is written
  against `jax.numpy` and the generated code is traceable (no data-dependent
  Python control flow - fixed loops unroll under trace).
- ``compile='torch'`` - wrap with `torch.compile` (graph breaks permitted,
  so tape operations degrade gracefully instead of failing).
- ``compile='tinygrad'`` - wrap with `tinygrad.TinyJit`; the adjoint of
  tinygrad code is itself a tinygrad graph, so the scheduler fuses it.

Numba is deliberately not offered: the generated code calls Python-level
runtime helpers (the tape, `unbroadcast`, ...) that nopython mode cannot
compile, and object mode would only add overhead.
"""

from __future__ import absolute_import

BACKENDS = ('python', 'jax', 'torch', 'tinygrad')


def _attach(compiled, df):
    for attr in ('__tangent_source__', '__tangent_entries__', '__tangent_motion__'):
        if hasattr(df, attr):
            try:
                setattr(compiled, attr, getattr(df, attr))
            except (AttributeError, TypeError):
                # Some compiler wrappers (e.g. TinyJit instances) may not
                # accept arbitrary attributes; the original stays reachable
                # through __wrapped__.
                break
    try:
        compiled.__wrapped__ = df
    except (AttributeError, TypeError):
        pass
    return compiled


def lower(df, backend):
    """Return `df` lowered into the requested backend compiler."""
    if backend is None or backend == 'python':
        return df
    if backend == 'jax':
        import jax

        return _attach(jax.jit(df), df)
    if backend == 'torch':
        import torch

        return _attach(torch.compile(df), df)
    if backend == 'tinygrad':
        from tinygrad import TinyJit

        return _attach(TinyJit(df), df)
    if backend == 'numba':
        raise ValueError(
            "compile='numba' is not supported: the generated gradient calls "
            'Python-level runtime helpers (the tape, unbroadcast, ...) that '
            "numba's nopython mode cannot compile. Use compile='jax', "
            "'torch', or 'tinygrad', or run the plain Python "
            "(compile='python')."
        )
    raise ValueError('Unknown compile backend %r; choose one of %s' % (backend, BACKENDS))
