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
"""Explain a gradient: primal, adjoint, numeric check, and a source map.

`tangent.explain(f, x)` is the debuggability pitch as one call: it shows the
function, the generated gradient source, evaluates both, cross-checks the
gradient against central finite differences, and points out inputs that do
not affect the output at this point.

`tangent.source_map(df)` maps each line of the generated gradient back to the
primal statement it differentiates (recovered from the `# Grad of:` comments
the transformer emits), including the line number in the user's file when it
can be found.
"""

from __future__ import absolute_import

import inspect

import numpy


def source_map(df, func=None):
    """Map generated-gradient lines back to primal statements.

    Args:
      df: A gradient function produced by `tangent.grad`/`tangent.autodiff`
          (it carries the generated source in `__tangent_source__`).
      func: Optionally the primal function, to resolve primal line numbers in
          the user's source file. Defaults to the one recorded on `df`.

    Returns:
      A list of dicts `{line, code, primal, primal_line}`: the 1-based line
      number and text in the generated source, the primal statement that line
      serves (None for scaffolding), and the primal's 1-based line number in
      the user's file (None when unresolvable).
    """
    source = getattr(df, '__tangent_source__', None)
    if source is None:
        wrapped = getattr(df, '__wrapped__', None)
        source = getattr(wrapped, '__tangent_source__', None)
    if source is None:
        raise ValueError(
            'This function carries no generated source; pass a gradient '
            'produced by tangent.grad / tangent.autodiff.'
        )
    func = func or getattr(df, '__tangent_primal__', None)

    primal_lines = {}
    if func is not None:
        try:
            lines, start = inspect.getsourcelines(func)
            for offset, text in enumerate(lines):
                primal_lines[text.strip()] = start + offset
        except (OSError, TypeError):
            pass

    entries = []
    current_primal = None
    markers = ('# Grad of: ', '# Primal and tangent of: ', '# Beginning of forward pass')
    for lineno, text in enumerate(source.splitlines(), start=1):
        stripped = text.strip()
        for marker in markers[:2]:
            if stripped.startswith(marker):
                current_primal = stripped[len(marker) :]
                break
        entries.append(
            {
                'line': lineno,
                'code': text,
                'primal': current_primal,
                'primal_line': primal_lines.get(current_primal),
            }
        )
    return entries


def _central_difference(func, args, wrt, eps=1e-6):
    """Central-difference gradients of sum(func(*args)) for the wrt args."""

    def scalar_out(*a):
        out = func(*a)
        if isinstance(out, tuple):
            return sum(numpy.sum(o) for o in out)
        return numpy.sum(out)

    grads = []
    for index in wrt:
        x = numpy.asarray(args[index], dtype=float)
        g = numpy.zeros_like(x)
        it = numpy.nditer(x, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            xp = x.copy()
            xm = x.copy()
            xp[idx] += eps / 2
            xm[idx] -= eps / 2
            ap = list(args)
            am = list(args)
            ap[index] = xp if x.ndim else float(xp)
            am[index] = xm if x.ndim else float(xm)
            g[idx] = (scalar_out(*ap) - scalar_out(*am)) / eps
        grads.append(g if x.ndim else float(g))
    return grads if len(grads) > 1 else grads[0]


def explain(func, *args, wrt=(0,), out=print):
    """Show the primal, the generated adjoint, and a verified gradient.

    Args:
      func: The function to differentiate.
      *args: Arguments to evaluate at.
      wrt: Argument indices to differentiate with respect to.
      out: Where to print (pass `lambda s: None` to silence).

    Returns:
      A dict with `value`, `gradient`, `numeric_gradient`, `max_error`,
      `gradient_source`, and `source_map`.
    """
    import tangent

    df = tangent.grad(func, wrt=wrt)
    gradient_source = getattr(df, '__tangent_source__', '<unavailable>')

    value = func(*args)
    gradient = df(*args)
    numeric = _central_difference(func, args, wrt)
    max_error = float(
        numpy.max(
            [
                numpy.max(numpy.abs(numpy.asarray(g) - numpy.asarray(n)))
                for g, n in zip(
                    gradient if isinstance(gradient, tuple) else (gradient,),
                    numeric if isinstance(numeric, list) else [numeric],
                )
            ]
        )
    )

    try:
        primal_source = inspect.getsource(func)
    except (OSError, TypeError):
        primal_source = '<source unavailable>'

    bar = '─' * 72
    out(bar)
    out('PRIMAL  %s%s' % (func.__name__, inspect.signature(func)))
    out(bar)
    out(primal_source.rstrip())
    out(bar)
    out('GRADIENT (generated source)')
    out(bar)
    out(gradient_source.rstrip())
    out(bar)
    out('value      = %r' % (value,))
    out('gradient   = %r' % (gradient,))
    out('fd check   = %r' % (numeric,))
    out('max |Δ|    = %.3g  %s' % (max_error, 'OK' if max_error < 1e-4 else 'MISMATCH'))
    grads_tuple = gradient if isinstance(gradient, tuple) else (gradient,)
    for k, g in zip(wrt, grads_tuple):
        if numpy.all(numpy.asarray(g) == 0):
            out(
                'note: argument %d has zero gradient here - it does not '
                'affect the output at this point.' % k
            )
    out(bar)

    return {
        'value': value,
        'gradient': gradient,
        'numeric_gradient': numeric,
        'max_error': max_error,
        'gradient_source': gradient_source,
        'source_map': source_map(df, func),
    }
