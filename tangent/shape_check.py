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
"""Compile-time shape/dtype checking by abstract interpretation.

`tangent.check_shapes(func, *inputs)` runs `func` on **abstract arrays** that
carry only a shape and dtype (no data), applying NumPy's real broadcasting and
linear-algebra rules. A rank/broadcast/matmul mismatch raises a `ShapeError`
that points at the offending line of the *user's* source - catching the class
of bug (rank and broadcast mistakes) that finite differences only reveal at run
time, and doing it without allocating the arrays.

Design for no false positives: an operation whose shape rule is not modeled
yields an unknown shape (dimensions become `None`, which broadcasts against
anything) rather than a guessed one, so the checker only ever reports a
mismatch it is certain about.
"""

from __future__ import absolute_import

import numpy

from tangent.errors import ShapeError

# The abstract inputs pass through the user's function; a ShapeError surfaces
# with a Python traceback whose deepest non-internal frame is the offending
# user line. Frames in this module (compared by absolute path, not a substring
# that would also match a user file called *shape_check.py) are skipped.
_THIS_FILE = __file__


def _shape_of(x):
    if isinstance(x, ShapedArray):
        return x.shape
    if isinstance(x, (int, float, complex, bool, numpy.number)):
        return ()
    shp = getattr(x, 'shape', None)
    if shp is not None:
        return tuple(shp)
    return None  # unknown


def _dtype_of(x):
    if isinstance(x, ShapedArray):
        return x.dtype
    try:
        return numpy.asarray(x).dtype if not isinstance(x, ShapedArray) else x.dtype
    except Exception:
        return numpy.dtype(float)


def _broadcast(shapes, op):
    """NumPy broadcasting with `None` dimensions as wildcards.

    Returns the broadcast shape, or raises ShapeError naming `op`. Any shape
    that is `None` (fully unknown rank) makes the whole result unknown.
    """
    known = [s for s in shapes if s is not None]
    if len(known) != len(shapes):
        return None
    if not known:
        return ()
    ndim = max(len(s) for s in known)
    result = []
    for axis in range(ndim):
        dims = []
        for s in known:
            i = len(s) - ndim + axis
            if i >= 0:
                dims.append(s[i])
        out = 1
        for d in dims:
            if d is None:
                out = None if out in (1, None) else out
            elif out in (1, None) and d != 1:
                out = d if out == 1 else out
            elif d == 1 or d == out:
                pass
            else:
                raise ShapeError(
                    'Incompatible shapes for %s: cannot broadcast %s'
                    % (op, ' and '.join(str(s) for s in shapes if s is not None))
                )
        result.append(out)
    return tuple(result)


class ShapedArray(object):
    """An array with a shape and dtype but no data, for abstract evaluation."""

    # Bind operators ahead of NumPy's, and let np.* functions dispatch here.
    __array_priority__ = 100

    def __init__(self, shape, dtype=numpy.dtype(float)):
        self.shape = tuple(shape)
        self.dtype = numpy.dtype(dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        n = 1
        for d in self.shape:
            if d is None:
                return None
            n *= d
        return n

    @property
    def T(self):
        return ShapedArray(self.shape[::-1], self.dtype)

    def __repr__(self):
        return 'ShapedArray(shape=%s, dtype=%s)' % (self.shape, self.dtype)

    # --- NumPy dispatch --------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'reduce':
            return _reduce_shape(inputs[0], kwargs)
        if method != '__call__':
            return NotImplemented
        out_shape = _broadcast([_shape_of(i) for i in inputs], ufunc.__name__)
        out_dtype = numpy.result_type(*[_dtype_of(i) for i in inputs])
        n_out = ufunc.nout
        result = ShapedArray(out_shape, out_dtype)
        return result if n_out == 1 else tuple(result for _ in range(n_out))

    def __array_function__(self, func, types, args, kwargs):
        handler = _ARRAY_FUNCTIONS.get(func)
        if handler is None:
            return _unknown_like(args)
        return handler(*args, **kwargs)

    # --- operators (two ShapedArrays don't trigger __array_ufunc__) ------

    def _binop(self, other, name):
        return ShapedArray(
            _broadcast([_shape_of(self), _shape_of(other)], name),
            numpy.result_type(_dtype_of(self), _dtype_of(other)),
        )

    def __add__(self, o):
        return self._binop(o, 'add')

    __radd__ = __add__

    def __sub__(self, o):
        return self._binop(o, 'subtract')

    def __rsub__(self, o):
        return self._binop(o, 'subtract')

    def __mul__(self, o):
        return self._binop(o, 'multiply')

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._binop(o, 'divide')

    def __rtruediv__(self, o):
        return self._binop(o, 'divide')

    def __pow__(self, o):
        return self._binop(o, 'power')

    def __rpow__(self, o):
        return self._binop(o, 'power')

    def __neg__(self):
        return ShapedArray(self.shape, self.dtype)

    def __matmul__(self, o):
        return _matmul_shape(self, o)

    def __rmatmul__(self, o):
        return _matmul_shape(o, self)

    def _cmp(self, o):
        return ShapedArray(
            _broadcast([_shape_of(self), _shape_of(o)], 'compare'), numpy.dtype(bool)
        )

    __gt__ = __lt__ = __ge__ = __le__ = _cmp

    def __getitem__(self, index):
        return ShapedArray(_index_shape(self.shape, index), self.dtype)

    # --- common method spellings (arr.sum(), arr.reshape(...), ...) ------

    def sum(self, axis=None, **kw):
        return _reduce_shape(self, {'axis': axis})

    def mean(self, axis=None, **kw):
        return _reduce_shape(self, {'axis': axis})

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _reshape_shape(self, shape)

    def astype(self, dtype):
        return ShapedArray(self.shape, dtype)

    def copy(self):
        return ShapedArray(self.shape, self.dtype)

    # `len(x)` and iteration appear in supported primal patterns
    # (`for i in range(len(x))`, `for v in x`).
    def __len__(self):
        if not self.shape or self.shape[0] is None:
            raise ShapeError('len() of an unsized or unknown-length ShapedArray')
        return self.shape[0]

    def __iter__(self):
        n = len(self)
        elem = ShapedArray(self.shape[1:], self.dtype)
        for _ in range(n):
            yield elem


def _unknown_like(args):
    for a in args:
        if isinstance(a, ShapedArray):
            return ShapedArray((None,) * a.ndim if a.ndim else (), a.dtype)
    return ShapedArray((), numpy.dtype(float))


def _reduce_shape(x, kwargs):
    shape = _shape_of(x)
    if shape is None:
        return ShapedArray((), _dtype_of(x))
    axis = kwargs.get('axis', None)
    keepdims = kwargs.get('keepdims', False)
    if axis is None:
        return ShapedArray((), _dtype_of(x))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a % len(shape) for a in axes)
    out = []
    for i, d in enumerate(shape):
        if i in axes:
            if keepdims:
                out.append(1)
        else:
            out.append(d)
    return ShapedArray(tuple(out), _dtype_of(x))


def _matmul_shape(a, b):
    sa, sb = _shape_of(a), _shape_of(b)
    if sa is None or sb is None:
        return ShapedArray(None, numpy.result_type(_dtype_of(a), _dtype_of(b)))
    dt = numpy.result_type(_dtype_of(a), _dtype_of(b))
    if len(sa) == 1 and len(sb) == 1:
        if sa[0] is not None and sb[0] is not None and sa[0] != sb[0]:
            raise ShapeError('matmul: mismatched vector lengths %s and %s' % (sa, sb))
        return ShapedArray((), dt)
    ka = sa[-1]
    kb = sb[-2] if len(sb) >= 2 else sb[0]
    if ka is not None and kb is not None and ka != kb:
        raise ShapeError('matmul: cannot multiply shapes %s and %s (%s != %s)' % (sa, sb, ka, kb))
    if len(sa) >= 2 and len(sb) >= 2:
        batch = _broadcast([sa[:-2], sb[:-2]], 'matmul')
        return ShapedArray(tuple(batch) + (sa[-2], sb[-1]), dt)
    if len(sa) >= 2 and len(sb) == 1:
        return ShapedArray(sa[:-1], dt)
    if len(sa) == 1 and len(sb) >= 2:
        return ShapedArray(sb[:-2] + (sb[-1],), dt)
    return ShapedArray(None, dt)


def _reshape_shape(x, shape):
    src = _shape_of(x)
    shape = tuple(shape)
    if src is None or any(d is None for d in src) or -1 in shape:
        return ShapedArray(tuple(d if d != -1 else None for d in shape), _dtype_of(x))
    total = 1
    for d in src:
        total *= d
    new_total = 1
    for d in shape:
        new_total *= d
    if new_total != total:
        raise ShapeError('reshape: cannot reshape size %d array into shape %s' % (total, shape))
    return ShapedArray(shape, _dtype_of(x))


def _index_shape(shape, index):
    if shape is None:
        return None
    if not isinstance(index, tuple):
        index = (index,)
    if any(idx is Ellipsis for idx in index):
        return (None,) * max(0, len(shape) - sum(1 for i in index if i is not Ellipsis))
    out = []
    axis = 0
    for idx in index:
        if axis >= len(shape):
            break
        if isinstance(idx, (int, numpy.integer)):
            axis += 1  # integer index removes the axis
        elif isinstance(idx, slice):
            out.append(None)  # slice length not tracked statically
            axis += 1
        else:
            out.append(None)
            axis += 1
    out.extend(shape[axis:])
    return tuple(out)


def _concatenate_shape(arrays, axis=0):
    shapes = [_shape_of(a) for a in arrays]
    if any(s is None for s in shapes):
        return ShapedArray(None, numpy.result_type(*[_dtype_of(a) for a in arrays]))
    ndim = len(shapes[0])
    axis = axis % ndim
    total = 0
    for s in shapes:
        if len(s) != ndim:
            raise ShapeError('concatenate: mismatched ranks %s' % (shapes,))
        total = None if (total is None or s[axis] is None) else total + s[axis]
    out = list(shapes[0])
    out[axis] = total
    return ShapedArray(tuple(out), numpy.result_type(*[_dtype_of(a) for a in arrays]))


def _stack_shape(arrays, axis=0):
    shapes = [_shape_of(a) for a in arrays]
    if any(s is None for s in shapes):
        return ShapedArray(None, numpy.result_type(*[_dtype_of(a) for a in arrays]))
    base = shapes[0]
    out = list(base)
    out.insert(axis % (len(base) + 1), len(arrays))
    return ShapedArray(tuple(out), numpy.result_type(*[_dtype_of(a) for a in arrays]))


_ARRAY_FUNCTIONS = {
    numpy.sum: lambda x, axis=None, **k: _reduce_shape(x, {'axis': axis, **k}),
    numpy.mean: lambda x, axis=None, **k: _reduce_shape(x, {'axis': axis, **k}),
    numpy.prod: lambda x, axis=None, **k: _reduce_shape(x, {'axis': axis, **k}),
    numpy.max: lambda x, axis=None, **k: _reduce_shape(x, {'axis': axis, **k}),
    numpy.min: lambda x, axis=None, **k: _reduce_shape(x, {'axis': axis, **k}),
    numpy.matmul: _matmul_shape,
    numpy.dot: _matmul_shape,
    numpy.reshape: lambda x, shape, **k: _reshape_shape(x, shape),
    numpy.transpose: lambda x, axes=None: ShapedArray(
        _shape_of(x)[::-1] if axes is None and _shape_of(x) is not None else _shape_of(x),
        _dtype_of(x),
    ),
    numpy.concatenate: lambda arrs, axis=0, **k: _concatenate_shape(arrs, axis),
    numpy.stack: lambda arrs, axis=0, **k: _stack_shape(arrs, axis),
    numpy.where: lambda c, x, y: ShapedArray(
        _broadcast([_shape_of(c), _shape_of(x), _shape_of(y)], 'where'),
        numpy.result_type(_dtype_of(x), _dtype_of(y)),
    ),
}


def _to_abstract(x):
    if isinstance(x, ShapedArray):
        return x
    if isinstance(x, numpy.ndarray):
        return ShapedArray(x.shape, x.dtype)
    if isinstance(x, (list, tuple)):
        arr = numpy.asarray(x)
        return ShapedArray(arr.shape, arr.dtype)
    return x  # scalars pass through unchanged


def _locate(exc):
    """Return 'file:line: source' for the deepest user frame in a traceback."""
    import traceback

    tb = exc.__traceback__
    frames = traceback.extract_tb(tb)
    import os

    for frame in reversed(frames):
        fn = frame.filename
        if os.path.abspath(fn) == os.path.abspath(_THIS_FILE):
            continue
        if os.sep + 'numpy' + os.sep in fn:
            continue
        return '%s:%d: %s' % (fn, frame.lineno, (frame.line or '').strip())
    return None


def check_shapes(func, *inputs):
    """Abstractly evaluate `func` on the given inputs' shapes.

    Args:
      func: The function to check (a primal, or a `tangent.grad` gradient).
      *inputs: Example inputs - real arrays, or `ShapedArray`s, or scalars.
          Only their shapes and dtypes are used; no data is read.

    Returns:
      The output `ShapedArray` (or structure of them).

    Raises:
      ShapeError: On a rank/broadcast/matmul/reshape mismatch, with the
          offending line of `func`'s source in the message.
    """
    abstract = [_to_abstract(x) for x in inputs]
    try:
        return func(*abstract)
    except ShapeError as e:
        loc = _locate(e)
        if loc and loc not in str(e):
            raise ShapeError('%s\n    at %s' % (e, loc)) from None
        raise
