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
"""Templates for gradient expressions.

The first argument to the adjoint must be the return value of the primal.

Use `d[x]` to denote the gradient of a variable `x`.

If the primal returns a tuple, the first argument to the adjoint is a tuple,
and the adjoint is supposed to define `d[y]` as a tuple.

Templates do not support use of `**kwargs`.

If a keyword argument isn't present in the adjoint, it means that Tangent
doesn't support it, and an error will be raised if it appears in user code.

Adjoints have access to the inputs of the primal, output of the primal, and
gradients with respect to the output. They are expected to contain expressions
for the gradient with respect to the input. They don't have access to any
intermediate variables from the primal.

"""

from __future__ import absolute_import

import math
import types

import gast
import numpy
import tangent
from tangent import tracing
from tangent import utils


# TODO: Avoid requiring non-differentiables to define @tangent_s.
# All non-differentiable function need to create shadow zero-filled variables
# in forward mode. Currently we achieve that by defining identity @tangent_
# versions of those functions, but a beter approach would be to do that
# automatically.

# Create decorators that add templates to dictionaries
adjoints = {}
primals = {}


def get_module_functions(modules):
    """Finds functions that do not have implemented derivatives.

    Args:
      modules: A list of Python modules. Functions contained in these modules
          will be checked for membership in 'implemented', and if not found,
          will be added to an 'unimplemented' set
      implemented: A Python object containing implemented derivatives. A function
          should be checkable for membership using the `fn in implemented` syntax.

    Returns:
      module_fns: A set of functions, builtins or ufuncs in `modules`.
    """
    module_fns = set()
    for module in modules:
        for key in dir(module):
            attr = getattr(module, key)
            if isinstance(attr, (types.BuiltinFunctionType, types.FunctionType, numpy.ufunc)):
                module_fns.add(attr)
            elif callable(attr) and type(attr).__name__ == '_ArrayFunctionDispatcher':
                # NumPy 2.x wraps most public functions (sort, median, pad, ...)
                # in _ArrayFunctionDispatcher, which is none of the types above.
                # Without this branch the unimplemented sets end up nearly
                # empty, so calls to these ops recursed into NumPy's own source
                # and crashed with opaque errors instead of raising the clean
                # Reverse/ForwardNotImplementedError.
                module_fns.add(attr)
    return module_fns


def create_register(dict_):
    def register(key):
        def _(f):
            dict_[key] = f
            return f

        return _

    return register


adjoint = create_register(adjoints)
primal = create_register(primals)


# Functions: f => f, df
@adjoint(gast.FunctionDef)
def dfunction_def(adjoint_body, return_dx):
    def df():
        adjoint_body
        return_dx


# Control flow
@primal(gast.For)
def for_(body, i, iter_, target, push, push_target, _target, _stack, op_id_iter, op_id_target):
    i = 0
    for target in iter_:
        _target = target
        body
        push_target(_stack, _target, op_id_target)
        i += 1
    push(_stack, i, op_id_iter)


@adjoint(gast.For)
def dfor_(adjoint_body, i, pop, pop_target, target, _stack, op_id_iter, op_id_target):
    i = pop(_stack, op_id_iter)
    for _ in range(i):
        target = pop_target(_stack, op_id_target)
        adjoint_body


# Checkpointed For loop (Phase 2: Memory-efficient gradient computation)
# Separate dictionaries for checkpointed templates
primals_checkpointed = {}
adjoints_checkpointed = {}

primal_checkpointed = create_register(primals_checkpointed)
adjoint_checkpointed = create_register(adjoints_checkpointed)


# Segment (sqrt-n) checkpointing. The primal runs the loop UNTAPED (the
# original, untransformed body), pushing only a snapshot of the loop-carried
# state every `segment_size(n)` iterations. The adjoint restores each snapshot
# in reverse order (LIFO pops naturally yield the last segment first), replays
# just that segment with the taped body, and immediately consumes the segment's
# tape with the adjoint body. Peak tape memory is one segment plus the
# snapshots: O(sqrt(n)) instead of O(n). Replay is sound because push/pop pairs
# are balanced within each iteration, and the loop target is re-derived by
# indexing the (saved) iterable rather than being taped.


@primal_checkpointed(gast.For)
def for_checkpointed(
    orig_body,
    i,
    iter_,
    target,
    push,
    _it,
    _seg,
    _snap,
    snap_save,
    _stack,
    op_id_iter,
    op_id_it,
    op_id_snap,
):
    _it = iter_
    _seg = tangent.segment_size(len(_it))
    i = 0
    for target in _it:
        if i % _seg == 0:
            _snap = tangent.snapshot(snap_save)
            push(_stack, _snap, op_id_snap)
        orig_body
        i += 1
    push(_stack, _it, op_id_it)
    push(_stack, i, op_id_iter)


@adjoint_checkpointed(gast.For)
def dfor_checkpointed(
    body,
    adjoint_body,
    i,
    pop,
    push_target,
    pop_target,
    target,
    _target,
    _it,
    _seg,
    _s,
    _k,
    _k2,
    _start,
    _len,
    snap_restore,
    _stack,
    op_id_iter,
    op_id_it,
    op_id_snap,
    op_id_target,
):
    i = pop(_stack, op_id_iter)
    _it = pop(_stack, op_id_it)
    _seg = tangent.segment_size(i)
    for _s in range(tangent.num_segments(i, _seg)):
        snap_restore = pop(_stack, op_id_snap)
        _start, _len = tangent.segment_bounds(i, _seg, _s)
        for _k in range(_len):
            target = _it[_start + _k]
            _target = target
            body
            push_target(_stack, _target, op_id_target)
        for _k2 in range(_len):
            target = pop_target(_stack, op_id_target)
            adjoint_body


# Online (Stumm-Walther) checkpointing for while-loops: the trip count is
# unknown, so snapshots of the loop-carried state are kept under a fixed
# budget, geometrically thinned as the loop runs (tangent.online_store). The
# adjoint restores each surviving snapshot in reverse, replays its segment
# with the taped body - the loop condition is NOT re-evaluated during replay;
# the recorded iteration counts drive it - and consumes the segment's tape.
@primal_checkpointed(gast.While)
def while_checkpointed(
    orig_body,
    i,
    test,
    push,
    _snaps,
    _seg,
    _snap,
    snap_save,
    _budget,
    _stack,
    op_id_iter,
    op_id_snaps,
):
    _snaps = []
    _seg = 1
    i = 0
    while test:
        if i % _seg == 0:
            _snap = tangent.snapshot(snap_save)
            _snaps, _seg = tangent.online_store(_snaps, i, _snap, _seg, _budget)
        orig_body
        i += 1
    push(_stack, _snaps, op_id_snaps)
    push(_stack, i, op_id_iter)


@adjoint_checkpointed(gast.While)
def dwhile_checkpointed(
    body,
    adjoint_body,
    i,
    pop,
    _snaps,
    _s,
    _k,
    _k2,
    _start,
    _len,
    _snapval,
    snap_restore,
    _stack,
    op_id_iter,
    op_id_snaps,
):
    i = pop(_stack, op_id_iter)
    _snaps = pop(_stack, op_id_snaps)
    for _s in range(len(_snaps)):
        _start, _len, _snapval = tangent.online_segment(_snaps, _s, i)
        snap_restore = _snapval
        for _k in range(_len):
            body
        for _k2 in range(_len):
            adjoint_body


@primal(gast.While)
def while_(body, i, test, push, _stack, op_id):
    i = 0
    while test:
        body
        i += 1
    push(_stack, i, op_id)


@adjoint(gast.While)
def dwhile_(adjoint_body, i, pop, _stack, op_id):
    i = pop(_stack, op_id)
    for _ in range(i):
        adjoint_body


@primal(gast.If)
def if_(cond, test, body, orelse, push, _stack, op_id):
    cond = test
    if cond:
        body
    else:
        orelse
    push(_stack, cond, op_id)


@adjoint(gast.If)
def dif_(cond, adjoint_body, adjoint_orelse, pop, _stack, op_id):
    cond = pop(_stack, op_id)
    if cond:
        adjoint_body
    else:
        adjoint_orelse


# Conditional expression (ternary operator): z = body if test else orelse
# Note: The primal doesn't assign to result; it returns an IfExp expression
# The assignment happens in visit_Assign
@primal(gast.IfExp)
def ifexp_(cond, test, push, _stack, op_id):
    cond = test
    push(_stack, cond, op_id)


@adjoint(gast.IfExp)
def difexp_(result, cond, body, orelse, pop, _stack, op_id):
    cond = pop(_stack, op_id)
    if cond:
        d[body] = d[result]
    else:
        d[orelse] = d[result]


# Binary ops: z = op(x, y)
@adjoint(gast.Mult)
def mult(z, x, y):
    d[x] = tangent.unbroadcast(d[z] * y, x)
    d[y] = tangent.unbroadcast(d[z] * x, y)


@adjoint(gast.Add)
def add(z, x, y):
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = tangent.unbroadcast(d[z], y)


@adjoint(gast.Pow)
def pow(z, x, y):
    d[x] = y * x ** (y - 1) * d[z]
    d[y] = numpy.log(x) * x**y * d[z]


@adjoint(gast.Sub)
def sub(z, x, y):
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = -tangent.unbroadcast(d[z], y)


@adjoint(gast.Div)
def div(z, x, y):
    d[x] = d[z] / y
    d[y] = -d[z] * x / (y * y)


# Unary ops: y = op(x)
@adjoint(gast.USub)
def usub(y, x):
    d[x] = -d[y]


@adjoint(gast.UAdd)
def uadd(y, x):
    d[x] = d[y]


# Matrix multiplication operator: z = x @ y. The partial gradients dispatch on
# the operand type through tangent.matmul_grad_x/_y so each backend (NumPy,
# tinygrad, JAX, TF, ...) supplies its own rank-promotion handling.
@adjoint(gast.MatMult)
def matmult(z, x, y):
    d[x] = tangent.matmul_grad_x(d[z], x, y)
    d[y] = tangent.matmul_grad_y(d[z], x, y)


def matmul_grad_x_numpy(dz, x, y):
    """d[x] for z = x @ y, covering the vector/matrix rank promotions."""
    dz = numpy.asarray(dz)
    if x.ndim == 1 and y.ndim == 1:
        return dz * y
    if x.ndim == 2 and y.ndim == 1:
        return numpy.outer(dz, y)
    return numpy.matmul(dz, numpy.swapaxes(y, -1, -2))


def matmul_grad_y_numpy(dz, x, y):
    """d[y] for z = x @ y, covering the vector/matrix rank promotions."""
    dz = numpy.asarray(dz)
    if x.ndim == 1 and y.ndim == 1:
        return dz * x
    if x.ndim == 1 and y.ndim == 2:
        return numpy.outer(x, dz)
    return numpy.matmul(numpy.swapaxes(x, -1, -2), dz)


utils.register_matmul_grad(numpy.ndarray, matmul_grad_x_numpy, matmul_grad_y_numpy)


#
# NumPy adjoints
#


@adjoint(numpy.log)
def log(y, x):
    d[x] = d[y] / x


@adjoint(numpy.cos)
def cos(y, x):
    d[x] = -d[y] * numpy.sin(x)


@adjoint(numpy.sin)
def sin(y, x):
    d[x] = d[y] * numpy.cos(x)


@adjoint(numpy.tan)
def tan(y, x):
    cx = numpy.cos(x)
    d[x] = d[y] / (cx * cx)


@adjoint(numpy.cosh)
def cosh(y, x):
    d[x] = d[y] * numpy.sinh(x)


@adjoint(numpy.sinh)
def sinh(y, x):
    d[x] = d[y] * numpy.cosh(x)


@adjoint(numpy.tanh)
def tanh(y, x):
    d[x] = d[y] * (1.0 - (y * y))


@adjoint(numpy.arccos)
def arccos(y, x):
    # Guarded division for the |x| = 1 singularity: the derivative diverges
    # there, and plain -d[y] / sqrt(1 - x*x) evaluates to nan when the seed
    # d[y] is zero (0/0). That nan poisons second-derivative chains that
    # legitimately pass a zero seed through the singular point. With the
    # guard, a zero seed contributes zero while a nonzero seed still yields
    # the correct +/-inf.
    d[x] = numpy.where(d[y] != 0, -d[y] / numpy.sqrt(1.0 - x * x), 0.0)


@adjoint(numpy.arcsin)
def arcsin(y, x):
    # See the arccos adjoint for the |x| = 1 singularity guard.
    d[x] = numpy.where(d[y] != 0, d[y] / numpy.sqrt(1.0 - x * x), 0.0)


@adjoint(numpy.arctan)
def arctan(y, x):
    d[x] = d[y] / (1.0 + x * x)


@adjoint(numpy.exp)
def exp(y, x):
    d[x] = y * d[y]


@adjoint(numpy.sqrt)
def sqrt(y, x):
    d[x] = d[y] / (2.0 * y)


@adjoint(numpy.multiply)
def multiply(z, x, y):
    d[x] = tangent.unbroadcast(y * d[z], x)
    d[y] = tangent.unbroadcast(x * d[z], y)


@adjoint(numpy.dot)
def dot(y, x1, x2):
    d[x1] = tangent.grad_dot(d[y], x1, x2)
    d[x2] = numpy.transpose(
        tangent.grad_dot(numpy.transpose(d[y]), numpy.transpose(x2), numpy.transpose(x1))
    )


@adjoint(numpy.atleast_1d)
def atleast_1d(y, x):
    d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.atleast_2d)
def atleast_2d(y, x):
    d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.atleast_3d)
def atleast_3d(y, x):
    d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.reshape)
def reshape(y, x, y_shape):
    d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.transpose)
def transpose(y, x, axes=None):
    d[x] = numpy.transpose(d[y], tangent.transpose_inverse_axes(axes))


@adjoint(numpy.broadcast_arrays)
def broadcast_arrays(ys, *args):
    d[args] = tuple(tangent.unbroadcast_to(dy, numpy.shape(arg)) for arg, dy in zip(args, d[ys]))


@adjoint(numpy.sum)
def sum(y, x, axis=None, dtype=None, keepdims=False):
    d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x), axis, keepdims), x)


@adjoint(numpy.mean)
def mean(y, x, axis=None, dtype=None, keepdims=False):
    n = tangent.astype(tangent.array_size(x, axis), x)
    d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x), axis, keepdims), x) / n


@adjoint(numpy.maximum)
def maximum(ans, x, y):
    d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
    d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


#
# Ufunc spellings of the basic arithmetic operators. These mirror the
# gast.Add/Sub/Mult/Div/Pow operator adjoints above, including the
# unbroadcast handling for binary ops.
#


@adjoint(numpy.add)
def aadd_ufunc(z, x, y):
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = tangent.unbroadcast(d[z], y)


@adjoint(numpy.subtract)
def asubtract(z, x, y):
    d[x] = tangent.unbroadcast(d[z], x)
    d[y] = -tangent.unbroadcast(d[z], y)


# numpy.divide is numpy.true_divide, so this registration covers both
# spellings.
@adjoint(numpy.divide)
def adivide(z, x, y):
    d[x] = tangent.unbroadcast(d[z] / y, x)
    d[y] = tangent.unbroadcast(-d[z] * x / (y * y), y)


@adjoint(numpy.negative)
def anegative(y, x):
    d[x] = -d[y]


@adjoint(numpy.power)
def apower(z, x, y):
    d[x] = tangent.unbroadcast(y * x ** (y - 1) * d[z], x)
    d[y] = tangent.unbroadcast(numpy.log(x) * x**y * d[z], y)


@adjoint(numpy.float_power)
def afloat_power(z, x, y):
    d[x] = tangent.unbroadcast(y * x ** (y - 1) * d[z], x)
    d[y] = tangent.unbroadcast(numpy.log(x) * x**y * d[z], y)


#
# Additional elementwise math functions
#


@adjoint(numpy.arctan2)
def aarctan2(z, x, y):
    d[x] = tangent.unbroadcast(d[z] * y / (x * x + y * y), x)
    d[y] = tangent.unbroadcast(-d[z] * x / (x * x + y * y), y)


@adjoint(numpy.hypot)
def ahypot(z, x, y):
    d[x] = tangent.unbroadcast(d[z] * x / z, x)
    d[y] = tangent.unbroadcast(d[z] * y / z, y)


@adjoint(numpy.logaddexp)
def alogaddexp(z, x, y):
    d[x] = tangent.unbroadcast(d[z] * numpy.exp(x - z), x)
    d[y] = tangent.unbroadcast(d[z] * numpy.exp(y - z), y)


@adjoint(numpy.arcsinh)
def aarcsinh(y, x):
    d[x] = d[y] / numpy.sqrt(x * x + 1.0)


@adjoint(numpy.arccosh)
def aarccosh(y, x):
    d[x] = d[y] / numpy.sqrt(x * x - 1.0)


@adjoint(numpy.arctanh)
def aarctanh(y, x):
    d[x] = d[y] / (1.0 - x * x)


@adjoint(numpy.exp2)
def aexp2(y, x):
    d[x] = d[y] * y * numpy.log(2.0)


@adjoint(numpy.cbrt)
def acbrt(y, x):
    d[x] = d[y] / (3.0 * y * y)


@adjoint(numpy.fmax)
def afmax(ans, x, y):
    d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
    d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


@adjoint(numpy.fmin)
def afmin(ans, x, y):
    d[x] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(x, ans, y), x)
    d[y] = tangent.unbroadcast(d[ans] * tangent.balanced_eq(y, ans, x), y)


#
# Shape and reduction operations
#


@adjoint(numpy.cumsum)
def acumsum(y, x, axis=None):
    # `axis == None` (rather than `is`) because the template substitutes the
    # call site's literal axis value, and `1 is None` is a SyntaxWarning.
    if axis == None:  # pylint: disable=singleton-comparison
        d[x] = numpy.reshape(numpy.flip(numpy.cumsum(numpy.flip(d[y], 0), 0), 0), numpy.shape(x))
    else:
        d[x] = numpy.flip(numpy.cumsum(numpy.flip(d[y], axis), axis), axis)


@adjoint(numpy.flip)
def aflip(y, x, axis=None):
    d[x] = numpy.flip(d[y], axis)


@adjoint(numpy.sort)
def asort(y, x, axis=-1):
    d[x] = tangent.unsort(d[y], x, axis)


@adjoint(numpy.cumprod)
def acumprod(y, x):
    d[x] = tangent.uncumprod(d[y], y, x)


@adjoint(numpy.pad)
def apad(y, x, pad_width):
    d[x] = tangent.unpad(d[y], pad_width, x)


@adjoint(numpy.take)
def atake(y, x, indices, axis=None):
    d[x] = tangent.untake(d[y], indices, x, axis)


# Two-operand, explicit-output einsum (`np.einsum('ij,jk->ik', a, b)`). The
# heavy lifting - rearranging the equation for each operand's gradient - lives
# in tangent.einsum_grad, which sees the equation string at run time.
@adjoint(numpy.einsum)
def aeinsum(z, subscripts, x, y):
    d[x] = tangent.einsum_grad(subscripts, 0, d[z], x, y)
    d[y] = tangent.einsum_grad(subscripts, 1, d[z], x, y)


@adjoint(numpy.linalg.cholesky)
def acholesky(y, x):
    d[x] = tangent.cholesky_grad(y, d[y])


# w = eigvalsh(A): dA = V @ diag(dw) V.T with V the eigenvectors of A.
@adjoint(numpy.linalg.eigvalsh)
def aeigvalsh(y, x):
    d[x] = tangent.eigvalsh_grad(x, d[y])


# argsort yields an integer permutation: nothing differentiable flows through
# it (its forward twin in tangents.py returns integer zeros).
from tangent import non_differentiable as _non_differentiable  # noqa: E402

_non_differentiable.register_non_differentiable_functions(numpy.argsort)


@adjoint(numpy.ravel)
def aravel(y, x):
    d[x] = numpy.reshape(d[y], numpy.shape(x))


@adjoint(numpy.swapaxes)
def aswapaxes(y, x, axis1, axis2):
    d[x] = numpy.swapaxes(d[y], axis1, axis2)


@adjoint(numpy.moveaxis)
def amoveaxis(y, x, source, destination):
    d[x] = numpy.moveaxis(d[y], destination, source)


@adjoint(numpy.tile)
def atile(y, x, reps):
    d[x] = tangent.untile(d[y], x, reps)


@adjoint(numpy.repeat)
def arepeat(y, x, repeats, axis=None):
    d[x] = tangent.unrepeat(d[y], x, repeats, axis)


@adjoint(numpy.roll)
def aroll(y, x, shift, axis=None):
    d[x] = numpy.roll(d[y], numpy.negative(shift), axis)


#
# Linear algebra
#


@adjoint(numpy.linalg.solve)
def asolve(z, a, b):
    """Adjoint for z = numpy.linalg.solve(a, b), i.e. a @ z = b.

    d[b] = solve(a^T, d[z]); d[a] = -d[b] (x) z^T (an outer product when b is
    a vector, a matmul against z^T when b is a matrix).
    """
    _dsolve_b = numpy.linalg.solve(numpy.swapaxes(a, -1, -2), d[z])
    if numpy.ndim(b) == numpy.ndim(a) - 1:
        d[a] = -numpy.einsum('...i,...j->...ij', _dsolve_b, z)
    else:
        d[a] = -numpy.matmul(_dsolve_b, numpy.swapaxes(z, -1, -2))
    d[b] = _dsolve_b


@adjoint(numpy.linalg.norm)
def anorm(y, x, axis=None, keepdims=False):
    """Adjoint for the default (2-norm / Frobenius) numpy.linalg.norm.

    The `ord` argument is deliberately unsupported: for any other norm this
    gradient would be wrong, so calls passing `ord` fail loudly at
    differentiation time instead.
    """
    d[x] = (
        tangent.unreduce(d[y], numpy.shape(x), axis, keepdims)
        * x
        / tangent.unreduce(y, numpy.shape(x), axis, keepdims)
    )


#
# Neural Network Activation Functions
#


def numpy_relu(x):
    """ReLU activation: max(0, x)."""
    return numpy.maximum(0, x)


@adjoint(numpy_relu)
def arelu(y, x):
    """Gradient of ReLU: 1 where x > 0, else 0."""
    d[x] = d[y] * (x > 0).astype(x.dtype)


def numpy_sigmoid(x):
    """Sigmoid activation: 1/(1 + exp(-x))."""
    return 1.0 / (1.0 + numpy.exp(-x))


@adjoint(numpy_sigmoid)
def asigmoid(y, x):
    """Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    d[x] = d[y] * y * (1.0 - y)


def numpy_tanh(x):
    """Hyperbolic tangent activation (alias to numpy.tanh)."""
    return numpy.tanh(x)


# Note: numpy.tanh gradient is already defined above


def numpy_leaky_relu(x, alpha=0.01):
    """Leaky ReLU: x if x > 0 else alpha * x."""
    return numpy.where(x > 0, x, alpha * x)


@adjoint(numpy_leaky_relu)
def aleaky_relu(y, x, alpha=0.01):
    """Gradient of Leaky ReLU: 1 where x > 0, else alpha."""
    d[x] = d[y] * numpy.where(x > 0, 1.0, alpha)


def numpy_elu(x, alpha=1.0):
    """ELU activation: x if x > 0 else alpha * (exp(x) - 1)."""
    return numpy.where(x > 0, x, alpha * (numpy.exp(x) - 1.0))


@adjoint(numpy_elu)
def aelu(y, x, alpha=1.0):
    """Gradient of ELU: 1 if x > 0 else alpha * exp(x)."""
    d[x] = d[y] * numpy.where(x > 0, 1.0, alpha * numpy.exp(x))


def numpy_softplus(x):
    """Softplus activation: log(1 + exp(x))."""
    return numpy.log(1.0 + numpy.exp(x))


@adjoint(numpy_softplus)
def asoftplus(y, x):
    """Gradient of softplus: sigmoid(x) = 1/(1 + exp(-x))."""
    # Gradient is sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + numpy.exp(-x))
    d[x] = d[y] * sigmoid_x


# Activation functions are defined above and will be imported by __init__.py
# No need to export them here - they're already module-level functions


@adjoint(numpy.array)
def aarray(ans, x):
    d[x] = tangent.astype(d[ans], x)


@adjoint(numpy.linalg.det)
def adet(z, x):
    """d|A|/dA = adj(A).T

    See  Jacobi's formula: https://en.wikipedia.org/wiki/Jacobi%27s_formula
    """
    adjugate = numpy.linalg.det(x) * numpy.linalg.pinv(x)
    d[x] = d[z] * numpy.transpose(adjugate)


#
# Built-in Python functions
#


@adjoint(abs)
def absolute_builtin(y, x):
    """Adjoint for built-in abs(): ∂L/∂x = sign(x)·∂L/∂z

    The gradient of abs(x) is:
    - +1 where x > 0
    - -1 where x < 0
    - undefined at x = 0 (we use 0 by convention)

    For arrays, use numpy.abs instead for better performance.
    """
    # Use numpy.sign which handles scalars and arrays
    d[x] = d[y] * numpy.sign(x)


@adjoint(min)
def min_builtin(y, x1, x2):
    d[x1] = d[y] * (x1 <= x2)
    d[x2] = d[y] * (x2 < x1)


@adjoint(max)
def max_builtin(y, x1, x2):
    d[x1] = d[y] * (x1 >= x2)
    d[x2] = d[y] * (x2 > x1)


# Built-in numeric casts. float() is the identity map on reals, so gradients
# pass straight through; int() truncates, whose derivative is zero almost
# everywhere. Registering both keeps casts symmetric across modes (each has a
# matching @tangent_ rule) instead of erroring in one mode and not the other.
@adjoint(float)
def float_builtin(y, x):
    d[x] = 1.0 * d[y]


@adjoint(int)
def int_builtin(y, x):
    d[x] = 0.0 * d[y]


#
# Tangent adjoints
#


@adjoint(tangent.unreduce)
def aunreduce(y, x, shape, axis, keepdims):
    d[x] = tangent.unbroadcast(d[y], x)


@adjoint(tangent.unreduce_like)
def aunreduce_like(y, array, original_array, axis, keepdims):
    # unreduce_like broadcasts `array` to original_array's shape; the adjoint of
    # that broadcast reduces back to `array`'s shape. Without this adjoint,
    # third-order derivatives step into the helper's type-dispatch body and fail
    # on the builtin `type()` call.
    d[array] = tangent.unbroadcast(d[y], array)


@adjoint(tangent.unbroadcast)
def aunbroadcast(y, x, shape):
    d[x] = tangent.unreduce_like(d[y], x, None, False)


@adjoint(tangent.add_grad)
def aadd_grad(z, left, right):
    d[left] = tangent.unbroadcast(d[z], left)
    d[right] = tangent.unbroadcast(d[z], right)


# add_grad_at_index(grad_array, index, value) accumulates `value` into
# grad_array[index] and returns the container. It is a linear operation, so its
# adjoint passes the container's gradient through unchanged and routes the
# element's gradient to `value`. Registering an adjoint also keeps higher-order
# AD from trying to differentiate the helper's own (import-containing) body.
@adjoint(tangent.add_grad_at_index)
def a_add_grad_at_index(z, grad_array, index, value):
    d[grad_array] = d[z]
    d[value] = d[z][index]


@adjoint(tangent.astype)
def aastype(z, array, y):
    d[array] = tangent.astype(d[z], array)


# match_seed(primal, seed) reconciles the gradient seed with the structure of
# the return value (see tangent/utils.py). It only reads the *structure* of
# `primal`, so no gradient flows into it; with respect to `seed` it is linear
# (identity when the structures already match). Registering this adjoint lets
# higher-order AD differentiate through the seed reconciliation emitted at
# the top of the adjoint (see `ReverseAD.reconcile_seed`) instead of stepping
# into the helper's type-dispatch body.
@adjoint(tangent.match_seed)
def amatch_seed(z, primal, seed):
    d[seed] = tangent.match_seed_grad(seed, d[z])


# match_seed_grad(seed, dz) is itself linear in dz; its transpose is the
# forward reconciliation against dz's structure (a scalar cotangent broadcast
# back over the summed leaves, identity otherwise). Registering it keeps
# third- and higher-order derivatives inside these two primitives.
@adjoint(tangent.match_seed_grad)
def amatch_seed_grad(z, seed, dz):
    d[dz] = tangent.match_seed(dz, d[z])


# In these adjoints the op_id is a non-differentiable tape marker (a string
# constant), so it is passed through unchanged - exactly like `stack`. Wrapping
# it as `d[op_id]` (the gradient operator) is meaningless for a marker and, in
# higher-order AD where these adjoints actually fire, leaked an undefined `d`
# into the generated code.
@adjoint(tangent.push)
def apush(stack, val, op_id):
    d[val] = tangent.pop(stack, op_id)


@adjoint(tangent.pop)
def apop(z, stack, op_id):
    tangent.push(stack, d[z], op_id)


@adjoint(tangent.push_stack)
def apush_stack(stack, val, op_id):
    d[val] = tangent.pop_stack(stack, op_id)


@adjoint(tangent.pop_stack)
def apop_stack(z, stack, op_id):
    tangent.push_stack(stack, d[z], op_id)


@adjoint(tangent.copy)
def acopy(z, x):
    d[x] = tangent.copy(d[z])


@adjoint(tangent.stop_gradient)
def astop_gradient(y, x):
    d[x] = tangent.init_grad(x)


# The sort helpers are linear permutations of their gradient argument and
# inverses of each other, so their adjoints form a closed pair - second and
# higher derivatives through np.sort never leave the set. (The permutation is
# locally constant in x, so no gradient flows to x.)
@adjoint(tangent.unsort)
def aunsort(z, dy, x, axis=-1):
    d[dy] = tangent.sort_like(d[z], x, axis)


@adjoint(tangent.sort_like)
def asort_like(z, dx, x, axis=-1):
    d[dx] = tangent.unsort(d[z], x, axis)


# unpad/untake are linear in their gradient argument; their adjoints are the
# original forward ops.
@adjoint(tangent.unpad)
def aunpad(z, dy, pad_width, x):
    d[dy] = numpy.pad(d[z], pad_width)


@adjoint(tangent.untake)
def auntake(z, dy, indices, x, axis=None):
    d[dy] = numpy.take(d[z], indices, axis)


# List building. `xs.append(v)` is desugared to `xs = tangent.list_append(xs,
# v)` and `v = xs.pop()` to `v = tangent.list_last(xs); xs =
# tangent.list_init(xs)` (see list_method_desugar.py). The three primitives are
# closed under differentiation: each adjoint below is written only in terms of
# the other primitives (plus init_grad), so second- and higher-order
# derivatives never leave the set. None of the adjoints needs the primal
# values except for their *structure* (init_grad), which is exactly what the
# tape preserves.
@adjoint(tangent.list_append)
def alist_append(ys, xs, elt):
    d[xs] = tangent.list_init(d[ys])
    d[elt] = tangent.list_last(d[ys])


@adjoint(tangent.list_last)
def alist_last(v, xs):
    d[xs] = tangent.list_append(tangent.init_grad(tangent.list_init(xs)), d[v])


@adjoint(tangent.list_init)
def alist_init(ys, xs):
    d[xs] = tangent.list_append(d[ys], tangent.init_grad(tangent.list_last(xs)))


#
# Tracing primitives
#


@primal(tracing.Traceable)
def traceable_primal(result, fn, vjp, tmp, args):
    result, vjp = tangent.trace_grad(fn, args)


@adjoint(tracing.Traceable)
def traceable_adjoint(result, vjp, dargs):
    dargs = vjp(d[result])


#
# Blacklist unimplemented NumPy grads
#

# We can enumerate all of the functions that we'd like grads for.
# Until we've written the adjoints of all functions we want to support,
# we will throw an explicit "no grad found" error for those we have not
# finished. UNIMPLEMENTED will contain the list of all of these unimplemented
# grad functions
UNIMPLEMENTED_ADJOINTS = get_module_functions(
    (numpy, numpy.fft, numpy.linalg, numpy.random, math)
) - set(adjoints)
