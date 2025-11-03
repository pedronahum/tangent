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
      if isinstance(
          attr, (types.BuiltinFunctionType, types.FunctionType, numpy.ufunc)):
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
def for_(body, i, iter_, target, push, push_target, _target, _stack, op_id_iter,
         op_id_target):
  i = 0
  for target in iter_:
    _target = target
    body
    push_target(_stack, _target, op_id_target)
    i += 1
  push(_stack, i, op_id_iter)


@adjoint(gast.For)
def dfor_(adjoint_body, i, pop, pop_target, target, _stack, op_id_iter,
          op_id_target):
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


@primal_checkpointed(gast.For)
def for_checkpointed(body, i, iter_, target, push, push_target, _target, _stack,
                     op_id_iter, op_id_target, _checkpoint_dict, _checkpoint_positions_list):
  """For loop with checkpointing - Phase 4a: selective target storage."""
  # Compute optimal checkpoint positions
  _num_checkpoints = tangent.compute_optimal_checkpoints(len(iter_))
  _checkpoint_positions_list = tangent.compute_checkpoint_positions(len(iter_), _num_checkpoints)
  _checkpoint_positions_set = set(_checkpoint_positions_list)
  _checkpoint_dict = {}

  i = 0
  for target in iter_:
    _target = target

    # Store checkpoint at checkpoint positions (before body execution)
    if i in _checkpoint_positions_set:
      _checkpoint_dict[i] = {'target': _target, 'iteration': i}

    i += 1
    body  # Body executes normally, pushes to stack as usual

    # Push target only at checkpoints (after body)
    if (i - 1) in _checkpoint_positions_set:
      push_target(_stack, _target, op_id_target)

  # Final pushes
  push(_stack, i, op_id_iter)
  push(_stack, _checkpoint_dict, '_checkpoint_dict')
  push(_stack, _checkpoint_positions_list, '_checkpoint_positions')


@adjoint_checkpointed(gast.For)
def dfor_checkpointed(adjoint_body, i, pop, pop_target, target, _stack,
                      op_id_iter, op_id_target, _checkpoint_dict, _checkpoint_positions_list):
  """Adjoint for checkpointed loop - Phase 4a: selective pops with dict reconstruction."""
  # Retrieve checkpoint data
  _checkpoint_positions_list = pop(_stack, '_checkpoint_positions')
  _checkpoint_dict = pop(_stack, '_checkpoint_dict')
  i = pop(_stack, op_id_iter)

  # Convert to set for O(1) lookup
  _checkpoint_positions_set = set(_checkpoint_positions_list)
  _num_checkpoints = len(_checkpoint_dict)

  # Backward iteration: pop checkpoints, reconstruct others
  for _iteration in range(i - 1, -1, -1):
    if _iteration in _checkpoint_positions_set:
      # This was a checkpoint - pop from stack
      target = pop_target(_stack, op_id_target)
    else:
      # Not a checkpoint - reconstruct target value
      # Phase 4a: For range() loops, target == iteration index
      # This is a simplification that works for the common case
      target = _iteration

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
  d[y] = numpy.log(x) * x ** y * d[z]


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
  d[x] = -d[y] / numpy.sqrt(1.0 - x * x)


@adjoint(numpy.arcsin)
def arcsin(y, x):
  d[x] = d[y] / numpy.sqrt(1.0 - x * x)


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
  d[x] = y * d[z]
  d[y] = x * d[z]


@adjoint(numpy.dot)
def dot(y, x1, x2):
  d[x1] = tangent.grad_dot(d[y], x1, x2)
  d[x2] = numpy.transpose(tangent.grad_dot(numpy.transpose(d[y]),
                                           numpy.transpose(x2),
                                           numpy.transpose(x1)))


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
def transpose(y, x):
  d[x] = numpy.transpose(d[y])


@adjoint(numpy.broadcast_arrays)
def broadcast_arrays(ys, *args):
  d[args] = tuple(tangent.unbroadcast_to(dy, numpy.shape(arg))
                  for arg, dy in zip(args, d[ys]))


@adjoint(numpy.sum)
def sum(y, x, axis=None, dtype=None, keepdims=False):
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x)


@adjoint(numpy.mean)
def mean(y, x, axis=None, dtype=None, keepdims=False):
  n = tangent.astype(tangent.array_size(x, axis), x)
  d[x] = tangent.astype(tangent.unreduce(d[y], numpy.shape(x),
                                         axis, keepdims), x) / n


@adjoint(numpy.maximum)
def maximum(ans, x, y):
  d[x] = d[ans] * tangent.balanced_eq(x, ans, y)
  d[y] = d[ans] * tangent.balanced_eq(y, ans, x)


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
def aarray(ans,x):
  d[x] = tangent.astype(d[ans],x)


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
def min_builtin(y, *args):
  """Adjoint for built-in min(): gradient flows to the minimum argument(s).

  If multiple arguments have the minimum value, the gradient is distributed
  equally among them.
  """
  # For each argument, check if it equals the minimum
  # This handles the case where multiple args have the same value
  for arg in args:
    d[arg] = d[y] * (arg == y)


@adjoint(max)
def max_builtin(y, *args):
  """Adjoint for built-in max(): gradient flows to the maximum argument(s).

  If multiple arguments have the maximum value, the gradient is distributed
  equally among them.
  """
  # For each argument, check if it equals the maximum
  for arg in args:
    d[arg] = d[y] * (arg == y)


#
# Tangent adjoints
#


@adjoint(tangent.unreduce)
def aunreduce(y, x, shape, axis, keepdims):
  d[x] = tangent.unbroadcast(d[y], x)


@adjoint(tangent.unbroadcast)
def aunbroadcast(y, x, shape):
  d[x] = tangent.unreduce_like(d[y], x, None, False)


@adjoint(tangent.add_grad)
def aadd_grad(z, left, right):
  d[left] = tangent.unbroadcast(d[z], left)
  d[right] = tangent.unbroadcast(d[z], right)


@adjoint(tangent.astype)
def aastype(z, array, y):
  d[array] = tangent.astype(d[z], array)


@adjoint(tangent.push)
def apush(stack, val, op_id):
  d[val] = tangent.pop(stack, d[op_id])


@adjoint(tangent.pop)
def apop(z, stack, op_id):
  tangent.push(stack, d[z], d[op_id])


@adjoint(tangent.push_stack)
def apush_stack(stack, val, op_id):
  d[val] = tangent.pop_stack(stack, d[op_id])


@adjoint(tangent.pop_stack)
def apop_stack(z, stack, op_id):
  tangent.push_stack(stack, d[z], d[op_id])


@adjoint(tangent.copy)
def acopy(z, x):
  d[x] = tangent.copy(d[z])

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
    (numpy, numpy.fft, numpy.linalg, numpy.random, math)) - set(adjoints)
