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
"""Perform reverse mode automatic differentiation on an AST.

This module contains the machinery to take the AST of a function and to return
the AST of the gradient, either in split or joint mode. It does so by first
walking the tree with the `ReverseAD` transformer, then handling e.g. the
initializiation of gradients using the `_fix` function, and finally combining
the primal and adjoint functions in either `split` or `joint` mode.
"""
from __future__ import absolute_import
import collections
import copy
import inspect
from uuid import uuid4

import gast
import six

from tangent import annotate
from tangent import annotations as anno
from tangent import ast as ast_
from tangent import cfg
from tangent import comments
from tangent import create
from tangent import errors
from tangent import fixes
from tangent import funcsigs
from tangent import grads
from tangent import naming
from tangent import non_differentiable
from tangent import quoting
from tangent import template
from tangent import tracing
from tangent import utils


# Some AST nodes to fill in to templates that use stacks or reset gradients
PUSH = quoting.quote('tangent.push')
POP = quoting.quote('tangent.pop')
anno.setanno(PUSH, 'push_func', True)
anno.setanno(POP, 'pop_func', True)
PUSH_STACK = quoting.quote('tangent.push_stack')
POP_STACK = quoting.quote('tangent.pop_stack')
anno.setanno(PUSH_STACK, 'push_func', True)
anno.setanno(POP_STACK, 'pop_func', True)


def _generate_op_id():
  return quoting.quote("'_{}'".format(uuid4().hex[:8]))


def get_push_pop():
  """Create pop and push nodes that are linked.

  Returns:
    A push and pop node which have `push_func` and `pop_func` annotations
        respectively, identifying them as such. They also have a `pop` and
        `push` annotation respectively, which links the push node to the pop
        node and vice versa.
  """
  push = copy.deepcopy(PUSH)
  pop = copy.deepcopy(POP)
  anno.setanno(push, 'pop', pop)
  anno.setanno(push, 'gen_push', True)
  anno.setanno(pop, 'push', push)
  op_id = _generate_op_id()
  return push, pop, op_id


def get_push_pop_stack():
  """Create pop and push nodes for substacks that are linked.

  Returns:
    A push and pop node which have `push_func` and `pop_func` annotations
        respectively, identifying them as such. They also have a `pop` and
        `push` annotation respectively, which links the push node to the pop
        node and vice versa.
  """
  push = copy.deepcopy(PUSH_STACK)
  pop = copy.deepcopy(POP_STACK)
  anno.setanno(push, 'pop', pop)
  anno.setanno(push, 'gen_push', True)
  anno.setanno(pop, 'push', push)
  op_id = _generate_op_id()
  return push, pop, op_id


class ReverseAD(object):
  """Generate a primal and adjoint for a given AST tree.

  This class walks the AST recursively and for each node returns a new primal
  and an adjoint. Note that it relies on function calls being resolved by the
  `resolve_calls` function.

  Each created node is annoted with its corresponding primal or adjoint in
  the `pri` and `adj` annotations respectively.

  Args:
    wrt: A tuple of argument indices with respect to which the gradient should
        be taken.
    preserve_result: A boolean indicating whether or not the generated gradient
        function should also return the output of the original function.
    check_dims: A boolean indicating whether the seed derivatives should have
        their dimensions checked to match their primal counterpart.

  Attributes:
    required: List of user-defined functions that the primal calls.
        The primals and adjoints of this function need to be available in the
        global namespace.
  """

  def __init__(self, wrt, preserve_result, check_dims, checkpoint_config=None):
    self.required = []
    self.wrt = wrt
    self.preserve_result = preserve_result
    self.check_dims = check_dims
    # Phase 2: Checkpointing configuration
    self.checkpoint_config = checkpoint_config or {}

  def visit(self, node):
    """Visit a node.

    This method is largely modelled after the ast.NodeTransformer class.

    Args:
      node: The node to visit.

    Returns:
      A tuple of the primal and adjoint, each of which is a node or a list of
      nodes.
    """
    method = 'visit_' + node.__class__.__name__
    if not hasattr(self, method):
      raise ValueError('Unknown node type: %s' % node.__class__.__name__)
    visitor = getattr(self, method)

    # If this node is a statement, inform all child nodes what the active
    # variables in this statement are
    if anno.hasanno(node, 'active_in'):
      self.active_variables = anno.getanno(node, 'active_in')
    pri, adj = visitor(node)

    # Annotate primal and adjoint statements
    if isinstance(pri, gast.AST):
      anno.setdefaultanno(pri, 'adj', adj)
    else:
      for node in pri:
        anno.setdefaultanno(node, 'adj', adj)
    if isinstance(adj, gast.AST):
      anno.setdefaultanno(adj, 'pri', pri)
    else:
      for node in adj:
        anno.setdefaultanno(node, 'pri', pri)

    return pri, adj

  @property
  def stack(self):
    if not hasattr(self, '_stack'):
      self._stack = quoting.quote(self.namer.unique(naming.STACK_NAME))
    return ast_.copy_node(self._stack)

  @property
  def substack(self):
    if not hasattr(self, '_substack'):
      self._substack = quoting.quote(self.namer.unique(naming.SUBSTACK_NAME))
    return ast_.copy_node(self._substack)

  def is_active(self, node):
    """Checks whether a statement is active.

    An assignment is active when its right hand side contains active
    variables.

    Args:
      node: an instance of gast.Assign

    Returns:
      Whether the statement is active.
    """
    # Special case: If the right hand side is a pop statement, we want to
    # process it
    if (isinstance(node.value, gast.Call) and
        anno.getanno(node.value, 'func', False) == utils.pop):
      return True
    for succ in gast.walk(node.value):
      if (isinstance(succ, gast.Name) and isinstance(succ.ctx, gast.Load) and
          succ.id in self.active_variables):
        return True
    return False

  def visit_FunctionDef(self, node):
    # Construct a namer to guarantee we create unique names that don't
    # override existing names
    self.namer = naming.Namer.build(node)

    # Check that this function has exactly one return statement at the end
    return_nodes = [n for n in gast.walk(node) if isinstance(n, gast.Return)]
    if ((len(return_nodes) > 1) or not isinstance(node.body[-1], gast.Return)):
      raise ValueError('function must have exactly one return statement')
    return_node = ast_.copy_node(return_nodes[0])

    # Perform AD on the function body
    body, adjoint_body = self.visit_statements(node.body[:-1])

    # Annotate the first statement of the primal and adjoint as such
    if body:
      body[0] = comments.add_comment(body[0], 'Beginning of forward pass')
    if adjoint_body:
      adjoint_body[0] = comments.add_comment(
          adjoint_body[0], 'Beginning of backward pass')

    # Before updating the primal arguments, extract the arguments we want
    # to differentiate with respect to
    dx = gast.Tuple([create.create_grad(node.args.args[i], self.namer)
                     for i in self.wrt], ctx=gast.Load())

    if self.preserve_result:
      # Append an extra Assign operation to the primal body
      # that saves the original output value
      stored_result_node = quoting.quote(self.namer.unique('result'))
      assign_stored_result = template.replace(
          'result=orig_result',
          result=stored_result_node,
          orig_result=return_node.value)
      body.append(assign_stored_result)
      dx.elts.append(stored_result_node)


    for _dx in dx.elts:
      _dx.ctx = gast.Load()
    return_dx = gast.Return(value=dx)

    # We add the stack as first argument of the primal
    node.args.args = [self.stack] + node.args.args

    # Rename the function to its primal name
    func = anno.getanno(node, 'func')
    node.name = naming.primal_name(func, self.wrt)

    # The new body is the primal body plus the return statement
    node.body = body + node.body[-1:]

    # Find the cost; the first variable of potentially multiple return values
    # The adjoint will receive a value for the initial gradient of the cost
    y = node.body[-1].value
    if isinstance(y, gast.Tuple):
      y = y.elts[0]
    dy = gast.Name(id=self.namer.grad(y.id), ctx=gast.Param(),
                   annotation=None)

    if self.check_dims:

      def shape_match_template(primal, adjoint):
        assert tangent.shapes_match(
            primal, adjoint
        ), 'Shape mismatch between return value (%s) and seed derivative (%s)' % (
            numpy.shape(primal), numpy.shape(adjoint))

      shape_check = template.replace(shape_match_template, primal=y, adjoint=dy)
      adjoint_body = shape_check + adjoint_body

    # Construct the adjoint
    adjoint_template = grads.adjoints[gast.FunctionDef]
    adjoint, = template.replace(adjoint_template, namer=self.namer,
                                adjoint_body=adjoint_body, return_dx=return_dx)
    adjoint.args.args.extend([self.stack, dy])
    adjoint.args.args.extend(node.args.args[1:])
    adjoint.name = naming.adjoint_name(func, self.wrt)

    return node, adjoint

  def visit_statements(self, nodes):
    """Generate the adjoint of a series of statements."""
    primals, adjoints = [], collections.deque()
    for node in nodes:
      primal, adjoint = self.visit(node)
      if not isinstance(primal, list):
        primal = [primal]
      if not isinstance(adjoint, list):
        adjoint = [adjoint]
      # Methods will return `None` if the node is to be removed, so remove them
      primals.extend(filter(None, primal))
      # We reverse the order of the adjoints, but not the statements in
      # the adjoint itself
      adjoints.extendleft(filter(None, adjoint[::-1]))
    return primals, list(adjoints)

  def visit_For(self, node):
    if node.orelse:
      raise ValueError

    # Construct the primal and adjoint of the loop
    body, adjoint_body = self.visit_statements(node.body)

    # We create a loop counter which will be pushed on the stack
    push, pop, op_id = get_push_pop()
    counter = self.namer.counter()

    # In `for i in range ...` the variable `i` is the target, which we
    # temporarily set aside each iteration to push to the stack later
    push_target, pop_target, op_id_target = get_push_pop()
    tmp_target = create.create_temp(node.target, self.namer)

    # Phase 2: Check if we should use checkpointing for this loop
    use_checkpointing = self._should_checkpoint_loop(node)

    if use_checkpointing:
      # Use checkpointed templates - Phase 3++: selective storage
      # Create variables for checkpoint management
      checkpoint_dict_var = create.create_temp(gast.Name(id='checkpoint_dict'), self.namer)
      checkpoint_positions_var = create.create_temp(gast.Name(id='checkpoint_positions'), self.namer)

      primal_template = grads.primals_checkpointed[gast.For]
      primal = template.replace(
          primal_template,
          body=body,
          i=counter,
          push=push,
          target=node.target,
          iter_=node.iter,
          push_target=push_target,
          _target=tmp_target,
          _stack=self.stack,
          op_id_iter=op_id,
          op_id_target=op_id_target,
          _checkpoint_dict=checkpoint_dict_var,
          _checkpoint_positions_list=checkpoint_positions_var)

      adjoint_template = grads.adjoints_checkpointed[gast.For]
      adjoint = template.replace(
          adjoint_template,
          adjoint_body=adjoint_body,
          i=counter,
          pop=pop,
          pop_target=pop_target,
          target=ast_.copy_node(node.target),
          _stack=self.stack,
          op_id_iter=op_id,
          op_id_target=op_id_target,
          _checkpoint_dict=checkpoint_dict_var,
          _checkpoint_positions_list=checkpoint_positions_var)
    else:
      # Use standard templates
      primal_template = grads.primals[gast.For]
      primal = template.replace(
          primal_template,
          body=body,
          i=counter,
          push=push,
          target=node.target,
          iter_=node.iter,
          push_target=push_target,
          _target=tmp_target,
          _stack=self.stack,
          op_id_iter=op_id,
          op_id_target=op_id_target)

      adjoint_template = grads.adjoints[gast.For]
      adjoint = template.replace(
          adjoint_template,
          adjoint_body=adjoint_body,
          i=counter,
          pop=pop,
          pop_target=pop_target,
          target=ast_.copy_node(node.target),
          _stack=self.stack,
          op_id_iter=op_id,
          op_id_target=op_id_target)

    return primal, adjoint

  def _should_checkpoint_loop(self, node):
    """Decide if this loop should use checkpointing.

    Phase 2: Conservative approach - only checkpoint if explicitly enabled
    and loop length can be determined.

    Args:
      node: AST For node

    Returns:
      bool: True if checkpointing should be used
    """
    # Check if checkpointing is enabled
    if not self.checkpoint_config.get('enabled', False):
      return False

    # Try to estimate loop length
    loop_length = self._estimate_loop_length(node.iter)

    if loop_length is None:
      # Unknown length - don't checkpoint in Phase 2 (conservative)
      return False

    # Use checkpointing if length exceeds threshold
    threshold = self.checkpoint_config.get('min_length', 100)
    return loop_length >= threshold

  def _estimate_loop_length(self, iter_node):
    """Try to statically determine loop length from range() calls.

    Args:
      iter_node: AST node representing the loop iterator

    Returns:
      int or None: Estimated loop length, or None if unknown
    """
    # Handle range(n) calls
    if isinstance(iter_node, gast.Call):
      if isinstance(iter_node.func, gast.Name) and iter_node.func.id == 'range':
        if len(iter_node.args) == 1:
          # range(n)
          # Python 3.8+ uses gast.Constant, older versions use gast.Num
          arg = iter_node.args[0]
          if hasattr(gast, 'Num') and isinstance(arg, gast.Num):
            return arg.n
          elif isinstance(arg, gast.Constant):
            return arg.value
        elif len(iter_node.args) == 2:
          # range(start, stop)
          stop_val = None
          start_val = 0

          # Get stop value
          stop_arg = iter_node.args[1]
          if hasattr(gast, 'Num') and isinstance(stop_arg, gast.Num):
            stop_val = stop_arg.n
          elif isinstance(stop_arg, gast.Constant):
            stop_val = stop_arg.value

          # Get start value
          start_arg = iter_node.args[0]
          if hasattr(gast, 'Num') and isinstance(start_arg, gast.Num):
            start_val = start_arg.n
          elif isinstance(start_arg, gast.Constant):
            start_val = start_arg.value

          if stop_val is not None:
            return stop_val - start_val

    # Unknown length
    return None

  def visit_While(self, node):
    if node.orelse:
      raise ValueError

    body, adjoint_body = self.visit_statements(node.body)

    # We create a loop counter which will be pushed on the stack
    push, pop, op_id = get_push_pop()
    counter = self.namer.counter()

    primal_template = grads.primals[gast.While]
    primal = template.replace(
        primal_template,
        namer=self.namer,
        body=body,
        i=counter,
        push=push,
        test=node.test,
        _stack=self.stack,
        op_id=op_id)

    adjoint_template = grads.adjoints[gast.While]
    adjoint = template.replace(
        adjoint_template,
        namer=self.namer,
        adjoint_body=adjoint_body,
        i=counter,
        pop=pop,
        _stack=self.stack,
        op_id=op_id)

    return primal, adjoint

  def visit_With(self, node):
    """Deal with the special with insert_grad_of(x) statement."""
    if ast_.is_insert_grad_of_statement(node):
      primal = []
      adjoint = node.body
      if isinstance(adjoint[0], gast.With):
        _, adjoint = self.visit(adjoint[0])
      node.body[0] = comments.add_comment(node.body[0], 'Inserted code')
      # Rename the gradients
      replacements = {}
      for item in node.items:
        if (not isinstance(item.context_expr.args[0], gast.Name) or
            not isinstance(item.optional_vars, gast.Name)):
          raise ValueError
        replacements[item.optional_vars.id] = create.create_grad(
            item.context_expr.args[0], self.namer)
      template.ReplaceTransformer(replacements).visit(node)
      return primal, adjoint
    else:
      return node, []

  def visit_If(self, node):
    # Get the primal and adjoint of the blocks
    body, adjoint_body = self.visit_statements(node.body)
    orelse, adjoint_orelse = self.visit_statements(node.orelse)

    # We will store the condition on the stack
    cond = self.namer.cond()
    push, pop, op_id = get_push_pop()

    # Fill in the templates
    primal_template = grads.primals[gast.If]
    primal = template.replace(
        primal_template,
        body=body,
        cond=cond,
        test=node.test,
        orelse=orelse,
        push=push,
        _stack=self.stack,
        op_id=op_id)
    adjoint_template = grads.adjoints[gast.If]
    adjoint = template.replace(
        adjoint_template,
        cond=cond,
        adjoint_body=adjoint_body,
        adjoint_orelse=adjoint_orelse,
        pop=pop,
        _stack=self.stack,
        op_id=op_id)
    return primal, adjoint

  def visit_IfExp(self, node):
    """Handle conditional expressions (ternary operator): body if test else orelse."""
    # Store the condition on the stack so we can use it in the adjoint
    cond_var = self.namer.cond()
    push, pop, op_id = get_push_pop()

    # Save the current target and create temps for each branch
    saved_target = self.target
    saved_orig_target = self.orig_target

    # Visit body branch - the result goes into orig_target
    self.target = create.create_temp(self.orig_target, self.namer)
    self.orig_target = saved_orig_target
    body_result, body_adjoint = self.visit(node.body)
    body_temp = self.target

    # Visit orelse branch - the result goes into orig_target
    self.target = create.create_temp(self.orig_target, self.namer)
    self.orig_target = saved_orig_target
    orelse_result, orelse_adjoint = self.visit(node.orelse)
    orelse_temp = self.target

    # Restore target
    self.target = saved_target
    self.orig_target = saved_orig_target

    # Extract the expression and preparatory statements from the visited branches
    # If the result is a list (from visiting nested IfExp), the last element should be
    # an Assign node, and we need the RHS expression
    body_stmts = []
    if isinstance(body_result, list):
      body_stmts = body_result[:-1]  # All but last
      # Last item should be an Assign, extract its value
      if isinstance(body_result[-1], gast.Assign):
        body_expr = body_result[-1].value
      else:
        body_expr = body_result[-1]
    else:
      body_expr = body_result

    orelse_stmts = []
    if isinstance(orelse_result, list):
      orelse_stmts = orelse_result[:-1]  # All but last
      # Last item should be an Assign, extract its value
      if isinstance(orelse_result[-1], gast.Assign):
        orelse_expr = orelse_result[-1].value
      else:
        orelse_expr = orelse_result[-1]
    else:
      orelse_expr = orelse_result

    # Primal: save condition, compute ternary, push condition
    save_cond = template.replace('cond = test', cond=cond_var, test=node.test)
    push_cond = template.replace('push(_stack, cond, op_id)',
                                 push=push, _stack=self.stack,
                                 cond=cond_var, op_id=op_id)

    # Create IfExp using the saved condition
    ifexp_expr = gast.IfExp(
        test=gast.Name(id=cond_var, ctx=gast.Load(), annotation=None),
        body=body_expr,
        orelse=orelse_expr)

    # Create the assignment
    assign = gast.Assign(
        targets=[ast_.copy_node(self.orig_target)],
        value=ifexp_expr)

    # Combine all primal statements
    primal = body_stmts + orelse_stmts + [save_cond, push_cond, assign]

    # Adjoint: pop condition, then conditionally execute the branch adjoints
    # The gradient of the result needs to be routed to the chosen branch
    pop_cond = template.replace('cond = pop(_stack, op_id)',
                                cond=cond_var, pop=pop,
                                _stack=self.stack, op_id=op_id)

    # Create conditional adjoint: if cond was true, execute body_adjoint, else orelse_adjoint
    # Also need to initialize the branch temp gradients from the result gradient
    init_body_grad = template.replace(
        'd[body_temp] = d[result]',
        namer=self.namer,
        replace_grad=template.Replace.FULL,
        body_temp=body_temp,
        result=self.target)

    init_orelse_grad = template.replace(
        'd[orelse_temp] = d[result]',
        namer=self.namer,
        replace_grad=template.Replace.FULL,
        orelse_temp=orelse_temp,
        result=self.target)

    # Ensure adjoints are lists and filter out any that are empty lists
    body_adj_list = body_adjoint if isinstance(body_adjoint, list) else ([body_adjoint] if body_adjoint else [])
    orelse_adj_list = orelse_adjoint if isinstance(orelse_adjoint, list) else ([orelse_adjoint] if orelse_adjoint else [])

    adj_if = gast.If(
        test=gast.Name(id=cond_var, ctx=gast.Load(), annotation=None),
        body=[init_body_grad] + body_adj_list,
        orelse=[init_orelse_grad] + orelse_adj_list)

    adjoint = [pop_cond, adj_if]

    return primal, adjoint

  def visit_Attribute(self, node):
    raise ValueError('attributes are not yet supported for gradients')

  def visit_AugAssign(self, node):
    raise ValueError('AugAssign not supported for gradients. '
      'Should have been removed by ANF. This is a bug.')

  def visit_Assign(self, node):
    """Visit assignment statement."""
    if len(node.targets) != 1:
      raise ValueError('no support for chained assignment')

    # Before the node gets modified, get a source code representation
    # to add as a comment later on
    if anno.hasanno(node, 'pre_anf'):
      orig_src = anno.getanno(node, 'pre_anf')
    else:
      orig_src = quoting.unquote(node)

    # Set target for the RHS visitor to access
    self.orig_target = ast_.copy_node(node.targets[0])

    # If we know we're going to be putting another stack on the stack,
    # we should try to make that explicit
    if isinstance(node.value, gast.Call) and \
        anno.hasanno(node.value, 'func') and \
        anno.getanno(node.value, 'func') in (utils.Stack, utils.pop_stack):
      push, pop, op_id = get_push_pop_stack()
    else:
      push, pop, op_id = get_push_pop()
    push_stack, pop_stack, op_id_stack = get_push_pop_stack()

    # Every assignment statement requires us to store the pre-value, and in the
    # adjoint restore the value, and reset the gradient
    store = template.replace(
        'push(_stack, y, op_id)',
        push=push,
        y=self.orig_target,
        _stack=self.stack,
        op_id=op_id)
    create_substack = template.replace(
        'substack = tangent.Stack()', substack=self.substack)
    store_substack = template.replace(
        'push(stack, substack, op_id)',
        push=push_stack,
        stack=self.stack,
        substack=self.substack,
        op_id=op_id_stack)
    restore = template.replace(
        'y = pop(_stack, op_id)',
        _stack=self.stack,
        pop=pop,
        y=ast_.copy_node(self.orig_target),
        op_id=op_id)
    restore_substack = template.replace(
        'substack = pop(stack, op_id)',
        pop=pop_stack,
        stack=self.stack,
        substack=self.substack,
        op_id=op_id_stack)
    reset = template.replace(
        'd[y] = init_grad(y, allow_lazy_initializer=True)',
        y=self.orig_target,
        init_grad=utils.INIT_GRAD,
        namer=self.namer,
        replace_grad=template.Replace.FULL)

    # If there are no active nodes, we don't need to find an adjoint
    # We simply store and restore the state, and reset the gradient
    if not self.is_active(node):
      return [store, node], [restore, reset]

    # We create a temporary variable for the target that the RHS can use
    self.target = create.create_temp(self.orig_target, self.namer)
    create_tmp = template.replace(
        'tmp = y', tmp=self.target, y=self.orig_target)

    # Get the primal and adjoint of the RHS expression
    try:
      fx, adjoint_rhs = self.visit(node.value)
    except ValueError as e:
      context = [t.id if hasattr(t, 'id') else t for t in node.targets]
      raise ValueError(
          'Failed to process assignment to: %s. Error: %s' % (context, e))
    if not isinstance(adjoint_rhs, list):
      adjoint_rhs = [adjoint_rhs]

    # Walk the RHS adjoint AST to find temporary adjoint variables to sum
    accumulations = []
    for n in adjoint_rhs:
      for succ in gast.walk(n):
        if anno.hasanno(succ, 'temp_adjoint_var'):
          xi = anno.getanno(succ, 'temp_adjoint_var')
          dxi_partial = ast_.copy_node(succ)
          accumulations.append(template.replace(
              'd[xi] = add_grad(d[xi], dxi_partial)',
              namer=self.namer, replace_grad=template.Replace.FULL,
              xi=xi, dxi_partial=dxi_partial, add_grad=utils.ADD_GRAD))

    # The primal consists of storing the state and then performing the
    # assignment with the new primal.
    # The primal `fx` may be optionally (but rarely) redefined when the
    # adjoint is generated, in `fx, adjoint_rhs = self.visit(node.value)`.
    # If we see that the primal value is an Assign node, or a list of nodes
    # (with at least one being an Assign) node, we allow the primal to change.
    # Otherwise, we'll build our own Assign node.
    if isinstance(fx, gast.Assign):
      assign = [fx]
    elif (isinstance(fx, list) and
          any([isinstance(ifx, gast.Assign) for ifx in fx])):
      assign = fx
    else:
      assign = template.replace(
          'y = fx', y=ast_.copy_node(self.orig_target), fx=fx)
      assign = [assign]
    primal = [store, create_substack, store_substack] + assign

    # The adjoint involves creating the temporary, restoring the store,
    # calculating the adjoint, resetting the gradient, and finally accumulating
    # the partials
    adjoint = [create_tmp, restore_substack, restore
              ] + adjoint_rhs + [reset] + accumulations

    # If the LHS is a subscript assignment with variable index, we need to
    # store and restore that as well
    if (isinstance(self.orig_target, gast.Subscript) and
        isinstance(self.orig_target.slice.value, gast.Name)):
      push, pop, op_id = get_push_pop()
      i = self.orig_target.slice.value
      push_index = template.replace(
          'push(_stack, i, op_id)',
          push=push,
          i=i,
          _stack=self.stack,
          op_id=op_id)
      pop_index = template.replace(
          'i = pop(_stack, op_id)',
          pop=pop,
          i=i,
          _stack_=self.stack,
          op_id=op_id)

      primal.insert(len(primal), push_index)
      adjoint.insert(0, pop_index)

    # Add a comment in the backwards pass, indicating which
    # lines in the forward pass generated the adjoint
    for i, adj in enumerate(adjoint):
      adjoint[i] = comments.add_comment(adj, 'Grad of: %s' % orig_src)

    return primal, adjoint

  def visit_Pass(self, node):
    return node, []

  def visit_Subscript(self, node):
    """Handle subscript access in reverse mode.

    For `result = container[i]`, the adjoint is `d_container[i] = d_result`.
    This assigns the gradient of the result to element i of the container's gradient.

    For immutable types (JAX, TensorFlow), we use tangent.update_grad_at_index()
    which provides functional updates instead of in-place assignment.
    """
    # Create the gradient of the container: d_container
    grad_container = create.create_grad(node.value, self.namer)
    grad_container.ctx = gast.Load()

    # Create the gradient of the result: d_result
    rhs_grad = create.create_grad(self.target, self.namer)
    rhs_grad.ctx = gast.Load()

    # Generate: d_container = tangent.update_grad_at_index(d_container, index, d_result)
    # This handles both mutable (NumPy) and immutable (JAX, TensorFlow) types
    update_call = gast.Call(
        func=gast.Attribute(
            value=gast.Name(id='tangent', ctx=gast.Load(), annotation=None),
            attr='update_grad_at_index',
            ctx=gast.Load()),
        args=[
            grad_container,  # grad_array parameter
            node.slice,      # index parameter
            rhs_grad         # value parameter
        ],
        keywords=[])

    # Assign the result back to d_container
    grad_container_store = create.create_grad(node.value, self.namer)
    grad_container_store.ctx = gast.Store()
    adjoint = gast.Assign(targets=[grad_container_store], value=update_call)

    return node, adjoint

  def visit_Name(self, node):
    adjoint = template.replace('d[x] = tangent.copy(d[y])',
                               namer=self.namer, x=node, y=self.target)
    return node, adjoint

  def visit_Num(self, node):
    return node, []

  def visit_Constant(self, node):
    """Handle gast.Constant nodes (Python 3.8+)."""
    return node, []

  def visit_Tuple(self, node):
    return self.visit_container(node)

  def visit_List(self, node):
    return self.visit_container(node)

  def visit_Dict(self, node):
    """Handle dictionary construction.

    For `result = {'a': x, 'b': y}`, we propagate gradients:
    d[x] = d_result['a']
    d[y] = d_result['b']

    Thanks to ANF transformation, ALL dict values (including nested dicts
    and complex expressions) are extracted to temp variables first.
    So all values here are simple Names.
    """
    adjoint = []

    for key_node, value_node in zip(node.keys, node.values):
      # Create d_result[key] for RHS: gradient of dict element
      grad_target = create.create_grad(self.target, self.namer)
      grad_target.ctx = gast.Load()
      grad_subscript = gast.Subscript(
          value=grad_target,
          slice=key_node,
          ctx=gast.Load())

      # Create the assignment: d_value = d_result[key]
      adjoint.append(template.replace('d[x] = rhs', namer=self.namer,
                                      x=value_node, rhs=grad_subscript))

    return node, adjoint

  def visit_container(self, node):
    """Handle tuple/list unpacking in reverse mode.

    For assignments like `a, b = x**2, x*3`, we need to propagate gradients
    from each element to its corresponding value. The gradient of the container
    is accessed by index: d[x] = d[container][i]
    """
    adjoint = []
    for i, elt in enumerate(node.elts):
      # Create d[t][i] manually: subscript the gradient of the whole container
      grad_target = create.create_grad(self.target, self.namer)
      grad_target.ctx = gast.Load()
      index_node = gast.Constant(value=i, kind=None)
      grad_target_subscript = gast.Subscript(
          value=grad_target,
          slice=index_node,
          ctx=gast.Load())

      # Now create the assignment: d[x] = d[t][i]
      adjoint.append(template.replace('d[x] = rhs', namer=self.namer,
                                      x=elt, rhs=grad_target_subscript))
    return node, adjoint

  def visit_Pass(self, node):
    return node, []

  def visit_Break(self, node):
    """Handle break statements.

    Break statements are preserved in the primal but have no adjoint.
    The loop unwinding will handle the control flow correctly.
    """
    return node, []

  def visit_Continue(self, node):
    """Handle continue statements.

    Continue statements are preserved in the primal but have no adjoint.
    The loop unwinding will handle the control flow correctly.
    """
    return node, []

  def visit_Try(self, node):
    """Handle try/except/finally blocks.

    This is a basic implementation that processes the try body and handlers,
    but does not capture exception state on the stack. This means gradients
    will flow through whichever path was actually executed at runtime.

    Limitations:
    - Does not checkpoint which handler was executed
    - Finally blocks run in both forward and backward pass
    - Exception state is not preserved for the adjoint
    """
    # Visit the try body
    body, adjoint_body = self.visit_statements(node.body)

    # Visit exception handlers
    handlers = []
    adjoint_handlers = []
    for handler in node.handlers:
      handler_body, handler_adjoint = self.visit_statements(handler.body)
      # Ensure bodies are not empty (Python syntax requirement)
      if not handler_body:
        handler_body = [gast.Pass()]
      if not handler_adjoint:
        handler_adjoint = [gast.Pass()]

      # Create new handler with transformed body
      new_handler = gast.ExceptHandler(
          type=handler.type,
          name=handler.name,
          body=handler_body)
      handlers.append(new_handler)

      # Create adjoint handler
      adjoint_handler = gast.ExceptHandler(
          type=handler.type,
          name=handler.name,
          body=handler_adjoint)
      adjoint_handlers.append(adjoint_handler)

    # Visit orelse (executed if no exception)
    orelse, adjoint_orelse = self.visit_statements(node.orelse)

    # Visit finally block
    finalbody, adjoint_finalbody = self.visit_statements(node.finalbody)

    # Ensure no empty blocks
    if not body:
      body = [gast.Pass()]
    if not adjoint_body:
      adjoint_body = [gast.Pass()]

    # Build primal try statement
    primal = gast.Try(
        body=body,
        handlers=handlers,
        orelse=orelse,
        finalbody=finalbody)

    # Build adjoint try statement
    # Note: The adjoint also uses try/except structure to mirror the primal's control flow
    adjoint = gast.Try(
        body=adjoint_body,
        handlers=adjoint_handlers,
        orelse=adjoint_orelse,
        finalbody=adjoint_finalbody)

    return primal, adjoint

  def visit_BinOp(self, node):
    op = type(node.op)
    if op not in grads.adjoints:
      raise ValueError('unknown binary operator')
    adjoint_template = grads.adjoints[op]
    adjoint = template.replace(adjoint_template,
                               namer=self.namer,
                               x=node.left, y=node.right, z=self.target)
    return node, adjoint

  def visit_UnaryOp(self, node):
    op = type(node.op)
    if op not in grads.adjoints:
      raise ValueError('unknown unary operator')
    adjoint_template = grads.adjoints[op]
    adjoint = template.replace(adjoint_template, namer=self.namer,
                               x=node.operand, y=self.target)
    return node, adjoint

  def visit_Compare(self, node):
    return node, []

  def visit_BoolOp(self, node):
    """Boolean operations (and, or) are not differentiable."""
    return node, []

  def visit_Assert(self, node):
    return node, []

  def primal_and_adjoint_for_tracing(self, node):
    """Build the primal and adjoint of a traceable function.

    Args:
      node: ast.Call node of a function we wish to trace, instead of transform

    Returns:
      primal: new ast.Assign node to replace the original primal call
      adjoint: new ast.Assign node using the VJP generated in primal to
        calculate the adjoint.
    """
    primal_template = grads.primals[tracing.Traceable]
    adjoint_template = grads.adjoints[tracing.Traceable]

    # Prep
    to_pack = node.args
    target = ast_.copy_node(self.orig_target)
    vjp = quoting.quote(self.namer.unique('%s_grad' % node.func.id))
    tmp = create.create_temp(quoting.quote('tmp'), self.namer)
    assert len(node.keywords) == 0

    # Full replacement of primal
    # TODO: do we need to set 'pri_call' on this?
    primal = template.replace(
        primal_template,
        namer=self.namer,
        result=target,
        fn=node.func,
        tmp=tmp,
        vjp=vjp,
        args=gast.Tuple(elts=to_pack, ctx=gast.Load()))

    # Building adjoint using the vjp generated with the primal
    dto_pack = gast.Tuple(
        elts=[create.create_temp_grad(arg, self.namer) for arg in to_pack],
        ctx=gast.Store())

    adjoint = template.replace(
        adjoint_template,
        namer=self.namer,
        result=target,
        vjp=vjp,
        dargs=dto_pack)

    return primal, adjoint

  def visit_Call(self, node):
    """Create adjoint for call.

    We don't allow unpacking of parameters, so we know that each argument
    gets passed in explicitly, allowing us to create partials for each.
    However, templates might perform parameter unpacking (for cases where
    the number of arguments is variable) and express their gradient as a
    tuple. In this case, we have to unpack this tuple of partials.
    """
    # Find the function we are differentiating
    func = anno.getanno(node, 'func')

    if func in non_differentiable.NON_DIFFERENTIABLE:
      return node, []

    if func == tracing.Traceable:
      return self.primal_and_adjoint_for_tracing(node)

    if func in grads.UNIMPLEMENTED_ADJOINTS:
      raise errors.ReverseNotImplementedError(func)


    # If we don't have an adjoint, we will have to step into the called
    # function and differentiate it
    if func not in grads.adjoints:
      # Only consider Name nodes as potential active arguments
      # Other node types (Constant, Compare, etc.) are not differentiable
      active_args = tuple(i for i, arg in enumerate(node.args)
                          if isinstance(arg, gast.Name) and arg.id in self.active_variables)

      already_counted = False
      for f, a in self.required:
        if f.__name__ == func.__name__ and set(a) == set(active_args):
          already_counted = True
          break
      if not already_counted:
        self.required.append((func, active_args))

      pri_name = naming.primal_name(func, active_args)
      pri_call = gast.Call(
          func=gast.Name(id=pri_name, ctx=gast.Load(), annotation=None),
          args=[self.substack] + node.args,
          keywords=node.keywords)
      anno.setanno(pri_call, 'pri_call', True)

      dy = create.create_grad(self.target, self.namer)
      dy.ctx = gast.Load()
      dx = create.create_grad(node.args[0], self.namer)
      dx.ctx = gast.Store()
      adj_name = naming.adjoint_name(func, active_args)
      adj_call = gast.Call(
          func=gast.Name(id=adj_name, ctx=gast.Load(), annotation=None),
          args=[self.substack, dy] + node.args,
          keywords=node.keywords)
      anno.setanno(adj_call, 'adj_call', True)
      adjoint = [template.replace('dxs = dfx', namer=self.namer, dfx=adj_call)]
      for j, i in enumerate(active_args):
        adjoint.append(template.replace('d[x] = dxs[i]', namer=self.namer,
                                        x=node.args[i].id, i=gast.Constant(value=j, kind=None)))
      return pri_call, adjoint

    # We have a template for the gradient that we need to fill in
    template_ = grads.adjoints[func]

    # Match the function call to the template
    sig = funcsigs.signature(template_)
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    kwargs = dict((keyword.arg, keyword.value) for keyword in node.keywords)
    bound_args = sig.bind(*node.args, **kwargs)

    # Fill in any missing kwargs with the defaults from the template
    args = quoting.parse_function(template_).body[0].args
    kwargs = dict(zip(*map(reversed, [args.args, args.defaults])))
    kwargs.update(dict(zip(args.kwonlyargs, args.kw_defaults)))
    for arg, val in kwargs.items():
      if arg.id not in bound_args.arguments:
        bound_args.arguments[arg.id] = val

    # Let's fill in the template. The first argument is the output, which
    # was stored in a temporary variable
    output_name = six.get_function_code(template_).co_varnames[0]
    arg_replacements = {output_name: ast_.copy_node(self.target)}
    arg_replacements.update(bound_args.arguments)

    # If the template uses *args, then we pack the corresponding inputs
    packing = []
    flags = six.get_function_code(template_).co_flags

    if flags & inspect.CO_VARARGS:
      to_pack = node.args[six.get_function_code(template_).co_argcount - 1:]
      vararg_name = six.get_function_code(template_).co_varnames[-1]
      target = gast.Name(annotation=None, id=vararg_name, ctx=gast.Store())
      value = gast.Tuple(elts=to_pack, ctx=gast.Load())
      packing = [gast.Assign(targets=[target], value=value)]

      # And we fill in the packed tuple into the template
      arg_replacements[six.get_function_code(
          template_).co_varnames[-1]] = target
    adjoint = template.replace(template_, namer=self.namer, **arg_replacements)
    unpacking = []
    if flags & inspect.CO_VARARGS:
      # If the template packs arguments, then we have to unpack the
      # derivatives afterwards
      # We also have to update the replacements tuple then
      dto_pack = [create.create_temp_grad(arg, self.namer)
                  for arg in to_pack]
      value = create.create_grad(target, self.namer)
      target = gast.Tuple(elts=dto_pack, ctx=gast.Store())
      unpacking = [gast.Assign(targets=[target], value=value)]

    return node, packing + adjoint + unpacking

  def visit_Expr(self, node):
    # We need to special-case pushes, e.g. utils.push(_stack,x,op_id)
    adjoint = []
    if (isinstance(node.value, gast.Call) and
        anno.getanno(node.value, 'func') == utils.push):
      orig_src = quoting.unquote(node)
      stack, val, op_id = node.value.args
      push_template = grads.adjoints[utils.push]
      adjoint_rhs = template.replace(
          push_template, namer=self.namer, stack=stack, val=val, op_id=op_id)

      # Walk the RHS adjoint AST to find temporary adjoint variables to
      # sum
      accumulation = template.replace(
          'd[xi] = add_grad(d[xi], dxi_partial)', namer=self.namer,
          replace_grad=template.Replace.FULL, xi=val,
          dxi_partial=ast_.copy_node(adjoint_rhs[0].targets[0]),
          add_grad=utils.ADD_GRAD)
      adjoint = adjoint_rhs + [accumulation]
      for i, adj in enumerate(adjoint):
        adjoint[i] = comments.add_comment(adj, 'Grad of: %s' % orig_src)
    return node, adjoint


def reverse_ad(node, wrt, preserve_result, check_dims, checkpoint_config=None):
  """Perform reverse-mode AD on an AST.

  This function analyses the AST to determine which variables are active and
  proceeds by taking the naive derivative. Before returning the primal and
  adjoint it annotates push and pop statements as such.

  Args:
    node: A `FunctionDef` AST node.
    wrt: A tuple of argument indices with respect to which we take the
        derivative.
    preserve_result: A boolean indicating whether the generated
        derivative function should also return the original return value.
    check_dims: A boolean indicating whether the seed derivatives should have
        their dimensions checked to match their primal counterpart.
    checkpoint_config: Optional dictionary with checkpointing configuration.
        Keys: 'enabled' (bool), 'min_length' (int), 'num_checkpoints' (int or None)


  Returns:
    mod: A `Module` node containing the naive primal and adjoint of the
        function which can be fed to the `split` and `joint` functions.
    required: A list of tuples of functions and argument indices. These
        functions were called by the function but did not have an adjoint.
  """
  if not isinstance(node, gast.FunctionDef):
    raise TypeError
  # Activity analysis
  cfg.forward(node, cfg.Active(wrt))

  ad = ReverseAD(wrt, preserve_result, check_dims, checkpoint_config)
  pri, adj = ad.visit(node)
  mod = gast.Module(body=[pri, adj])
  mod = annotate.find_stacks(mod)
  return mod, ad.required, ad.stack


def store_state(node, reaching, defined, stack):
  """Push the final state of the primal onto the stack for the adjoint.

  Python's scoping rules make it possible for variables to not be defined in
  certain blocks based on the control flow path taken at runtime. In order to
  make sure we don't try to push non-existing variables onto the stack, we
  defined these variables explicitly (by assigning `None` to them) at the
  beginning of the function.

  All the variables that reach the return statement are pushed onto the
  stack, and in the adjoint they are popped off in reverse order.

  Args:
    node: A module with the primal and adjoint function definitions as returned
        by `reverse_ad`.
    reaching: The variable definitions that reach the end of the primal.
    defined: The variables defined at the end of the primal.
    stack: The stack node to use for storing and restoring state.

  Returns:
    node: A node with the requisite pushes and pops added to make sure that
        state is transferred between primal and adjoint split motion calls.
  """
  defs = [def_ for def_ in reaching if not isinstance(def_[1], gast.arguments)]
  if not len(defs):
    return node
  reaching, original_defs = zip(*defs)

  # Explicitly define variables that might or might not be in scope at the end
  assignments = []
  for id_ in set(reaching) - defined:
    assignments.append(quoting.quote('{} = None'.format(id_)))

  # Store variables at the end of the function and restore them
  store = []
  load = []
  for id_, def_ in zip(reaching, original_defs):
    # If the original definition of a value that we need to store
    # was an initialization as a stack, then we should be using `push_stack`
    # to store its state, and `pop_stack` to restore it. This allows
    # us to avoid doing any `add_grad` calls on the stack, which result
    # in type errors in unoptimized mode (they are usually elided
    # after calling `dead_code_elimination`).
    if isinstance(
        def_, gast.Assign) and 'tangent.Stack()' in quoting.unquote(def_.value):
      push, pop, op_id = get_push_pop_stack()
    else:
      push, pop, op_id = get_push_pop()
    store.append(
        template.replace(
            'push(_stack, val, op_id)',
            push=push,
            val=id_,
            _stack=stack,
            op_id=op_id))
    load.append(
        template.replace(
            'val = pop(_stack, op_id)',
            pop=pop,
            val=id_,
            _stack=stack,
            op_id=op_id))

  body, return_ = node.body[0].body[:-1], node.body[0].body[-1]
  node.body[0].body = assignments + body + store + [return_]
  node.body[1].body = load[::-1] + node.body[1].body

  return node


def split(node, stack):
  """Carry over the state from the primal to the adjoint.

  Args:
    node: A module with the primal and adjoint function definitions as returned
        by `reverse_ad`.
    stack: The stack node to use for storing and restoring state.

  Returns:
    func: A `Module` node with two function definitions containing the primal
        and adjoint respectively.
  """
  node, defined, reaching = _fix(node)

  # Store and restore the state
  node = store_state(node, reaching, defined, stack)

  # Clean up
  anno.clearanno(node)
  return node


def joint(node):
  """Merge the bodies of primal and adjoint into a single function.

  Args:
    node: A module with the primal and adjoint function definitions as returned
        by `reverse_ad`.

  Returns:
    func: A `Module` node with a single function definition containing the
        combined primal and adjoint.
  """
  node, _, _ = _fix(node)
  body = node.body[0].body[:-1] + node.body[1].body
  func = gast.Module(body=[gast.FunctionDef(
      name=node.body[0].name, args=node.body[1].args, body=body,
      decorator_list=[], returns=None)])
  # Clean up
  anno.clearanno(func)
  return func


def _fix(node):
  """Fix the naive construction of the adjont.

  See `fixes.py` for details.

  This function also returns the result of reaching definitions analysis so
  that `split` mode can use this to carry over the state from primal to
  adjoint.

  Args:
    node: A module with the primal and adjoint function definitions as returned
        by `reverse_ad`.

  Returns:
    node: A module with the primal and adjoint function with additional
        variable definitions and such added so that pushes onto the stack and
        gradient accumulations are all valid.
    defined: The variables defined at the end of the primal.
    reaching: The variable definitions that reach the end of the primal.
  """
  # Do reaching definitions analysis on primal and adjoint
  pri_cfg = cfg.CFG.build_cfg(node.body[0])
  defined = cfg.Defined()
  defined.visit(pri_cfg.entry)
  reaching = cfg.ReachingDefinitions()
  reaching.visit(pri_cfg.entry)

  cfg.forward(node.body[1], cfg.Defined())
  cfg.forward(node.body[1], cfg.ReachingDefinitions())

  # Remove pushes of variables that were never defined
  fixes.CleanStack().visit(node)
  fixes.FixStack().visit(node.body[0])

  # Change accumulation into definition if possible
  fixes.CleanGrad().visit(node.body[1])
  # Define gradients that might or might not be defined
  fixes.FixGrad().visit(node.body[1])
  return node, defined.exit, reaching.exit
