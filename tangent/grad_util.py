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
"""Utilities to take derivatives of Python functions.

For notation and theory, please refer to Chapter 3 of "Evaluating Derivatives"
by Griewank and Walther.

We expose a few APIs to do this.
- grad(f): generate gradient of a function f : R^n -> R. Scalar output will be
checked.
- autodiff(f, mode='forward'): generate the forward-mode derivative of a
function f.
    Best for functions f: R^n -> R^m where m >> n
- autodiff(f, mode='reverse'): generate the reverse-mode derivative of a
function f.
  Best for functions f: R^n -> R^m where n >> m.


Forward-mode and reverse-mode are the two main ways to calculate derivatives.
Given a function `F`, with two arguments:

```
Z = F(X, Y)
```

we can calculate in "forward mode", which returns

```
# Forward mode
# dZ, given dX, X, Y
dZ = dF(X, Y, dX))
```

or, we can calculate the derivative of the in "reverse mode", which returns
```
# Reverse mode
# bX, given bZ, X, Y
bX = bF(X, Y, bZ)
```

"""

from __future__ import absolute_import

import enum
import inspect
import gast
import numpy
from tangent import annotations as anno
from tangent import ast as ast_
from tangent import comments
from tangent import compile as compile_
from tangent import control_flow_validator
from tangent import fence
from tangent import forward_ad
from tangent import naming
from tangent import optimization
from tangent import passes
from tangent import quoting
from tangent import reverse_ad

INPUT_DERIVATIVE = enum.Enum('InputDerivative', ('Required', 'DefaultOne', 'DefaultOnes'))

# Import caching utilities
from tangent.function_cache import cached_autodiff, cached_grad


def unwrap_function(func):
    """Unwrap JAX JIT-compiled and custom derivative functions.

    JAX uses __wrapped__ to store the original function in objects like:
    - custom_jvp
    - custom_vjp
    - PjitFunction

    This follows the entire __wrapped__ chain, including plain-function wrappers
    created with functools.wraps. That matters for differentiating a cached
    gradient function a second time (grad-of-grad): the caching layer wraps the
    generated function, and the wrapper - itself a regular function - carries
    Tangent's module globals rather than the numpy/tangent names the generated
    code references. `inspect` unwraps these wrappers when fetching source, so the
    namespace must be taken from the same underlying function.

    Args:
      func: A function that may be wrapped

    Returns:
      The unwrapped function if it has __wrapped__, otherwise the original func
    """
    unwrapped = func
    seen = set()
    while hasattr(unwrapped, '__wrapped__') and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        unwrapped = unwrapped.__wrapped__
    return unwrapped


def autodiff_ast(
    func,
    wrt,
    motion,
    mode,
    preserve_result,
    check_dims,
    verbose,
    checkpoint_config=None,
    reconcile_seed=False,
):
    """Perform AD on a single function and return the AST.

    Args:
      See `grad`.
      checkpoint_config: Optional dictionary with checkpointing configuration.
      reconcile_seed: Whether reverse mode should emit a runtime seed
          reconciliation at the top of the adjoint, so container (pytree) return
          values receive a structurally matching seed (see `reverse_ad`).

    Returns:
      node: The AST of a module containing the adjoint and primal function
          definitions.
      required: A list of non-built in functions that this function called, and
          of which the primals and adjoints need to be made available in order
          for the returned function to run.
    """
    # Parse the function, then lower it through the frontend pass pipeline
    # (see tangent/passes.py for the pass order and its rationale).
    node = quoting.parse_function(func)
    # Fetch the source once and reuse it below. `inspect.getsource` raises
    # OSError when the source is unavailable and TypeError for objects without
    # code; in practice it cannot fail here since parse_function just fetched
    # the same source, but the validators degrade gracefully without it.
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ''
    # Nested defs crash several of the desugaring passes and the reverse transform;
    # reject them up front with a clear error, before any pass sees them.
    node = fence.validate_no_nested_functions(node, source)

    node = passes.run_passes(node, func, source, mode)
    if verbose >= 2:
        print('ANF')
        print(quoting.to_source(node))

    # Validate control flow patterns after ANF transformation
    control_flow_validator.validate_control_flow(node, source, verbose=verbose >= 1)

    if mode == 'reverse':
        node, required, stack = reverse_ad.reverse_ad(
            node.body[0], wrt, preserve_result, check_dims, checkpoint_config, reconcile_seed
        )
        if verbose >= 2:
            print('RAW')
            print(quoting.to_source(node))
        if motion == 'split':
            node = reverse_ad.split(node, stack)
        else:
            node = reverse_ad.joint(node)
        if verbose >= 2:
            print('MOTION')
            print(quoting.to_source(node))
    elif mode == 'forward':
        node, required = forward_ad.forward_ad(node.body[0], wrt, preserve_result, check_dims)
    return node, required


def autodiff_tree(
    func,
    wrt,
    motion,
    mode,
    preserve_result,
    check_dims,
    verbose,
    checkpoint_config=None,
    reconcile_seed=False,
):
    """Perform AD on all functions in a call tree.

    This function walks the call tree and differentiates each function in it. It
    also ensures that the global namespaces that each function in the call tree
    was in are merged.

    The `tangent` and `numpy` packages are added to the namespace here, so that
    the gradient templates can assume that they are present.

    Args:
      See `grad`.
      checkpoint_config: Optional dictionary with checkpointing configuration.

    Returns:
      final: A single module which contains the primals and adjoints of all the
          functions in the call tree.
      namespace: A merged dictionary with all the variables in the global
          namespaces of each function. The primals and adjoints need access to
          these in order to execute.
    """
    # Imported here to avoid circular imports
    import tangent

    namespace = {'tangent': tangent, 'numpy': numpy}

    done = set()
    final = gast.Module(body=[])
    # Unwrap the top-level function so its namespace matches the source we
    # differentiate (see unwrap_function); e.g. a cached gradient function is
    # wrapped and would otherwise contribute Tangent's globals instead of numpy.
    unwrapped_top = unwrap_function(func)
    namespace.update(unwrapped_top.__globals__)

    # Add closure variables to namespace
    if unwrapped_top.__closure__:
        namespace.update(
            dict(
                zip(
                    unwrapped_top.__code__.co_freevars,
                    (cell.cell_contents for cell in unwrapped_top.__closure__),
                )
            )
        )

    # Only the top-level function gets the seed reconciliation: the seeds of
    # the other functions in the call tree are constructed by the generated
    # adjoint code itself and already match structurally.
    node, required = autodiff_ast(
        func,
        wrt,
        motion,
        mode,
        preserve_result,
        check_dims,
        verbose,
        checkpoint_config,
        reconcile_seed,
    )
    final.body.extend(node.body)

    to_do = set(required)
    if motion == 'split' and mode == 'reverse':
        done.add((func, wrt))
        to_do -= done

    while to_do:
        func, wrt = to_do.pop()
        # Unwrap JAX JIT functions to access __globals__
        unwrapped_func = unwrap_function(func)
        namespace.update(unwrapped_func.__globals__)

        # Add closure variables to namespace
        if unwrapped_func.__closure__:
            namespace.update(
                dict(
                    zip(
                        unwrapped_func.__code__.co_freevars,
                        (cell.cell_contents for cell in unwrapped_func.__closure__),
                    )
                )
            )

        node, required = autodiff_ast(
            func=func,
            wrt=wrt,
            motion='split',
            mode=mode,
            preserve_result=True,
            check_dims=False,
            verbose=verbose,
            checkpoint_config=checkpoint_config,
        )

        final.body.extend(node.body)
        done.add((func, wrt))
        to_do.update(required)
        to_do -= done

    return final, namespace


def vjp(func, wrt=(0,), optimized=True, check_dims=True, preserve_result=False, verbose=0):
    """Convenience function to produce vector-Jacobian products.

    See `autodiff` for function arguments.
    Uses reverse-mode joint-motion autodiff to produce the VJP.
    """
    return _autodiff_uncached(
        func,
        wrt=wrt,
        motion='joint',
        mode='reverse',
        optimized=optimized,
        preserve_result=preserve_result,
        input_derivative=INPUT_DERIVATIVE.Required,
        check_dims=check_dims,
        verbose=verbose,
    )


def jvp(func, wrt=(0,), optimized=True, check_dims=True, preserve_result=False, verbose=0):
    """Convenience function to produce Jacobian-vector products.

    See `autodiff` for function arguments.
    Uses forward-mode autodiff to produce the JVP.
    """
    return _autodiff_uncached(
        func,
        wrt=wrt,
        mode='forward',
        optimized=optimized,
        preserve_result=preserve_result,
        input_derivative=INPUT_DERIVATIVE.Required,
        check_dims=check_dims,
        verbose=verbose,
    )


# Elementwise functions that the straight-line coarsening pass can emit by
# bare name (see tangent/optimizations/coarsening.py), mapped to their NumPy
# implementations so a lowered adjoint can execute in a plain namespace.
_COARSEN_ELEMENTWISE_NUMPY = {
    'sin': numpy.sin,
    'cos': numpy.cos,
    'tan': numpy.tan,
    'exp': numpy.exp,
    'log': numpy.log,
    'sqrt': numpy.sqrt,
    'abs': numpy.abs,
    'sinh': numpy.sinh,
    'cosh': numpy.cosh,
    'tanh': numpy.tanh,
    'asin': numpy.arcsin,
    'acos': numpy.arccos,
    'atan': numpy.arctan,
}

# Module prefixes that mark a non-NumPy backend. Coarsening lowers expressions
# to bare elementwise names that execute as NumPy, so it is only correct for
# primals that themselves use NumPy (or bare math) elementwise ops.
_NON_NUMPY_PREFIXES = frozenset(('jnp', 'jax', 'torch', 'tf', 'tensorflow', 'kops', 'keras'))


def _coarsening_backend_safe(func_ast):
    """True if every call in func_ast is a bare name or an np.* attribute."""
    for node in gast.walk(func_ast):
        if not isinstance(node, gast.Call):
            continue
        base = node.func
        while isinstance(base, gast.Attribute):
            base = base.value
        if isinstance(base, gast.Name) and base.id in _NON_NUMPY_PREFIXES:
            return False
    return True


def _try_coarsened_grad(func, wrt, verbose=0):
    """Build a reverse-mode gradient via straight-line coarsening, or None.

    This is an opt-in alternative code path, enabled with
    optimizations={'coarsening': True}. It only applies when the function is a
    pure straight-line segment of NumPy elementwise arithmetic; otherwise it
    returns None and the caller falls back to the standard reverse-mode
    pipeline. The generated adjoint is a single symbolic vector-Jacobian product
    rather than one adjoint statement per primitive op.
    """
    import tangent
    from tangent.optimizations.coarsening import apply_coarsening

    try:
        node = quoting.parse_function(func)
    except Exception:
        return None
    if (
        not isinstance(node, gast.Module)
        or len(node.body) != 1
        or not isinstance(node.body[0], gast.FunctionDef)
    ):
        return None
    func_ast = node.body[0]
    if not _coarsening_backend_safe(func_ast):
        return None
    adj_ast = apply_coarsening(func_ast)
    if adj_ast is None:
        return None

    # Mirror autodiff_tree's namespace, then expose the bare elementwise names
    # the lowered adjoint uses (e.g. `cos` rather than `numpy.cos`).
    unwrapped = unwrap_function(func)
    namespace = {'tangent': tangent, 'numpy': numpy}
    namespace.update(unwrapped.__globals__)
    if unwrapped.__closure__:
        namespace.update(
            dict(
                zip(
                    unwrapped.__code__.co_freevars,
                    (cell.cell_contents for cell in unwrapped.__closure__),
                )
            )
        )
    namespace.update(_COARSEN_ELEMENTWISE_NUMPY)

    if verbose >= 1:
        print('[Coarsening] Using straight-line coarsening for %s' % func.__name__)
        print(quoting.to_source(adj_ast))

    module = compile_.compile_file(gast.Module(body=[adj_ast]), namespace)
    adj = getattr(module, adj_ast.name)

    def df(*args, **kwargs):
        init_grad = kwargs.pop('init_grad', 1.0)
        grads = adj(*args, init_grad)
        if not isinstance(grads, tuple):
            grads = (grads,)
        selected = tuple(grads[i] for i in wrt)
        if len(selected) == 1:
            (selected,) = selected
        return selected

    return df


def _autodiff_uncached(
    func,
    wrt=(0,),
    optimized=True,
    motion='joint',
    mode='reverse',
    preserve_result=False,
    check_dims=True,
    input_derivative=INPUT_DERIVATIVE.Required,
    verbose=0,
    checkpoint_config=None,
    optimizations=None,
    grad_config=None,
):
    """Build the vector-Jacobian or Jacobian-vector product of a function `func`.

    For a vector-Jacobian product (reverse-mode autodiff):
    This function proceeds by finding the primals and adjoints of all the
    functions in the call tree.
    For a Jacobian-vector product (forward-mode autodiff):
    We first find the primals and tangents of all functions in the call tree.

    It then wraps the top level function (i.e. the
    one passed as `func`) in a slightly more user-friendly interface. It then
    compiles the function and attaches to it the global namespace it needs to
    run.

    Args:
      func: The function to take the gradient of.
      wrt: A tuple of argument indices to differentiate with respect to. By
          default the derivative is taken with respect to the first argument.
      optimized: Whether to optimize the gradient function (`True` by default).
      motion: Either 'split' (separate functions for forward and backward pass)
          or 'joint' motion (a single combined function). Joint mode is the
          default.
      mode: Either 'forward' or 'reverse' mode. Forward mode is more efficient
          when the input dimensionality is lower than the output dimensionality,
          whereas it is the opposite for reverse mode.
      input_derivative: An enum indicating whether the user must supply an input
          derivative, and if not, what the default value is. See the
          possible values of INPUT_DERIVATIVE in this file.

      preserve_result: A boolean indicating whether or not the generated gradient
          function should also return the output of the original function.
          If False, the return signature of the input and output functions will be
          > val = func(*args)
          > df = grad(func,preserve_result=False)
          > gradval = df(*args)
          If True,
          > val = func(*args)
          > df = grad(func,preserve_result=True)
          > gradval, val = df(*args)
          Note that if taking gradients with respect to multiple arguments,
          the primal value will be appended to the return signature. Ex:
          > val = func(x,y)
          > df = grad(func,wrt=(0,1),preserve_result=True)
          > dx,dy,val = df(x,y)

      verbose: If 1 the source code of the generated functions will be
          output to stdout at various stages of the process for debugging
          purposes. If > 1, all intermediate code generation steps will print.
      checkpoint_config: Optional dictionary with checkpointing configuration.
          Keys: 'enabled' (bool), 'min_length' (int), 'num_checkpoints' (int or None)

    Returns:
      df: A function that calculates a derivative (see file-level documentation
      above
          for the kinds of derivatives available) with respect to arguments
          specified in `wrt`, using forward or reverse mode according to `mode`.
          If using reverse mode, the gradient is calculated in either split
          or joint motion according to the value passed in `motion`. If
          `preserve_result` is True, the function will also return the original
          result of `func`.
    """
    # If the function had the with insert_grad_of statements removed, retrieve them
    func = getattr(func, 'tangent', func)

    # Opt-in straight-line coarsening: when requested, and the function is a
    # pure NumPy straight-line segment, emit a single symbolic VJP instead of
    # running the standard reverse-mode pipeline. Falls back otherwise.
    if (
        mode == 'reverse'
        and not preserve_result
        and optimizations
        and optimizations.get('coarsening', False)
        and not (
            grad_config
            and (
                grad_config.get('output_index') is not None
                or grad_config.get('output_weights') is not None
            )
        )
    ):
        coarsened = _try_coarsened_grad(func, wrt, verbose)
        if coarsened is not None:
            return coarsened

    # Generate the derivative. User-facing gradients with a default seed
    # (`grad`) get a runtime seed reconciliation so functions returning
    # containers (pytrees) receive a structurally matching seed; `vjp`-style
    # calls (input_derivative Required) keep their exact caller-supplied seed.
    reconcile_seed = (
        mode == 'reverse' and motion == 'joint' and input_derivative == INPUT_DERIVATIVE.DefaultOne
    )
    node, namespace = autodiff_tree(
        func,
        wrt,
        motion,
        mode,
        preserve_result,
        check_dims,
        verbose,
        checkpoint_config,
        reconcile_seed,
    )

    if mode == 'reverse' and motion == 'joint':
        # Pull the stack definition and initial gradient into the function body
        # TODO: Use first FunctionDef instead of first element
        node.body[0] = _create_joint(node.body[0], func, wrt, input_derivative, grad_config)
        if verbose >= 2:
            print('INLINED')
            print(quoting.to_source(node))
    if mode == 'forward':
        node = _create_forward(node)

    # Apply optimizations
    if optimized:
        # Determine which optimizations to use
        if optimizations is None:
            optimizations = {}

        use_advanced_dce = optimizations.get('dce', True) and mode == 'reverse'
        use_strength_reduction = optimizations.get(
            'strength_reduction', False
        )  # Disabled by default
        use_cse = optimizations.get('cse', False)  # CSE disabled by default for now
        use_algebraic = optimizations.get(
            'algebraic', False
        )  # Algebraic disabled by default for now

        # Determine which pipeline to use
        if use_strength_reduction or use_cse or use_algebraic:
            # Use symbolic optimization pipeline (includes strength reduction, CSE, algebraic, and DCE)
            # Extract parameter names from wrt indices for advanced DCE
            import inspect

            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            requested_grads = (
                [param_names[i] for i in wrt if i < len(param_names)] if use_advanced_dce else None
            )

            if verbose >= 1:
                enabled = []
                if use_strength_reduction:
                    enabled.append('Strength')
                if use_cse:
                    enabled.append('CSE')
                if use_algebraic:
                    enabled.append('Algebraic')
                if use_advanced_dce:
                    enabled.append('DCE')
                print(f"[Optimization] Using symbolic pipeline with {', '.join(enabled)}")

            node = optimization.optimize_with_symbolic(
                node,
                requested_grads=requested_grads,
                enable_strength_reduction=use_strength_reduction,
                enable_cse=use_cse,
                enable_algebraic=use_algebraic,
                verbose=verbose,
            )
        elif use_advanced_dce:
            # Use unified optimization pipeline with advanced DCE only
            # Extract parameter names from wrt indices for advanced DCE
            import inspect

            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            requested_grads = [param_names[i] for i in wrt if i < len(param_names)]

            # Use unified optimization pipeline with advanced DCE
            if verbose >= 1:
                print(
                    f"[Optimization] Using unified pipeline with advanced DCE for {requested_grads}"
                )
            node = optimization.optimize_with_advanced_dce(node, requested_grads, verbose)
        else:
            # Use standard optimizations only
            node = optimization.optimize(node)

    node = comments.remove_repeated_comments(node)
    if verbose >= 1:
        print(quoting.to_source(node))

    # Compile and return
    module = compile_.compile_file(node, namespace)
    if mode == 'forward' or motion == 'joint':
        return getattr(module, node.body[0].name)
    else:
        # Compiling the top-level function in split mode makes no sense, but we use
        # it for testing; hence we don't care about the source being readable
        forward = getattr(module, node.body[0].name)
        backward = getattr(module, node.body[1].name)

        # Imported here to avoid circular imports
        import tangent

        def df(*args, **kwargs):
            _stack = tangent.Stack()
            init_grad = kwargs.pop('init_grad', 1.0)
            forward(_stack, *args, **kwargs)
            dx = backward(_stack, init_grad, *args, **kwargs)
            if len(dx) == 1:
                (dx,) = dx
            return dx

        return df


def _grad_uncached(
    func,
    wrt=(0,),
    optimized=True,
    preserve_result=False,
    check_dims=True,
    verbose=0,
    checkpoint=False,
    checkpoint_config=None,
    optimizations=None,
    output_index=None,
    output_weights=None,
):
    """Return the gradient of a function `func`.
    Args:
      func: The function to take the gradient of.
      wrt: A tuple of argument indices to differentiate with respect to. By
          default the derivative is taken with respect to the first argument.
      optimized: Whether to optimize the gradient function (`True` by default).
      preserve_result: A boolean indicating whether or not the generated gradient
          function should also return the output of the original function.
          If False, the return signature of the input and output functions will be
          > val = func(*args)
          > df = grad(func,preserve_result=False)
          > gradval = df(*args)
          If True,
          > val = func(*args)
          > df = grad(func,preserve_result=True)
          > gradval, val = df(*args)
          Note that if taking gradients with respect to multiple arguments,
          the primal value will be appended to the return signature. Ex:
          > val = func(x,y)
          > df = grad(func,wrt=(0,1),preserve_result=True)
          > dx,dy,val = df(x,y)
      check_dims: A boolean (`True` by default) indicating whether to check
          that the result of the original function `func` is a scalar, raising
          an error if it is not.
          Gradients are only valid for scalar-valued outputs, so we check
          this by defualt.
      verbose: If 1 the source code of the generated functions will be
          output to stdout at various stages of the process for debugging
          purposes. If > 1, all intermediate code generation steps will print.
      checkpoint: Enable automatic checkpointing for loops (default: False).
          Limited: applies only to `for i in range(n)` loops with a constant,
          zero-based range of at least 'min_length' iterations, and only the
          loop target variable is stored selectively (at ~sqrt(n) checkpoint
          positions) - other loop-body intermediates are still taped every
          iteration, so the measured overall memory reduction is small (~3%).
          See docs/checkpointing_user_guide.md.
      checkpoint_config: Dictionary with checkpointing configuration:
          - 'enabled': Enable checkpointing (default: value of checkpoint param)
          - 'min_length': Minimum loop length to checkpoint (default: 100)
          - 'num_checkpoints': Number of checkpoints or None for auto (default: None)
      optimizations: Dictionary with optimization flags (default: {'dce': True}):
          - 'dce': Enable dead code elimination (default: True)
      output_index: Integer or None. For multi-output functions (returning tuples),
          specifies which output to differentiate. If None (default), differentiates
          the sum of all outputs. Example:
          > def f(x): return x**2, x*3
          > df = grad(f, output_index=0)  # Gradient of first output only
          > grad = df(2.0)  # = 4.0 (gradient of x**2)
      output_weights: Tuple of floats or None. For multi-output functions, specifies
          weights for each output. If None (default), all weights are 1.0 (sum).
          Example:
          > df = grad(f, output_weights=(0.7, 0.3))  # 0.7*out1 + 0.3*out2
          Cannot be used together with output_index.

    Returns:
      df: A function that calculates the gradient with respect to arguments
          specified in `wrt`, using forward or reverse mode according to `mode`.
          If using reverse mode, the gradient is calculated in either split
          or joint motion according to the value passed in `motion`. If
          `preserve_result` is True, the function will also return the original
          result of `func`.
    """
    # Validate output_index and output_weights
    if output_index is not None and output_weights is not None:
        raise ValueError("Cannot specify both output_index and output_weights")

    # Store these for use in _create_joint
    grad_config = {'output_index': output_index, 'output_weights': output_weights}
    # Prepare checkpoint configuration
    if checkpoint_config is None:
        checkpoint_config = {}
    if checkpoint is True:
        checkpoint_config.setdefault('enabled', True)

    # Checkpointing and optimization coexist: dead code elimination pairs tape
    # pushes with the pops that consume them (see optimization._tape_pairings),
    # so the snapshot/iterable/count pushes of the segment-checkpointed loop
    # are either kept or removed together and the stack stays balanced.

    return _autodiff_uncached(
        func,
        wrt=wrt,
        motion='joint',
        mode='reverse',
        optimized=optimized,
        preserve_result=preserve_result,
        check_dims=check_dims,
        input_derivative=INPUT_DERIVATIVE.DefaultOne,
        verbose=verbose,
        checkpoint_config=checkpoint_config,
        optimizations=optimizations,
        grad_config=grad_config,
    )


# TODO: these are utility functions, designed only for internal use.
# Should be moved to a separate file.
def _create_joint(fwdbwd, func, wrt, input_derivative, grad_config=None):
    """Create a user-friendly gradient function.

    By default, gradient functions expect the stack to be passed to them
    explicitly. This function modifies the function so that the stack doesn't
    need to be passed and gets initialized in the function body instead.

    For consistency, gradient functions always return a tuple, even if the
    gradient of only one input was required. We unpack the tuple if it is of
    length one.

    Args:
      fwdbwd: An AST. The function definition of the joint primal and adjoint.
      func: A function handle. The original function that was differentiated.
      wrt: A tuple of integers. The arguments with respect to which we differentiated.
      grad_config: Optional dict with 'output_index' and 'output_weights' for multi-output functions.

    Returns:
      The function definition of the new function.
    """
    # Default grad_config if not provided
    if grad_config is None:
        grad_config = {'output_index': None, 'output_weights': None}
    # Correct return to be a non-tuple if there's only one element
    retval = fwdbwd.body[-1]
    if len(retval.value.elts) == 1:
        retval.value = retval.value.elts[0]

    # Make a stack init statement
    init_stack = quoting.quote('%s = tangent.Stack()' % fwdbwd.args.args[0].id)
    init_stack = comments.add_comment(init_stack, 'Initialize the tape')

    # Prepend the stack init to the top of the function
    fwdbwd.body = [init_stack] + fwdbwd.body

    # Replace the function arguments with the original ones
    grad_name = fwdbwd.args.args[1].id
    fwdbwd.args = quoting.parse_function(func).body[0].args

    # Give the function a nice name
    fwdbwd.name = naming.joint_name(func, wrt)

    # Allow the initial gradient to be passed as a keyword argument
    fwdbwd = ast_.append_args(fwdbwd, [grad_name])
    if input_derivative == INPUT_DERIVATIVE.DefaultOne:
        # The output arity is recorded during reverse-mode transformation (see
        # reverse_ad._output_arity) and threaded through the motion pass. It is
        # the source of truth for the shape of the default gradient seed.
        if anno.hasanno(fwdbwd, 'output_arity'):
            output_arity = anno.getanno(fwdbwd, 'output_arity')
            returns_tuple = output_arity is not None
            tuple_size = output_arity if returns_tuple else 0
        else:
            # Fallback for ASTs built without the annotation (e.g. handed to this
            # helper directly): infer the arity from the generated primal. This is
            # fragile - it relies on the check_dims shapes_match assert being
            # present - which is why the annotation above is preferred.
            returns_tuple, tuple_size = _infer_output_arity_from_primal(fwdbwd)

        # Set appropriate default based on return type and grad_config
        if returns_tuple and tuple_size > 0:
            # Multi-output function: determine gradient seed based on configuration
            output_index = grad_config.get('output_index')
            output_weights = grad_config.get('output_weights')

            if output_index is not None:
                # Gradient of specific output only
                if not (0 <= output_index < tuple_size):
                    raise ValueError(
                        f"output_index={output_index} is out of range for function "
                        f"'{getattr(func, '__name__', '<function>')}' which returns {tuple_size} values"
                    )
                # Create one-hot seed: (0, 0, ..., 1.0, ..., 0, 0)
                seed_values = ['0.0'] * tuple_size
                seed_values[output_index] = '1.0'
                default_str = '(' + ', '.join(seed_values) + ')'
                fwdbwd.args.defaults.append(quoting.quote(default_str))

            elif output_weights is not None:
                # Custom weighted combination
                if len(output_weights) != tuple_size:
                    raise ValueError(
                        f"output_weights has {len(output_weights)} values but function "
                        f"'{getattr(func, '__name__', '<function>')}' returns {tuple_size} values"
                    )
                # Use provided weights
                default_str = '(' + ', '.join([str(float(w)) for w in output_weights]) + ')'
                fwdbwd.args.defaults.append(quoting.quote(default_str))

            else:
                # Default: sum all outputs (backward compatible)
                default_str = '(' + ', '.join(['1.0'] * tuple_size) + ')'
                fwdbwd.args.defaults.append(quoting.quote(default_str))

                # Add warning only for default auto-sum behavior
                import warnings

                func_name = getattr(func, '__name__', '<function>')
                warnings.warn(
                    f"\nFunction '{func_name}' returns a tuple of {tuple_size} values. "
                    f"The gradient will compute d/dx(sum of all outputs) using seed {default_str}.\n"
                    f"This is mathematically correct for multi-output functions where you want "
                    f"the gradient of the sum.\n"
                    f"If you need individual gradients, use output_index parameter:\n"
                    f"  df = tangent.grad(f, output_index=0)  # Gradient of first output\n"
                    f"Or use output_weights for custom weighting:\n"
                    f"  df = tangent.grad(f, output_weights=(0.7, 0.3))  # Weighted combination\n"
                    f"See test_multi_output_grad.py for examples.",
                    UserWarning,
                    stacklevel=4,
                )
        else:
            # Single (non-tuple-literal) return: scalar, array or container
            # (dict/list/nested pytree). The scalar default works directly for
            # scalar and array outputs; for container outputs it is expanded into a
            # pytree of ones at runtime by the seed reconciliation that
            # `reverse_ad` emits at the top of the adjoint (see `reconcile_seed`).
            fwdbwd.args.defaults.append(quoting.quote('1.0'))
    return fwdbwd


def _infer_output_arity_from_primal(fwdbwd):
    """Fallback inference of the output arity from generated primal code.

    Looks for an assignment of a tuple whose target is checked by the
    check_dims shapes_match assert. Only used when the 'output_arity'
    annotation recorded by reverse-mode AD is missing; it silently fails to
    detect tuple returns when the assert is absent (check_dims=False) or was
    optimized away.

    Args:
      fwdbwd: The joint primal-and-adjoint function definition AST.

    Returns:
      A (returns_tuple, tuple_size) pair.
    """
    for stmt in fwdbwd.body:
        if isinstance(stmt, gast.Assign):
            # Check if this assigns a tuple (e.g., "t = a, b")
            if (
                isinstance(stmt.value, gast.Tuple)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], gast.Name)
            ):
                # This might be the return value - check if it's used in shapes_match
                var_name = stmt.targets[0].id
                for check_stmt in fwdbwd.body:
                    if isinstance(check_stmt, gast.Assert):
                        if (
                            isinstance(check_stmt.test, gast.Call)
                            and hasattr(check_stmt.test.func, 'attr')
                            and check_stmt.test.func.attr == 'shapes_match'
                            and len(check_stmt.test.args) >= 2
                            and isinstance(check_stmt.test.args[0], gast.Name)
                            and check_stmt.test.args[0].id == var_name
                        ):
                            # This is the return value and it's a tuple
                            return True, len(stmt.value.elts)
    return False, 0


def _create_forward(out_node):
    """Create a user-friendly forward function.

    Ensures that a single value instead of a tuple is returned if the user asked
    for the gradient with respect to only one input.

    Args:
      out_node: The function definition AST.

    Returns:
      The function definition with potentially changed return statement.
    """
    retval = out_node.body[0].body[-1]
    if len(retval.value.elts) == 1:
        retval.value = retval.value.elts[0]
    return out_node


# Apply caching decorators to create the public API functions
autodiff = cached_autodiff(_autodiff_uncached)
grad = cached_grad(_grad_uncached)
