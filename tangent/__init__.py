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
"""Several imports to flatten the Tangent namespace for end users."""

from __future__ import absolute_import
import functools

# Single source of truth for the package version; pyproject.toml reads this
# statically via [tool.setuptools.dynamic]. PyPI's `tangent` 0.1.9 is Google's
# unmaintained 2017 release — this fork's version must stay ahead of it.
__version__ = '0.2.0'

import gast

from tangent import annotate
from tangent import ast as ast_
from tangent import compile as compile_
from tangent.tracing import trace
from tangent.tracing import trace_grad
from tangent.utils import add_grad
from tangent.utils import array_size
from tangent.utils import astype
from tangent.utils import balanced_eq
from tangent.utils import copy
from tangent.utils import grad_dot
from tangent.utils import init_grad
from tangent.utils import insert_grad_of
from tangent.utils import matmul_grad_x
from tangent.utils import matmul_grad_y
from tangent.utils import pop
from tangent.utils import transpose_inverse_axes
from tangent.utils import pop_stack
from tangent.utils import get_shape
from tangent.utils import push
from tangent.utils import push_stack
from tangent.utils import shapes_match
from tangent.utils import match_seed
from tangent.utils import match_seed_grad
from tangent.utils import seed_pytree
from tangent.utils import Stack
from tangent.utils import unbroadcast
from tangent.utils import unreduce
from tangent.utils import unreduce_like
from tangent.utils import unrepeat
from tangent.utils import untile
from tangent.utils import update_grad_at_index
from tangent.utils import add_grad_at_index
from tangent.utils import list_append
from tangent.utils import list_init
from tangent.utils import list_last
from tangent.utils import num_segments
from tangent.utils import segment_bounds
from tangent.utils import segment_size
from tangent.utils import snapshot
from tangent.utils import cholesky_grad
from tangent.utils import eigvalsh_grad
from tangent.utils import einsum_grad
from tangent.utils import sort_like
from tangent.utils import uncumprod
from tangent.utils import unpad
from tangent.utils import unsort
from tangent.utils import untake

# NumPy activation functions for neural networks
from tangent.grads import (
    numpy_relu,
    numpy_sigmoid,
    numpy_tanh,
    numpy_leaky_relu,
    numpy_elu,
    numpy_softplus,
)

# Checkpointing for memory-efficient gradient computation
from tangent.checkpointing_simple import (
    compute_checkpoint_positions,
    checkpointed_loop,
    get_memory_savings,
)
from tangent.grad_checkpoint import (
    grad_with_checkpointing,
    estimate_checkpoint_savings,
    should_checkpoint,
)
from tangent.checkpoint_helpers import (
    compute_optimal_checkpoints,
    find_nearest_checkpoint,
    is_checkpoint_iteration,
    store_checkpoint,
    restore_checkpoint,
    estimate_memory_savings as estimate_memory_savings_helper,
    get_checkpoint_info,
    CheckpointAwareStack,
)

# Imported last to avoid circular imports
from tangent.grad_util import grad, autodiff, vjp, jvp
from tangent.errors import *
from tangent.function_cache import (
    clear_cache,
    get_cache_stats,
    reset_cache_stats,
    set_cache_size,
    get_cache_size,
)

# Optional backend extensions. A missing optional dependency is a normal,
# silent condition (logged at DEBUG level on the 'tangent' logger); a backend
# that is installed but fails to load is broken and warrants a real warning.
import importlib.util as _importlib_util
import logging as _logging
import warnings as _warnings

_logger = _logging.getLogger('tangent')

# Maps backend name -> 'available', 'not installed', or 'broken: <error>'.
_backend_status = {'numpy': 'available'}


def backend_status():
    """Return the load status of Tangent's optional backend extensions.

    Returns:
      A dict mapping backend names (e.g. 'jax', 'torch') to one of
      'available', 'not installed', or 'broken: <error message>'.

    Missing optional backends are logged at DEBUG level at import time; to see
    those messages, enable debug logging before importing tangent:

        logging.getLogger('tangent').setLevel(logging.DEBUG)
    """
    return dict(_backend_status)


def _optional_backend_failed(backend, error, requires, install_hint):
    """Record and report a failed optional-extension import.

    Silently logs at DEBUG level when the underlying dependency is simply not
    installed; emits a real warning when the dependency is present but the
    extension failed to load (broken/incompatible installation).
    """
    missing = []
    for dep in requires:
        try:
            if _importlib_util.find_spec(dep) is None:
                missing.append(dep)
        except (ImportError, ValueError):
            missing.append(dep)
    if missing:
        _backend_status[backend] = 'not installed'
        _logger.debug(
            '%s extensions not loaded (%s not installed). Install with: pip install %s',
            backend,
            ', '.join(missing),
            install_hint,
        )
    else:
        _backend_status[backend] = f'broken: {error}'
        _warnings.warn(
            f'{backend} is installed but its Tangent extensions failed to load: '
            f'{error}. Core autodiff functionality still works.'
        )


try:
    from tangent.tf_extensions import *

    _backend_status['tensorflow'] = 'available'
except (ImportError, AttributeError) as e:
    _optional_backend_failed('tensorflow', e, ['tensorflow'], 'tensorflow')

# JAX extensions (optional)
try:
    from tangent.jax_extensions import *

    _backend_status['jax'] = 'available'
except (ImportError, AttributeError) as e:
    _optional_backend_failed('jax', e, ['jax'], 'jax jaxlib')

# PyTorch extensions (optional)
try:
    from tangent.torch_extensions import *

    _backend_status['torch'] = 'available'
except (ImportError, AttributeError) as e:
    _optional_backend_failed('torch', e, ['torch'], 'torch')

# Keras extensions (optional; work with any Keras 3 backend)
try:
    from tangent.keras_extensions import *

    _backend_status['keras'] = 'available'
except (ImportError, AttributeError) as e:
    _optional_backend_failed('keras', e, ['keras'], 'keras')

# tinygrad extensions (optional; tinygrad's method-based tensor API)
try:
    from tangent.tinygrad_extensions import *

    _backend_status['tinygrad'] = 'available'
except (ImportError, AttributeError) as e:
    _optional_backend_failed('tinygrad', e, ['tinygrad'], 'tinygrad')

# Extended NumPy gradients (only requires numpy, so a failure here is a bug)
try:
    from tangent import numpy_extended

    # Varargs concat/stack helpers referenced by generated code as tangent.<name>
    from tangent.numpy_extended import np_concat_seq, np_stack_seq, np_concat_grads, np_stack_grads
except (ImportError, AttributeError) as e:
    _warnings.warn(f'Extended NumPy gradients not available: {e}')

# Extended TensorFlow gradients
try:
    from tangent import tf_extended

    # Varargs concat/stack helpers referenced by generated code as tangent.<name>
    from tangent.tf_extended import tf_concat_seq, tf_stack_seq, tf_concat_grads, tf_stack_grads
except (ImportError, AttributeError) as e:
    if _backend_status.get('tensorflow') == 'available':
        _warnings.warn(f'Extended TensorFlow gradients not available: {e}')
    else:
        _logger.debug('Extended TensorFlow gradients not loaded: %s', e)

# Visualization tools (optional; require matplotlib and networkx)
try:
    from tangent import visualization as _visualization
    from tangent.visualization import visualize, compare_gradients, show_gradient_code

    if _visualization.MATPLOTLIB_AVAILABLE and _visualization.NETWORKX_AVAILABLE:
        _backend_status['visualization'] = 'available'
    else:
        _backend_status['visualization'] = 'not installed'
        _logger.debug(
            'Visualization tools not fully available. Install with: pip install matplotlib networkx'
        )
except (ImportError, AttributeError) as e:
    _optional_backend_failed('visualization', e, ['matplotlib', 'networkx'], 'matplotlib networkx')


class RemoveWith(gast.NodeTransformer):
    """A transformer that removes `with insert_grad_of` statements."""

    def visit_With(self, node):
        if ast_.is_insert_grad_of_statement(node):
            return None
        else:
            return node


def tangent(f):
    """A decorator which removes the `with insert_grad_of` statement.

    This allows the function to be called as usual.

    Args:
      f: A function

    Returns:
      A function with any `with insert_grad_of` context managers removed.
    """
    node = annotate.resolve_calls(f)
    RemoveWith().visit(node)
    wrapped = functools.wraps(f)(compile_.compile_function(node))
    wrapped.tangent = f
    return wrapped
