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
from tangent.utils import pop
from tangent.utils import pop_stack
from tangent.utils import push
from tangent.utils import push_stack
from tangent.utils import shapes_match
from tangent.utils import Stack
from tangent.utils import unbroadcast
from tangent.utils import unreduce
from tangent.utils import unreduce_like

# NumPy activation functions for neural networks
from tangent.grads import (
    numpy_relu,
    numpy_sigmoid,
    numpy_tanh,
    numpy_leaky_relu,
    numpy_elu,
    numpy_softplus
)

# Checkpointing for memory-efficient gradient computation
from tangent.checkpointing_simple import (
    compute_checkpoint_positions,
    checkpointed_loop,
    get_memory_savings
)
from tangent.grad_checkpoint import (
    grad_with_checkpointing,
    estimate_checkpoint_savings,
    should_checkpoint
)
from tangent.checkpoint_helpers import (
    compute_optimal_checkpoints,
    find_nearest_checkpoint,
    is_checkpoint_iteration,
    store_checkpoint,
    restore_checkpoint,
    estimate_memory_savings as estimate_memory_savings_helper,
    get_checkpoint_info,
    CheckpointAwareStack
)

# Imported last to avoid circular imports
from tangent.grad_util import grad, autodiff, vjp, jvp
from tangent.errors import *
from tangent.function_cache import (clear_cache, get_cache_stats,
                                     reset_cache_stats, set_cache_size,
                                     get_cache_size)
try:
  from tangent.tf_extensions import *
except (ImportError, AttributeError) as e:
  # TensorFlow extensions are optional and may not work with TensorFlow 2.x
  # Tangent was designed for TensorFlow 1.x
  import warnings
  warnings.warn(f"TensorFlow extensions not available: {e}. Core autodiff functionality still works.")
  pass

# JAX extensions (optional)
try:
  from tangent.jax_extensions import *
except (ImportError, AttributeError) as e:
  # JAX is optional
  import warnings
  warnings.warn(f"JAX extensions not available: {e}. Install JAX with: pip install jax jaxlib")
  pass

# Extended NumPy gradients
try:
  from tangent import numpy_extended
except (ImportError, AttributeError) as e:
  import warnings
  warnings.warn(f"Extended NumPy gradients not available: {e}")
  pass

# Extended TensorFlow gradients
try:
  from tangent import tf_extended
except (ImportError, AttributeError) as e:
  import warnings
  warnings.warn(f"Extended TensorFlow gradients not available: {e}")
  pass

# Visualization tools (optional)
try:
  from tangent.visualization import visualize, compare_gradients, show_gradient_code
except ImportError as e:
  # Visualization requires matplotlib and networkx
  import warnings
  warnings.warn(f"Visualization tools not available: {e}. Install with: pip install matplotlib networkx")
  pass


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
