# Checkpointing User Guide

## Overview

Checkpointing is a memory-efficient technique for computing gradients through
long sequences. Instead of storing all intermediate states during the forward
pass (O(n) memory), checkpointing stores only a small number of "checkpoints"
(O(√n) memory) and recomputes intermediate values during the backward pass as
needed.

## What works today (and what doesn't)

**Works:**

1. **`tangent.grad(func, checkpoint=True)`** — segment (√n) checkpointing of
   counted loops inside generated gradients. Eligible loops run *untaped*:
   the forward sweep pushes only a snapshot of the loop-carried state (the
   variables assigned in the body that are defined at loop entry) every
   ceil(√n) iterations. The backward pass restores each snapshot in reverse
   order, replays just that segment with taping, and immediately consumes the
   segment's tape. This is the O(√n)-total-tape algorithm from the
   checkpointing literature, and the replay is exact recomputation — the
   gradients are **identical** to the fully-taped ones, not approximations.

   **Measured: 49.2 MB → 1.6 MB peak (96.8% reduction)** on a 2000-iteration
   loop carrying a 1000-float state (`benchmarks/checkpointing_memory.py`).

   Eligibility:
   - `for target in range(...)` loops with constant integer bounds — any
     start/stop/step form — of at least 100 iterations (configurable via
     `checkpoint_config={'min_length': ...}`). Loops that don't match fall
     back to standard full-tape differentiation, silently and safely.
   - First-order reverse mode. Taking a higher-order derivative *of a
     checkpointed gradient function* is not supported; use plain
     `tangent.grad` for derivatives you intend to differentiate again.
   - Composes with the optimizer (`optimized=True`, the default) and with
     nested loops (an eligible outer loop checkpoints; inner loops are taped
     within each replayed segment).
2. **`tangent.checkpointed_loop`** — a *manual, forward-pass-only* helper
   that runs `state = step(state)` for `seq_length` iterations while storing
   only O(√n) snapshots. Useful for memory-bounded forward simulations.
   **`tangent.grad` cannot differentiate through it** (it is an opaque
   higher-order call).
3. The bookkeeping utilities: `compute_checkpoint_positions`,
   `get_memory_savings`, `estimate_checkpoint_savings`, `should_checkpoint`,
   `compute_optimal_checkpoints` (used by `checkpointed_loop`).

**Does not work:**

- **`tangent.grad_with_checkpointing`** raises `NotImplementedError` for any
  function containing a loop; use `tangent.grad(f, checkpoint=True)` instead.
  For loop-free functions it simply delegates to `tangent.grad`.
- Checkpointing of `while` loops, non-`range` iterables, or loops whose
  length is not a compile-time constant (they fall back to full taping).


## Quick start: automatic checkpointing in `grad`

```python
import numpy as np
import tangent

def f(x):
    for i in range(1000):        # constant, zero-based range >= min_length
        x = np.tanh(x * 1.01)
    return x

df = tangent.grad(f, checkpoint=True)              # optimization on by default
df_opt = tangent.grad(f, checkpoint=True, optimized=True)  # explicit; also fine

# Tune the eligibility threshold:
df2 = tangent.grad(f, checkpoint_config={'enabled': True, 'min_length': 500})
```

The generated gradient runs the loop untaped and snapshots the loop-carried
state at √n segment boundaries; the backward pass replays one segment at a
time. Peak tape memory is one segment plus the snapshots — O(√n) instead of
O(n).

## Manual forward-pass checkpointing

### Basic usage

```python
import numpy as np
import tangent

def rnn_step(state):
    return np.tanh(state * 1.1 + 0.1)

x0 = np.zeros(512)

# Stores only 31 snapshots instead of 1000 states
final_state, checkpoints = tangent.checkpointed_loop(
    rnn_step,
    x0,
    seq_length=1000,
    num_checkpoints=31,  # or None for automatic sqrt(n)
)
```

`checkpoints` is a dict mapping iteration index → saved state. If you need
gradients, you must build the backward pass yourself (e.g., differentiate
`rnn_step` with `tangent.grad(rnn_step)` and drive the
recompute-from-checkpoint loop in your own code). Wrapping
`checkpointed_loop` inside a function passed to `tangent.grad` **does not
work** — Tangent will fail to transform the call (and nested `def`s are
rejected outright).

### LSTM-style tuple state

```python
def lstm_step(state):
    h, c = state
    # ... compute h_new, c_new ...
    return (h_new, c_new)

final_state, checkpoints = tangent.checkpointed_loop(
    lstm_step, (h0, c0), seq_length=1000, num_checkpoints=31)
```

Tuple, list, dict, and nested states are deep-copied per checkpoint.

## API Reference

### `checkpointed_loop(func, initial_state, seq_length, num_checkpoints=None)`

Execute a loop with checkpointing (forward pass only).

**Arguments:**
- `func` (Callable): Function to apply at each step (state → new_state)
- `initial_state` (array): Starting state for the sequence
- `seq_length` (int): Number of iterations
- `num_checkpoints` (int, optional): Number of checkpoints (default: √n)

**Returns:**
- `final_state` (array): Result after all iterations
- `checkpoints` (dict): Dictionary mapping positions to saved states

### `compute_checkpoint_positions(seq_length, num_checkpoints)`

Compute checkpoint positions (approximately evenly spaced).

```python
positions = tangent.compute_checkpoint_positions(1000, 31)
```

### `get_memory_savings(seq_length, num_checkpoints=None)`

Calculate expected *state-storage* savings of `checkpointed_loop` relative to
storing every state. Keys: `'without_checkpointing'`,
`'with_checkpointing'`, `'savings_percent'`, `'savings_ratio'`,
`'num_checkpoints'`, `'recomputation_factor'`.

```python
stats = tangent.get_memory_savings(1000)
print(f"State-storage reduction: {stats['savings_percent']:.1f}%")  # ~96.9%
```

This figure describes `checkpointed_loop`'s state storage. The tape memory
of `grad(..., checkpoint=True)` shows a comparable reduction — see
`benchmarks/checkpointing_memory.py` (96.8% measured).

### `should_checkpoint(seq_length, threshold=0.5)`

True if the estimated state-storage savings ratio of `checkpointed_loop`
exceeds `threshold`.

### `grad_with_checkpointing(func, wrt=(0,), num_checkpoints=None, **grad_kwargs)`

**Not implemented for functions with loops** — raises `NotImplementedError`
immediately (at wrapper-creation time) with pointers to the working
alternatives. Delegates to `tangent.grad` for loop-free functions.

## Performance considerations (manual helper)

- **Memory**: state storage reduced from O(n) to O(√n)
- **Recomputation**: a backward pass that recomputes from √n checkpoints
  performs ~O(n√n) extra forward steps in the naive schedule;
  `stats['recomputation_factor']` from `get_memory_savings` reports the
  average per-step recomputation for the stored schedule
- √n checkpoints is the classic memory/recompute balance point; pass a
  larger `num_checkpoints` to trade memory for less recomputation

## Troubleshooting

### "Results don't match"

Ensure your step function is deterministic and doesn't depend on external
mutable state:

```python
# Bad: depends on external loop counter
counter = 0
def step(state):
    global counter
    counter += 1
    return state * counter  # Different on recomputation!

# Good: pure function
def step(state):
    return state * 1.1
```

### `checkpoint=True` seems to change nothing

Check the loop's eligibility: it must be `for i in range(n)` with a constant
literal `n >= min_length` (default 100) and a zero-based range. Ineligible
loops silently use the standard full-tape path (this is deliberate — it is
the correct fallback).

## Examples

- `examples/checkpoint_demo.py` — demonstration of the manual helpers
- `tests/test_checkpointing_basic.py` — unit tests for the manual helpers
- `tests/test_tape_pairing.py::test_checkpointing_with_optimization` —
  `grad(checkpoint=True, optimized=True)` correctness test

## References

1. **Griewank, A., & Walther, A. (2000)**. Algorithm 799: Revolve: An
   implementation of checkpointing for the reverse or adjoint mode of
   computational differentiation. *ACM TOMS*, 26(1), 19–45.
2. **Gruslys, A., et al. (2016)**. Memory-Efficient Backpropagation Through
   Time. *NeurIPS 29*.

The segment implementation lives in `tangent/grads.py`
(`for_checkpointed`/`dfor_checkpointed`) and `tangent/reverse_ad.py`
(`visit_For`, `_loop_state_names`); `tests/test_checkpointing_segments.py`
pins both exactness and the memory reduction.
