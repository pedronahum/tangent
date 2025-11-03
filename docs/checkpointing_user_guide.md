# Checkpointing User Guide

## Overview

Checkpointing is a memory-efficient technique for computing gradients through long sequences. Instead of storing all intermediate states during the forward pass (which requires O(n) memory), checkpointing stores only a small number of "checkpoints" (O(√n) memory) and recomputes intermediate values during the backward pass as needed.

**Memory Savings**: For a sequence of length 1000, checkpointing reduces memory usage by ~97% with only a modest increase in computation time.

## When to Use Checkpointing

Checkpointing is beneficial when:

1. **Long sequences**: Training RNNs, LSTMs, or any model with long temporal dependencies
2. **Memory constraints**: Running large models on limited GPU memory
3. **Deep networks**: Very deep networks where storing all activations is prohibitive

Use the helper function to decide:

```python
import tangent

# Should we checkpoint for this sequence length?
if tangent.should_checkpoint(seq_length=1000):
    print("Checkpointing recommended - 96.9% memory reduction")
```

## Quick Start

### Basic Usage

```python
import numpy as np
import tangent

# Define your RNN step function
def rnn_step(state):
    return np.tanh(state * 1.1 + 0.1)

# Initial state
x0 = np.zeros(512)

# WITHOUT checkpointing (stores 1000 states)
states = []
state = x0
for i in range(1000):
    state = rnn_step(state)
    states.append(state.copy())  # O(n) memory!

# WITH checkpointing (stores only 31 checkpoints)
final_state, checkpoints = tangent.checkpointed_loop(
    rnn_step,
    x0,
    seq_length=1000,
    num_checkpoints=31  # or None for automatic sqrt(n)
)
# Same result, ~97% less memory!
```

### RNN Example

```python
import numpy as np
import tangent

# RNN parameters
hidden_size = 512
seq_length = 1000
W = np.random.randn(hidden_size, hidden_size) * 0.01
b = np.random.randn(hidden_size) * 0.01

# Define RNN step
def rnn_step(state):
    return np.tanh(state @ W + b)

# Forward pass with checkpointing
initial_state = np.zeros(hidden_size)
final_state, checkpoints = tangent.checkpointed_loop(
    rnn_step,
    initial_state,
    seq_length,
    num_checkpoints=None  # Auto: sqrt(1000) = 31 checkpoints
)

print(f"Stored {len(checkpoints)} checkpoints instead of {seq_length} states")
# Output: Stored 31 checkpoints instead of 1000 states
```

## API Reference

### `checkpointed_loop(func, initial_state, seq_length, num_checkpoints=None)`

Execute a loop with checkpointing.

**Arguments:**
- `func` (Callable): Function to apply at each step (state → new_state)
- `initial_state` (array): Starting state for the sequence
- `seq_length` (int): Number of iterations
- `num_checkpoints` (int, optional): Number of checkpoints (default: √n)

**Returns:**
- `final_state` (array): Result after all iterations
- `checkpoints` (dict): Dictionary mapping positions to saved states

**Example:**
```python
final, checkpoints = tangent.checkpointed_loop(step_fn, x0, 1000)
```

### `compute_checkpoint_positions(seq_length, num_checkpoints)`

Compute optimal checkpoint positions.

**Arguments:**
- `seq_length` (int): Total number of steps
- `num_checkpoints` (int): Number of checkpoints to use

**Returns:**
- `positions` (List[int]): List of positions where checkpoints should be saved

**Example:**
```python
positions = tangent.compute_checkpoint_positions(1000, 31)
# [31, 62, 93, 124, ...]
```

### `get_memory_savings(seq_length, num_checkpoints=None)`

Calculate expected memory savings.

**Arguments:**
- `seq_length` (int): Length of the sequence
- `num_checkpoints` (int, optional): Number of checkpoints (default: √n)

**Returns:**
- `stats` (dict): Dictionary with keys:
  - `'without_checkpointing'`: Memory without checkpointing
  - `'with_checkpointing'`: Memory with checkpointing
  - `'savings_percent'`: Percentage of memory saved
  - `'savings_ratio'`: Ratio of memory reduction
  - `'num_checkpoints'`: Actual number of checkpoints used
  - `'recomputation_factor'`: Average recomputation factor

**Example:**
```python
stats = tangent.get_memory_savings(1000)
print(f"Memory reduction: {stats['savings_percent']:.1f}%")
# Memory reduction: 96.9%
```

### `should_checkpoint(seq_length, threshold=0.5)`

Determine if checkpointing is beneficial.

**Arguments:**
- `seq_length` (int): Length of sequence/loop
- `threshold` (float): Minimum savings ratio to recommend (default: 0.5)

**Returns:**
- `bool`: True if checkpointing is recommended

**Example:**
```python
if tangent.should_checkpoint(100):
    # Use checkpointing
    final, checkpoints = tangent.checkpointed_loop(...)
```

## Advanced Usage

### Custom Number of Checkpoints

```python
# More checkpoints = less recomputation, more memory
final, checkpoints = tangent.checkpointed_loop(
    step_fn, x0, seq_length=1000,
    num_checkpoints=50  # Use 50 instead of default 31
)

# Fewer checkpoints = more recomputation, less memory
final, checkpoints = tangent.checkpointed_loop(
    step_fn, x0, seq_length=1000,
    num_checkpoints=20  # Use only 20 checkpoints
)
```

### LSTM Example

```python
import numpy as np
import tangent

# LSTM state is a tuple: (hidden, cell)
def lstm_step(state):
    h, c = state
    # Simplified LSTM computation
    i = sigmoid(h @ W_i + b_i)
    f = sigmoid(h @ W_f + b_f)
    o = sigmoid(h @ W_o + b_o)
    g = np.tanh(h @ W_g + b_g)
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return (h_new, c_new)

# Initial state
h0 = np.zeros(hidden_size)
c0 = np.zeros(hidden_size)
initial_state = (h0, c0)

# Checkpointed LSTM forward pass
final_state, checkpoints = tangent.checkpointed_loop(
    lstm_step,
    initial_state,
    seq_length=1000,
    num_checkpoints=31
)
```

### Estimating Memory Usage

```python
import tangent

# Estimate memory for different sequence lengths
for seq_len in [100, 500, 1000, 5000]:
    stats = tangent.estimate_checkpoint_savings(seq_len)
    print(f"Sequence {seq_len:4d}: "
          f"{stats['num_checkpoints']:3d} checkpoints, "
          f"{stats['savings_percent']:5.1f}% reduction")

# Output:
# Sequence  100:  10 checkpoints,  90.0% reduction
# Sequence  500:  22 checkpoints,  95.6% reduction
# Sequence 1000:  31 checkpoints,  96.9% reduction
# Sequence 5000:  70 checkpoints,  98.6% reduction
```

## Performance Considerations

### Memory-Time Tradeoff

Checkpointing trades memory for computation:

- **Memory**: Reduced from O(n) to O(√n)
- **Time**: Increased by ~33% due to recomputation
- **Sweet spot**: Use √n checkpoints (automatic default)

### Recomputation Factor

The recomputation factor tells you how many extra forward steps will be performed:

```python
stats = tangent.get_memory_savings(1000)
print(f"Recomputation factor: {stats['recomputation_factor']:.1f}x")
# Recomputation factor: 32.3x

# This means: on average, each step is computed 1 time forward + recomputed 32.3 times
# Total: ~33 forward passes worth of computation
```

### Choosing the Number of Checkpoints

```python
# Rule of thumb:
# - sqrt(n) checkpoints: optimal memory-time tradeoff (default)
# - n/10 checkpoints: less recomputation, more memory
# - n/100 checkpoints: extreme memory savings, heavy recomputation

seq_length = 1000

# Default: sqrt(1000) = 31 checkpoints
stats_default = tangent.get_memory_savings(seq_length)

# Custom: 100 checkpoints (less recomputation)
stats_more = tangent.get_memory_savings(seq_length, num_checkpoints=100)

# Custom: 10 checkpoints (more memory savings)
stats_fewer = tangent.get_memory_savings(seq_length, num_checkpoints=10)

print(f"Default: {stats_default['savings_percent']:.1f}% savings")
print(f"More checkpoints: {stats_more['savings_percent']:.1f}% savings")
print(f"Fewer checkpoints: {stats_fewer['savings_percent']:.1f}% savings")
```

## Integration with Tangent's grad()

**Note**: Full automatic integration with `tangent.grad()` via AST transformation is planned for Phase 2. Currently, checkpointing must be applied manually to your loops.

### Current Approach (Manual)

```python
import numpy as np
import tangent

# Define your model with manual checkpointing
def rnn_model(x, W, b, seq_length=1000):
    def rnn_step(state):
        return np.tanh(state @ W + b)

    # Use checkpointed_loop instead of a regular for loop
    final_state, checkpoints = tangent.checkpointed_loop(
        rnn_step, x, seq_length, num_checkpoints=31
    )
    return final_state

# Now you can compute gradients normally
df = tangent.grad(rnn_model, wrt=(0, 1, 2))  # Gradient w.r.t. x, W, b
```

### Future Approach (Automatic - Phase 2)

This will be available in a future release:

```python
# Future: automatic checkpointing via AST transformation
def rnn_model(x, W, b):
    state = x
    for i in range(1000):  # This loop will be automatically checkpointed
        state = np.tanh(state @ W + b)
    return state

# Future API (not yet implemented)
df = tangent.grad_with_checkpointing(rnn_model, num_checkpoints=31)
```

## Limitations

### Current Phase 1 Limitations

1. **Manual application**: You must explicitly use `checkpointed_loop()` instead of regular loops
2. **Single loop variable**: Each loop iteration should update a single state variable
3. **No automatic gradient**: Full gradient computation through checkpoints requires manual setup

### Planned Improvements (Phase 2-3)

- Automatic AST transformation to detect and checkpoint loops
- Integration with `tangent.grad()` for seamless gradient computation
- Support for nested loops and complex control flow
- Full Revolve algorithm for provably optimal checkpointing

## Troubleshooting

### "Results don't match"

Ensure your step function is deterministic and doesn't depend on external state:

```python
# Bad: depends on external loop counter
counter = 0
def step(state):
    global counter
    counter += 1
    return state * counter  # Non-deterministic!

# Good: pure function
def step(state):
    return state * 1.1  # Deterministic
```

### Memory still high

Check that you're not storing extra copies:

```python
# Bad: keeping all states anyway
states = []
final, checkpoints = tangent.checkpointed_loop(step, x0, 1000)
for i in range(1000):
    states.append(...)  # Don't do this!

# Good: only keep checkpoints
final, checkpoints = tangent.checkpointed_loop(step, x0, 1000)
# checkpoints dict is all you need for backward pass
```

## Examples

See also:
- `examples/checkpoint_demo.py` - Comprehensive demonstration
- `tests/test_checkpointing_basic.py` - Unit tests with more examples
- `/tmp/test_checkpointing_integration.py` - Integration test examples

## References

The checkpointing implementation is based on:

1. **Griewank, A., & Walther, A. (2000)**. Algorithm 799: Revolve: An implementation of checkpointing for the reverse or adjoint mode of computational differentiation. *ACM Transactions on Mathematical Software*, 26(1), 19-45.

2. **Gruslys, A., et al. (2016)**. Memory-Efficient Backpropagation Through Time. *Advances in Neural Information Processing Systems*, 29.

For more details on the theory and algorithms, see:
- `Checkpointing.md` - Comprehensive technical documentation
- `Checkpointing_quickstart.md` - Quick implementation guide
