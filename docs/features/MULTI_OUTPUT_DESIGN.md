# Multiple Return Values (Multi-Output Functions) - Design Document

## Problem Statement

Currently, Tangent auto-sums gradients when a function returns a tuple:

```python
def f(x):
    return x**2, x*3

df = tangent.grad(f)
result = df(2.0)  # = 7.0 (sum of gradients)
```

**This limits common ML patterns:**
1. **Loss + Metrics**: `return loss, accuracy, precision`
2. **Multi-head models**: `return class_logits, bbox_predictions`
3. **Auxiliary outputs**: `return main_output, auxiliary_loss`

## Use Cases

### Use Case 1: Training with Metrics
```python
def compute_loss_and_metrics(params, x, y):
    pred = model(params, x)
    loss = mse(pred, y)
    accuracy = compute_accuracy(pred, y)
    return loss, accuracy  # Want gradient of loss only

# Need: grad wrt loss, but also get accuracy value
```

### Use Case 2: Multi-Task Learning
```python
def multi_task_model(params, x):
    shared = shared_layers(params, x)
    task1_output = task1_head(shared)
    task2_output = task2_head(shared)
    return task1_output, task2_output

# Need: gradient of weighted sum or individual gradients
```

### Use Case 3: Regularized Loss
```python
def loss_with_reg(params, x, y):
    pred_loss = mse(model(params, x), y)
    reg_loss = l2_regularization(params)
    return pred_loss, reg_loss

# Need: gradient of (pred_loss + lambda * reg_loss)
```

## Design Options

### Option 1: Keep Current Behavior + Add `preserve_result=True`

**Current:**
```python
df = tangent.grad(f, preserve_result=True)
grads, (out1, out2) = df(x)  # Returns grads and all outputs
```

**Problem:** Still only gets gradient of sum, not individual control.

### Option 2: Add `output_gradients` Parameter

**Proposed:**
```python
# Specify which output to differentiate
df = tangent.grad(f, output_index=0)  # Gradient of first output only
grad = df(x)

# Or get all outputs
df = tangent.grad(f, output_index=0, preserve_result=True)
grad, (out1, out2) = df(x)
```

**Benefits:**
- Simple API
- Clear intent
- Backward compatible (default to sum)

### Option 3: Use `vjp` with Custom Seeds

**Current `vjp` approach:**
```python
from tangent import vjp

def f(x):
    return x**2, x*3

# Differentiate first output only
df = vjp(f)
grad = df(x, (1.0, 0.0))  # Seed = (1, 0) for first output

# Differentiate second output only
grad = df(x, (0.0, 1.0))  # Seed = (0, 1) for second output
```

**Problem:** Requires manual seed specification.

### Option 4: Return Multiple Gradient Functions (RECOMMENDED)

**Proposed:**
```python
# New API: tangent.grad returns tuple of gradient functions
df_tuple = tangent.grad(f, separate_outputs=True)

# df_tuple is (df1, df2) where:
# df1 = gradient wrt first output
# df2 = gradient wrt second output

df1, df2 = df_tuple
grad1 = df1(x)  # Gradient of first output
grad2 = df2(x)  # Gradient of second output

# Can also get all outputs preserved
df_tuple = tangent.grad(f, separate_outputs=True, preserve_result=True)
df1, df2 = df_tuple
grad1, (out1, out2) = df1(x)  # Grad of first + all outputs
```

**Benefits:**
- Most flexible
- Clear semantics
- Easy to use
- Composes well

## Recommended Solution

**Implement Option 2 + Option 4:**

### For Single Output Selection (Simple Case)
```python
# Get gradient of specific output
df = tangent.grad(f, output_index=0)
grad = df(x)  # Gradient of first output only
```

### For Multiple Independent Gradients (Advanced Case)
```python
# Get separate gradient functions
grads = tangent.grad_outputs(f)  # New function
df1, df2 = grads

# Use independently
grad1 = df1(x)  # Gradient of output 1
grad2 = df2(x)  # Gradient of output 2
```

### For Weighted Combination (Current Behavior)
```python
# Default: sum all outputs (backward compatible)
df = tangent.grad(f)
grad = df(x)  # Gradient of sum (current behavior)

# Custom weighting
df = tangent.grad(f, output_weights=(1.0, 0.5))
grad = df(x)  # Gradient of (1.0*out1 + 0.5*out2)
```

## Implementation Plan

### Phase 1: Add `output_index` parameter
- Modify `_grad_uncached` to accept `output_index`
- Generate gradient function for specific output
- Add tests

### Phase 2: Add `tangent.grad_outputs()`
- New high-level function
- Returns tuple of gradient functions
- Internally uses `output_index`

### Phase 3: Add `output_weights` parameter
- Allow custom weighting of outputs
- Generalization of current auto-sum behavior

## API Examples

### Example 1: Training Loss with Metrics
```python
def loss_and_accuracy(params, x, y):
    pred = model(params, x)
    loss = mse(pred, y)
    acc = accuracy(pred, y)
    return loss, acc

# Only differentiate loss (output 0)
dloss = tangent.grad(loss_and_accuracy, output_index=0)

# Get gradient and preserve both outputs
dloss = tangent.grad(loss_and_accuracy, output_index=0, preserve_result=True)
grad_params, (loss_val, acc_val) = dloss(params, x, y)
```

### Example 2: Multi-Task Learning
```python
def multi_task_loss(params, x, y1, y2):
    pred1, pred2 = model(params, x)
    loss1 = mse(pred1, y1)
    loss2 = mse(pred2, y2)
    return loss1, loss2

# Weighted combination
dloss = tangent.grad(multi_task_loss, output_weights=(0.7, 0.3))
grad = dloss(params, x, y1, y2)  # Gradient of 0.7*loss1 + 0.3*loss2

# Or separate gradients
dloss1, dloss2 = tangent.grad_outputs(multi_task_loss)
grad1 = dloss1(params, x, y1, y2)
grad2 = dloss2(params, x, y1, y2)
```

### Example 3: Regularized Loss
```python
def loss_with_reg(params, x, y):
    pred_loss = mse(model(params, x), y)
    reg_loss = l2_reg(params)
    return pred_loss, reg_loss

# Combined with custom weighting
lambda_reg = 0.01
dloss = tangent.grad(loss_with_reg, output_weights=(1.0, lambda_reg))
grad = dloss(params, x, y)
```

## Backward Compatibility

**Current behavior preserved:**
- `tangent.grad(f)` on tuple-returning function still sums (with warning)
- Users can opt-in to new behavior with `output_index` or `grad_outputs()`

## Testing Strategy

1. Test `output_index` with 2, 3, 4 outputs
2. Test `output_weights` with various combinations
3. Test `grad_outputs()` returns correct number of functions
4. Test with `preserve_result=True`
5. Test backward compatibility (default sum behavior)
6. Test with arrays and scalars
7. Test with complex ML scenarios

## Success Criteria

- [ ] Can differentiate specific output of multi-output function
- [ ] Can get separate gradient functions for each output
- [ ] Can specify custom output weights
- [ ] Preserves backward compatibility
- [ ] Clear documentation with ML examples
- [ ] Comprehensive test coverage
