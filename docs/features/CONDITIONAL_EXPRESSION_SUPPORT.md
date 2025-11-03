# Conditional Expression (Ternary Operator) Support in Tangent

**Status**: ✅ **COMPLETE** - All tests passing!

**Date**: 2025-11-03

## Summary

Tangent now fully supports Python's conditional expressions (ternary operator): `value_if_true if condition else value_if_false`.

This enables clean, concise code for conditional logic in differentiable functions, including common patterns like:
- ReLU and other activation functions
- Clipping and bounding operations
- Sign functions
- Piecewise functions
- Nested conditionals

## What Was Implemented

### 1. Fence Validation ([fence.py](tangent/fence.py:256))
- Changed `visit_IfExp` from rejecting to allowing conditional expressions

### 2. Naming Support ([naming.py](tangent/naming.py:358-389))
- Added `CMPOP_NAMES` dictionary mapping comparison operators to readable names
- Added `name_Compare()` method to handle comparison expressions like `x > 0`
- Added `name_IfExp()` method to generate names for ternary expressions

### 3. Reverse-Mode AD Support ([reverse_ad.py](tangent/reverse_ad.py:542-620))
- Added `visit_IfExp()` method that:
  - Visits both branches to compute their primals and adjoints
  - Handles nested ternary expressions correctly
  - Stores the condition on the stack for the backward pass
  - Routes gradients to the chosen branch in the adjoint

### 4. Constant Node Support ([reverse_ad.py](tangent/reverse_ad.py:754-756))
- Added `visit_Constant()` method to handle Python 3.8+ constant nodes
- Returns zero gradient for constants (correct mathematical behavior)

### 5. Gradient Templates ([grads.py](tangent/grads.py:224-239))
- Added primal template for IfExp (saves and pushes condition)
- Added adjoint template for IfExp (routes gradient to chosen branch)

## Technical Details

### How It Works

**Forward Pass (Primal)**:
```python
# For: result = body if test else orelse

# 1. Evaluate and save the condition
cond = test

# 2. Compute the ternary expression
result = body if cond else orelse

# 3. Push condition onto stack (needed for backward pass)
push(_stack, cond, op_id)
```

**Backward Pass (Adjoint)**:
```python
# 1. Pop the saved condition
cond = pop(_stack, op_id)

# 2. Route gradient to the branch that was executed
if cond:
    d[body] = d[result]  # Gradient flows through body
else:
    d[orelse] = d[result]  # Gradient flows through orelse
```

### Nested Conditional Handling

For nested ternaries like `x**2 if x > 1 else (x if x >= 0 else -x)`:
1. Inner ternary is visited first, producing primal statements and an expression
2. Preparatory statements from inner ternary are collected
3. Expression from inner ternary becomes the `orelse` branch of outer ternary
4. All statements are combined in the correct order

## Test Results

All 8 comprehensive tests passing:

| Test | Description | Status |
|------|-------------|--------|
| 1 | Simple sign function | ✅ PASS |
| 2 | Conditional with differentiable branches | ✅ PASS |
| 3 | Nested ternary expressions | ✅ PASS |
| 4 | ReLU activation function | ✅ PASS |
| 5 | Conditional within larger computation | ✅ PASS |
| 6 | Multiple independent ternary expressions | ✅ PASS |
| 7 | Different comparison operators | ✅ PASS |
| 8 | Ternary in return statement | ✅ PASS |

## Usage Examples

### Example 1: ReLU Activation
```python
import tangent

def relu(x):
    return x if x > 0 else 0.0

df = tangent.grad(relu)
print(df(5.0))   # 1.0 (positive input)
print(df(-3.0))  # 0.0 (negative input)
```

### Example 2: Clipping Function
```python
def clip(x, min_val=0.0, max_val=1.0):
    return min_val if x < min_val else (max_val if x > max_val else x)

df = tangent.grad(clip)
print(df(0.5))   # 1.0 (in range)
print(df(-1.0))  # 0.0 (clipped to min)
print(df(2.0))   # 0.0 (clipped to max)
```

### Example 3: Piecewise Function
```python
def piecewise(x):
    """Different behavior for different ranges."""
    return x**2 if x > 0 else 2*x

df = tangent.grad(piecewise)
print(df(3.0))   # 6.0 (derivative of x^2 at x=3)
print(df(-2.0))  # 2.0 (derivative of 2x)
```

### Example 4: Nested Conditionals
```python
def three_way(x):
    """Three different behaviors based on x."""
    return x**2 if x > 1 else (x if x >= 0 else -x)

df = tangent.grad(three_way)
print(df(2.0))   # 4.0 (derivative of x^2 at x=2)
print(df(0.5))   # 1.0 (derivative of x)
print(df(-1.0))  # -1.0 (derivative of -x)
```

## Files Modified

1. **[tangent/fence.py](tangent/fence.py)** (1 line changed)
   - Line 256: Changed from `self._reject(node, ...)` to `self._allow_and_continue(node)`

2. **[tangent/naming.py](tangent/naming.py)** (32 lines added)
   - Lines 358-389: Added comparison operator names and naming methods

3. **[tangent/reverse_ad.py](tangent/reverse_ad.py)** (82 lines added)
   - Lines 542-620: Added `visit_IfExp` method
   - Lines 754-756: Added `visit_Constant` method

4. **[tangent/grads.py](tangent/grads.py)** (16 lines added)
   - Lines 224-239: Added primal and adjoint templates

## Mathematical Correctness

The implementation correctly handles:
- **Gradient routing**: Only the executed branch receives gradients
- **Zero gradients for constants**: Constant branches produce zero gradients
- **Nested expressions**: Inner ternaries are differentiated correctly
- **Multiple conditionals**: Independent ternaries compose properly

## Performance Notes

- Condition evaluation happens once during forward pass
- Condition is stored on stack (minimal overhead)
- Backward pass uses stored condition (no re-evaluation)
- No performance degradation for nested ternaries

## Limitations

None known! The implementation handles:
- ✅ Simple ternaries
- ✅ Nested ternaries
- ✅ Multiple ternaries in sequence
- ✅ Ternaries in larger expressions
- ✅ All comparison operators (>, <, >=, <=, ==, !=, is, is not, in, not in)
- ✅ Constant and variable branches
- ✅ Complex expressions in branches

## Next Steps

This completes the conditional expression support. Next "quick win" features to consider:
- Dictionary comprehensions
- List comprehensions
- Set comprehensions
- Boolean operators (and, or, not)
- Augmented assignments (+=, -=, etc.)
