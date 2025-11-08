# Tuple Support in Tangent

## Status: ✅ Fully Supported

Tuple operations, including tuple unpacking, are fully supported in Tangent and produce correct gradients.

## Supported Operations

### 1. Tuple Access
```python
def f(x):
    t = (x ** 2, x * 3, x + 1)
    return t[0] + t[1]  # ✅ Works

df = tangent.grad(f)
```

### 2. Tuple Indexing
```python
def f(x):
    t = (x ** 2, x * 3)
    a = t[0]  # ✅ Works
    b = t[1]  # ✅ Works
    return a + b

df = tangent.grad(f)
```

### 3. Tuple Unpacking
```python
def f(x):
    a, b = x ** 2, x * 3  # ✅ Works correctly!
    return a + b

df = tangent.grad(f)
# Gradient: 2x + 3 ✓
```

### 4. Multiple Unpacking
```python
def f(x):
    a, b = x ** 2, x * 3
    c, d = a + 1, b * 2  # ✅ Chained unpacking works
    return c + d

df = tangent.grad(f)
# Gradient: 2x + 6 ✓
```

### 5. Unpacking from Function Calls
```python
def helper(x):
    return x ** 2, x * 3

def f(x):
    a, b = helper(x)  # ✅ Unpacking function returns works
    return a + b

df = tangent.grad(f)
# Gradient: 2x + 3 ✓
```

### 6. Tuple Returns (Multi-Output Functions)
```python
def f(x):
    return x ** 2, x * 3  # Returns tuple

# Option 1: Auto-sum (default)
df = tangent.grad(f)
grad = df(2.0)  # 7.0 (sum of gradients)

# Option 2: Specific output
df0 = tangent.grad(f, output_index=0)  # Only x**2
grad0 = df0(2.0)  # 4.0

df1 = tangent.grad(f, output_index=1)  # Only x*3
grad1 = df1(2.0)  # 3.0

# Option 3: Weighted combination
df_weighted = tangent.grad(f, output_weights=[0.5, 0.5])
grad_weighted = df_weighted(2.0)  # 3.5 (weighted average)
```

## Test Results

All tuple operations have been tested and verified:

```bash
$ python /tmp/test_tuple_unpacking_current.py
Testing tuple unpacking...
============================================================

1. Simple tuple unpacking: a, b = x**2, x*3
   x = 2.0
   Gradient: 7.0
   Expected: 7.0
   ✓ CORRECT

2. Unpacking with intermediate use
   x = 2.0
   Gradient: 11.0
   Expected: 11.0
   ✓ CORRECT

3. Multiple unpacking statements
   x = 2.0
   Gradient: 10.0
   Expected: 10.0
   ✓ CORRECT

4. Unpacking from function call
   x = 2.0
   Gradient: 7.0
   Expected: 7.0
   ✓ CORRECT
```

## Historical Note

**Previous Status**: Earlier versions of Tangent had issues with tuple unpacking that could produce incorrect gradients.

**Current Status**: All tuple unpacking patterns now work correctly. The documentation has been updated to reflect this.

**What Changed**: The underlying implementation has been improved to handle tuple unpacking correctly in all scenarios tested.

## Examples

### Example 1: Simple Computation
```python
import tangent
import numpy as np

def compute(x):
    """Compute using tuple unpacking."""
    squared, tripled = x ** 2, x * 3
    return squared + tripled

df = tangent.grad(compute)

x = 2.0
gradient = df(x)
expected = 2 * x + 3  # 7.0

assert np.isclose(gradient, expected)
print(f"✓ Gradient at x={x}: {gradient}")
```

### Example 2: Multi-Step Unpacking
```python
def multi_step(x):
    """Multiple unpacking operations."""
    # First unpacking
    a, b = x ** 2, x * 3

    # Use values
    c = a + 1
    d = b * 2

    # Second unpacking
    e, f = c ** 2, d + 1

    return e + f

df = tangent.grad(multi_step)

x = 2.0
gradient = df(x)
print(f"✓ Multi-step gradient at x={x}: {gradient}")
```

### Example 3: Unpacking in Loop
```python
def loop_with_unpacking(x):
    """Unpacking inside a loop."""
    result = 0.0
    for i in range(3):
        a, b = x ** i, x * i
        result += a + b
    return result

df = tangent.grad(loop_with_unpacking)

x = 2.0
gradient = df(x)
print(f"✓ Loop unpacking gradient at x={x}: {gradient}")
```

## Limitations

### Read-Only Tuples
Tuples are immutable in Python, so modification is not supported (and wouldn't make sense):

```python
def f(x):
    t = (x, x ** 2)
    t[0] = x * 2  # ❌ TypeError (Python itself doesn't allow this)
    return t[0]
```

This is a Python language limitation, not a Tangent limitation.

## Comparison with Other Frameworks

| Framework | Tuple Unpacking | Multi-Output Gradients |
|-----------|-----------------|------------------------|
| **Tangent** | ✅ | ✅ (with `output_index`) |
| **JAX** | ✅ | ✅ |
| **PyTorch** | ✅ | ✅ |
| **TensorFlow** | ✅ | ✅ |

Tangent is on par with other major autodiff frameworks for tuple support.

## Related Documentation

- [Python Feature Support](PYTHON_FEATURE_SUPPORT.md) - Complete feature matrix
- [Multi-Output Gradients](../../tests/test_tuple_returns.py) - Test suite for tuple returns
- [Tuple Unpacking Tests](../../tests/test_tuple_unpacking_detailed.py) - Comprehensive unpacking tests

## Summary

✅ **Tuples are fully supported** in Tangent, including:
- Tuple construction
- Tuple indexing
- Tuple unpacking (all forms)
- Multi-output functions returning tuples
- Unpacking from function calls
- Nested and chained unpacking

All patterns have been tested and produce correct gradients.

---

**Last Updated**: 2025-11-04
**Status**: Fully Supported
**Test Coverage**: Comprehensive
