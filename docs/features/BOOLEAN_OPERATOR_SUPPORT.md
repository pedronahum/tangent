# Boolean Operator Support in Tangent

## Overview

Tangent now supports Python's boolean operators (`and`, `or`, `not`) in functions being differentiated. These operators work naturally in control flow constructs like `if` statements and conditional expressions.

## Supported Operators

### 1. `and` Operator

The `and` operator performs short-circuit evaluation: if the left operand is `False`, the right operand is not evaluated.

```python
import tangent

def f(x):
    return x if x > 0 and x < 10 else x ** 2

df = tangent.grad(f)
print(df(5.0))    # 1.0 (in range [0,10])
print(df(15.0))   # 30.0 (out of range, returns x^2)
```

### 2. `or` Operator

The `or` operator performs short-circuit evaluation: if the left operand is `True`, the right operand is not evaluated.

```python
import tangent

def f(x):
    return x ** 2 if x < 0 or x > 10 else x

df = tangent.grad(f)
print(df(5.0))    # 1.0 (in range)
print(df(-2.0))   # -4.0 (out of range, returns x^2)
print(df(15.0))   # 30.0 (out of range, returns x^2)
```

### 3. `not` Operator

The `not` operator inverts a boolean value.

```python
import tangent

def f(x):
    is_negative = x < 0
    return x ** 2 if not is_negative else 0.0

df = tangent.grad(f)
print(df(3.0))    # 6.0 (positive, returns x^2)
print(df(-2.0))   # 0.0 (negative, returns 0)
```

## Complex Boolean Expressions

Boolean operators can be chained and nested to create complex conditions:

```python
import tangent

def f(x):
    """Returns x^2 if x is in [0,5) or [10,15), otherwise returns x."""
    return x ** 2 if (x > 0 and x < 5) or (x > 10 and x < 15) else x

df = tangent.grad(f)
print(df(3.0))    # 6.0 (in first range)
print(df(7.0))    # 1.0 (middle range)
print(df(12.0))   # 24.0 (in second range)
print(df(20.0))   # 1.0 (out of both ranges)
```

## Multiple Operators in Sequence

You can use multiple boolean operators in a single expression:

```python
import tangent

def f(x):
    return x ** 3 if x > 0 and x < 10 and x != 5 else x

df = tangent.grad(f)
print(df(3.0))    # 27.0 (all conditions true)
print(df(5.0))    # 1.0 (x == 5)
print(df(15.0))   # 1.0 (x > 10)
```

## Nested Boolean Operators

Boolean operators can be nested with parentheses:

```python
import tangent

def f(x):
    """Returns x^2 if x is in [0,10], otherwise x."""
    return x ** 2 if not (x < 0 or x > 10) else x

df = tangent.grad(f)
print(df(5.0))    # 10.0 (in range)
print(df(-2.0))   # 1.0 (out of range)
print(df(15.0))   # 1.0 (out of range)
```

## Boolean Operators with NumPy

Boolean operators work seamlessly with NumPy comparisons:

```python
import numpy as np
import tangent

def f(x):
    arr = np.array([1.0, 2.0, 3.0])
    in_range = x > np.min(arr) and x < np.max(arr)
    return x ** 2 if in_range else x

df = tangent.grad(f)
print(df(2.0))    # 4.0 (in range [1,3])
print(df(5.0))    # 1.0 (out of range)
```

## Important Notes

### Non-Differentiable Values

Boolean operations themselves are **not differentiable** because they return discrete values (`True` or `False`). However, they work perfectly in control flow constructs because the branches they control contain differentiable code.

```python
# This works - boolean used in control flow
def f(x):
    return x ** 2 if x > 0 and x < 10 else x

# This also works - boolean stored in variable, used in control flow
def g(x):
    is_valid = x > 0 and x < 10
    return x ** 2 if is_valid else x
```

### Short-Circuit Evaluation

Python's short-circuit evaluation is preserved:
- `and`: If left operand is `False`, right operand is not evaluated
- `or`: If left operand is `True`, right operand is not evaluated

This means side effects in the right operand may not occur:

```python
def f(x):
    # If x <= 0, the x < 10 comparison is never evaluated
    return x if x > 0 and x < 10 else 0.0
```

### Operator Precedence

Standard Python operator precedence applies:
1. `not` (highest precedence)
2. `and`
3. `or` (lowest precedence)

Use parentheses to make complex expressions clearer:

```python
# These are equivalent due to precedence:
x > 0 and x < 10 or x > 20 and x < 30
(x > 0 and x < 10) or (x > 20 and x < 30)

# Use parentheses for clarity
(x > 0 and x < 10) or (x > 20 and x < 30)
```

## Implementation Details

Boolean operator support required three small changes:

1. **Fence validation** ([fence.py:158-159](tangent/fence.py#L158-L159)): Enabled the `Not` operator
2. **Naming support** ([naming.py:384-394](tangent/naming.py#L384-L394)): Added naming for `BoolOp` nodes in ANF transformation
3. **Reverse-mode AD** ([reverse_ad.py:859-861](tangent/reverse_ad.py#L859-L861)): Marked boolean operations as non-differentiable

## Limitations

1. Boolean operators cannot be differentiated directly - they work in control flow only
2. Boolean expressions must evaluate to standard Python booleans (not arrays of booleans)
3. Boolean operators in list comprehensions work naturally (the comprehension itself may have limitations)

## Testing

Comprehensive tests are available in `/tmp/test_boolop_comprehensive.py`, covering:
- Simple `and`, `or`, `not` operators
- Complex chained expressions
- Multiple operators in sequence
- Boolean operators in assignments
- Boolean operators with NumPy
- Nested boolean operators

All tests pass successfully.

## See Also

- [Conditional Expression Support](CONDITIONAL_EXPRESSION_SUPPORT.md)
- [Lambda Function Support](LAMBDA_SUPPORT.md)
- [Closure Support](CLOSURE_SUPPORT.md)
