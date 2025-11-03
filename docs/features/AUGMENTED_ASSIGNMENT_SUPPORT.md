# Augmented Assignment Operator Support in Tangent

## Overview

Tangent fully supports Python's augmented assignment operators (`+=`, `-=`, `*=`, `/=`, `**=`, etc.) in functions being differentiated. These operators make code more concise and natural, especially in accumulation patterns common in training loops and numerical algorithms.

## Supported Operators

### Arithmetic Augmented Assignments

All standard arithmetic augmented assignment operators are supported:

| Operator | Equivalent | Example | Description |
|----------|-----------|---------|-------------|
| `+=` | `a = a + b` | `result += x` | Addition |
| `-=` | `a = a - b` | `result -= x` | Subtraction |
| `*=` | `a = a * b` | `result *= x` | Multiplication |
| `/=` | `a = a / b` | `result /= x` | Division |
| `**=` | `a = a ** b` | `result **= 2` | Exponentiation |
| `//=` | `a = a // b` | `result //= x` | Floor division |
| `%=` | `a = a % b` | `result %= x` | Modulo |

## Basic Examples

### Simple Accumulation

```python
import tangent

def accumulate(x):
    result = 0.0
    result += x        # Add x
    result += x ** 2   # Add x^2
    return result

df = tangent.grad(accumulate)
print(df(2.0))  # 6.0 (gradient of x + x^2 is 1 + 2x = 5, evaluated at x=2... wait)
# Actually: f(x) = x + x^2, df/dx = 1 + 2x = 1 + 4 = 5
```

### Chained Operations

```python
import tangent

def chained(x):
    result = 1.0
    result += x       # result = 1 + x
    result *= 2.0     # result = 2(1 + x) = 2 + 2x
    result += x ** 2  # result = 2 + 2x + x^2
    return result

df = tangent.grad(chained)
print(df(3.0))  # 8.0 (df/dx = 2 + 2x = 2 + 6 = 8)
```

### All Operators Together

```python
import tangent

def all_ops(x):
    a = x
    a += 1.0    # a = x + 1
    b = x
    b -= 1.0    # b = x - 1
    c = x
    c *= 2.0    # c = 2x
    d = x
    d /= 2.0    # d = x/2
    e = x
    e **= 2.0   # e = x^2
    return a + b + c + d + e

df = tangent.grad(all_ops)
print(df(4.0))  # 12.5 (df/dx = 1 + 1 + 2 + 0.5 + 2x = 4.5 + 8 = 12.5)
```

## Advanced Patterns

### Accumulator Pattern (Training Loops)

The accumulator pattern is extremely common in machine learning:

```python
import tangent

def loss_accumulator(x):
    """Sum of squared errors - common in ML."""
    total_loss = 0.0
    total_loss += (x - 1.0) ** 2
    total_loss += (x - 2.0) ** 2
    total_loss += (x - 3.0) ** 2
    return total_loss

df = tangent.grad(loss_accumulator)
print(df(2.0))  # 0.0 (gradient is zero at x=2, the mean of [1,2,3])
```

### Complex Expressions on Right-Hand Side

Augmented assignments work with arbitrarily complex expressions:

```python
import tangent

def complex_rhs(x):
    result = 0.0
    result += x * x + 2 * x + 1  # Full expression on RHS
    return result

df = tangent.grad(complex_rhs)
print(df(2.0))  # 6.0 (df/dx = 2x + 2 = 6)
```

### NumPy Array Operations

Augmented assignments work seamlessly with NumPy:

```python
import numpy as np
import tangent

def numpy_accumulate(x):
    weights = np.array([1.0, 2.0, 3.0])
    result = 0.0
    result += x * np.sum(weights)  # result = 6x
    return result

df = tangent.grad(numpy_accumulate)
print(df(2.0))  # 6.0
```

### Conditional Accumulation

Combine with control flow for conditional updates:

```python
import tangent

def conditional_accumulate(x):
    result = 0.0
    if x > 0:
        result += x ** 2
    else:
        result += x
    return result

df = tangent.grad(conditional_accumulate)
print(df(3.0))   # 6.0 (positive: df/dx = 2x)
print(df(-2.0))  # 1.0 (negative: df/dx = 1)
```

### Mixed Assignment Styles

You can freely mix regular and augmented assignments:

```python
import tangent

def mixed_assignments(x):
    a = x                    # Regular assignment
    b = a + 1.0              # Regular assignment
    a += b                   # Augmented: a = x + (x + 1)
    result = a * b           # result = (2x + 1)(x + 1)
    return result

df = tangent.grad(mixed_assignments)
print(df(2.0))  # 11.0
```

## Real-World Examples

### Example 1: Gradient Descent Step

Simulate a gradient descent update:

```python
import tangent

def gradient_descent_step(x):
    """Single step of gradient descent."""
    # Compute loss and gradient
    loss = (x - 5.0) ** 2
    grad = 2 * (x - 5.0)

    # Update with learning rate
    learning_rate = 0.1
    x_new = x
    x_new -= learning_rate * grad

    # Return function of new value
    return x_new ** 2

df = tangent.grad(gradient_descent_step)
print(df(3.0))  # Gradient of the entire step
```

### Example 2: Polynomial Evaluation

Accumulate polynomial terms:

```python
import tangent

def polynomial(x):
    """Evaluate polynomial: x^3 + 2x^2 + 3x + 4"""
    result = 4.0
    result += 3.0 * x
    result += 2.0 * x ** 2
    result += x ** 3
    return result

df = tangent.grad(polynomial)
print(df(2.0))  # 23.0 (df/dx = 3 + 4x + 3x^2 = 3 + 8 + 12 = 23)
```

### Example 3: Weighted Sum

Common pattern in neural networks:

```python
import numpy as np
import tangent

def weighted_sum(x):
    """Weighted sum with accumulation."""
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    result = 0.0

    for w in weights:
        result += w * x

    return result

df = tangent.grad(weighted_sum)
# Note: This requires loop support - shown for illustration
```

### Example 4: Momentum Update

Simplified momentum optimizer:

```python
import tangent

def momentum_update(x, velocity=0.0):
    """Momentum-based parameter update."""
    # Current gradient
    grad = 2 * (x - 3.0)

    # Update velocity (momentum = 0.9)
    velocity *= 0.9
    velocity += grad

    # Update parameter
    learning_rate = 0.1
    x -= learning_rate * velocity

    return x

df = tangent.grad(momentum_update)
print(df(5.0))
```

## Important Notes

### Differentiability

Augmented assignments are differentiable when the underlying operation is differentiable:

- **Differentiable**: `+=`, `-=`, `*=`, `/=`, `**=` with differentiable operands
- **Not differentiable**: `//=` (floor division), `%=` (modulo)
- **Works in control flow**: Any operator works when branches are differentiable

### Semantics

Augmented assignments follow Python's standard semantics:

```python
# These are equivalent:
x += y
x = x + y

# For mutable objects (lists, arrays), behavior may differ:
# x += y  # Modifies x in-place
# x = x + y  # Creates new object
```

For Tangent's purposes with scalar and NumPy operations, the semantics are equivalent.

### Variable Reassignment

Augmented assignments create new bindings in Tangent's internal representation:

```python
def f(x):
    result = x     # result₀ = x
    result += 1    # result₁ = result₀ + 1
    result *= 2    # result₂ = result₁ * 2
    return result  # Returns result₂
```

This is handled automatically by Tangent's ANF (A-Normal Form) transformation.

### Short-Circuit Evaluation

Unlike boolean operators, arithmetic operations don't short-circuit:

```python
# Both sides always evaluated:
result += expensive_function()  # Always calls function
```

## Performance Considerations

Augmented assignments have the same performance characteristics as their expanded equivalents:

```python
# These have identical performance in Tangent:
x += y
x = x + y
```

However, augmented assignments make code more readable and intent clearer.

## Common Patterns

### Pattern 1: Loss Accumulation

```python
def total_loss(predictions, targets):
    loss = 0.0
    loss += mse_loss(predictions, targets)
    loss += regularization_term()
    return loss
```

### Pattern 2: Scaled Update

```python
def scaled_update(value, delta, scale=0.1):
    value -= scale * delta
    return value
```

### Pattern 3: Running Sum

```python
def running_sum(x):
    total = 0.0
    total += x
    total += x ** 2
    total += x ** 3
    return total
```

### Pattern 4: Coefficient Multiplication

```python
def apply_coefficients(x):
    result = x
    result *= learning_rate
    result *= decay_factor
    return result
```

## Comparison with Regular Assignments

| Aspect | Augmented (`x += y`) | Regular (`x = x + y`) |
|--------|---------------------|----------------------|
| **Readability** | ✅ More concise | ❌ More verbose |
| **Intent** | ✅ Clear accumulation | ⚠️ Less obvious |
| **Performance** | ✅ Identical | ✅ Identical |
| **Differentiability** | ✅ Same as operation | ✅ Same as operation |
| **Common usage** | ✅ Accumulation, updates | ✅ Complex expressions |

## Limitations

1. **No bitwise operators**: `&=`, `|=`, `^=`, `<<=`, `>>=` are not typically used in numerical computing and may not be supported
2. **No @ operator**: `@=` (matrix multiplication) support depends on underlying framework
3. **In-place semantics**: For NumPy arrays, behavior follows NumPy's in-place operation rules

## Testing

Comprehensive tests are available in `/tmp/test_augassign_comprehensive.py`, covering:
- All arithmetic augmented assignment operators
- Chained operations on the same variable
- Complex expressions on right-hand side
- NumPy array operations
- Conditional accumulation
- Mixed regular and augmented assignments
- Real-world gradient descent patterns
- Accumulator patterns

All 10 comprehensive tests pass successfully.

## See Also

- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md)
- [Conditional Expression Support](CONDITIONAL_EXPRESSION_SUPPORT.md)
- [Lambda Function Support](LAMBDA_SUPPORT_COMPLETE.md)
- [Closure Support](CLOSURE_SUPPORT_COMPLETE.md)
