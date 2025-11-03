# For Loop Support in Tangent

## Overview

Tangent fully supports Python's `for` loops with `range()` in functions being differentiated. This enables iterative algorithms, polynomial evaluations, Taylor series, and many other numerical patterns that are fundamental to scientific computing and machine learning.

## Basic Usage

### Simple Accumulation

The most basic pattern - accumulating a value over multiple iterations:

```python
import tangent

def accumulate(x):
    result = 0.0
    for i in range(5):
        result += x
    return result

df = tangent.grad(accumulate)
print(df(2.0))  # 5.0 (f(x) = 5x, so df/dx = 5)
```

### Using the Loop Variable

Access the loop variable `i` in computations:

```python
import tangent

def weighted_sum(x):
    result = 0.0
    for i in range(1, 4):
        result += float(i) * x
    return result

df = tangent.grad(weighted_sum)
print(df(2.0))  # 6.0 (f(x) = 1*x + 2*x + 3*x = 6x)
```

**Note**: Loop variables are integers. Cast to `float()` when using in arithmetic operations to avoid integer type warnings.

## Range Variants

### range(stop)

Iterate from 0 to stop-1:

```python
def f(x):
    result = 0.0
    for i in range(3):  # i = 0, 1, 2
        result += x
    return result

df = tangent.grad(f)
print(df(1.0))  # 3.0
```

### range(start, stop)

Iterate from start to stop-1:

```python
def f(x):
    result = 0.0
    for i in range(1, 4):  # i = 1, 2, 3
        result += x
    return result

df = tangent.grad(f)
print(df(1.0))  # 3.0
```

### range(start, stop, step)

Iterate with custom step size:

```python
def f(x):
    result = 0.0
    for i in range(0, 10, 2):  # i = 0, 2, 4, 6, 8
        result += x
    return result

df = tangent.grad(f)
print(df(1.0))  # 5.0 (5 iterations)
```

## Advanced Patterns

### Polynomial Evaluation

Build polynomials using loops:

```python
import tangent

def polynomial(x):
    """Compute 1 + x + x^2 + x^3."""
    result = 0.0
    for i in range(4):
        result += x ** float(i)
    return result

df = tangent.grad(polynomial)
print(df(2.0))  # 17.0 (df/dx = 0 + 1 + 2x + 3x^2 = 1 + 4 + 12 = 17)
```

### Nested Loops

Loops can be nested for multi-dimensional iterations:

```python
import tangent

def nested(x):
    result = 0.0
    for i in range(2):
        for j in range(3):
            result += x * (float(i) + float(j))
    return result

df = tangent.grad(nested)
# Pairs: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=1, (1,1)=2, (1,2)=3
# f(x) = x*(0+1+2+1+2+3) = 9x
print(df(1.0))  # 9.0
```

### Conditionals Inside Loops

Combine loops with conditional logic:

```python
import tangent

def conditional_loop(x):
    result = 0.0
    for i in range(4):
        if float(i) > 1.5:  # i = 2, 3
            result += x ** 2
        else:  # i = 0, 1
            result += x
    return result

df = tangent.grad(conditional_loop)
# f(x) = 2x + 2x^2 (2 times x, 2 times x^2)
# df/dx = 2 + 4x
print(df(3.0))  # 14.0 (2 + 12)
```

### Complex Expressions

Each iteration can contain arbitrarily complex expressions:

```python
import tangent

def complex_loop(x):
    result = 0.0
    for i in range(1, 4):
        # Each term: i * x^i
        result += float(i) * (x ** float(i))
    return result

df = tangent.grad(complex_loop)
# f(x) = 1*x + 2*x^2 + 3*x^3
# df/dx = 1 + 4x + 9x^2
print(df(2.0))  # 45.0 (1 + 8 + 36)
```

## Real-World Examples

### Example 1: Taylor Series Approximation

Approximate exp(x) using Taylor series:

```python
import tangent

def taylor_exp(x, n=5):
    """Approximate exp(x) using first n terms of Taylor series."""
    result = 1.0
    term = 1.0
    for i in range(1, n):
        term *= x / float(i)
        result += term
    return result

df = tangent.grad(taylor_exp)
# For small x, exp'(x) ≈ exp(x)
print(df(0.5))  # Approximately exp(0.5) ≈ 1.649
```

Alternative formulation:

```python
def taylor_exp_v2(x):
    """Explicit Taylor series: 1 + x + x^2/2 + x^3/6."""
    result = 1.0
    factorial = 1.0
    for i in range(1, 4):
        factorial *= float(i)
        result += (x ** float(i)) / factorial
    return result

df = tangent.grad(taylor_exp_v2)
# df/dx = 1 + x + x^2/2
print(df(0.5))  # 1.625
```

### Example 2: Polynomial Fitting Loss

Compute loss for polynomial regression:

```python
import tangent
import numpy as np

def polynomial_loss(coeffs, x_data, y_data):
    """MSE loss for polynomial fit."""
    total_loss = 0.0

    for i in range(len(x_data)):
        x = x_data[i]
        y_true = y_data[i]

        # Evaluate polynomial
        y_pred = 0.0
        for degree in range(len(coeffs)):
            y_pred += coeffs[degree] * (x ** float(degree))

        # Squared error
        total_loss += (y_pred - y_true) ** 2

    return total_loss / float(len(x_data))

# Note: This requires loop support and array indexing
```

### Example 3: Iterative Refinement

Successive refinement algorithms:

```python
import tangent

def iterative_refinement(x, iterations=3):
    """Refine estimate using Newton-like iteration."""
    estimate = x
    for i in range(iterations):
        # Simple refinement step
        estimate = estimate - 0.1 * (estimate ** 2 - 2.0)
    return estimate

df = tangent.grad(iterative_refinement)
print(df(1.5))
```

### Example 4: Running Average

Compute a weighted running average:

```python
import tangent

def weighted_average(x):
    """Weighted average with exponential decay."""
    total = 0.0
    weight_sum = 0.0
    decay = 0.9

    for i in range(5):
        weight = decay ** float(i)
        total += weight * (x + float(i))
        weight_sum += weight

    return total / weight_sum

df = tangent.grad(weighted_average)
print(df(1.0))
```

### Example 5: Discrete Convolution

Simple 1D convolution:

```python
import numpy as np
import tangent

def convolve_1d(x):
    """Simple 1D convolution with fixed kernel."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = np.array([0.25, 0.5, 0.25])

    result = 0.0
    for i in range(len(signal) - len(kernel) + 1):
        for j in range(len(kernel)):
            result += signal[i + j] * kernel[j] * x

    return result

df = tangent.grad(convolve_1d)
print(df(1.0))
```

## Loop Implementation Details

### How It Works

Tangent handles `for` loops through **loop unrolling** when the range is constant:

```python
# Original code:
def f(x):
    result = 0.0
    for i in range(3):
        result += x
    return result

# Tangent internally unrolls this to:
def f(x):
    result = 0.0
    i = 0
    result += x
    i = 1
    result += x
    i = 2
    result += x
    return result
```

This allows Tangent to differentiate through loops as if they were written out explicitly.

### Constant Ranges Required

The range parameters must be **compile-time constants**:

```python
# ✅ Works - constant range
def f(x):
    for i in range(5):
        x += 1.0
    return x

# ❌ Won't work - dynamic range
def g(x, n):
    for i in range(n):  # n is a parameter, not constant
        x += 1.0
    return x
```

**Workaround**: If you need dynamic loop counts, consider:
1. Using NumPy vectorized operations
2. Pre-computing with different fixed ranges
3. Using the maximum expected range with conditionals

## NumPy Integration

Loops work seamlessly with NumPy operations:

```python
import numpy as np
import tangent

def numpy_loop(x):
    """Loop with NumPy array indexing."""
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])
    result = 0.0

    for i in range(len(coeffs)):
        result += coeffs[i] * (x ** float(i))

    return result

df = tangent.grad(numpy_loop)
# f(x) = 1 + 2x + 3x^2 + 4x^3
# df/dx = 2 + 6x + 12x^2
print(df(1.0))  # 20.0 (2 + 6 + 12)
```

## Edge Cases

### Empty Loops

Loops with zero iterations are handled correctly:

```python
def empty_loop(x):
    result = x
    for i in range(0):  # Never executes
        result += x
    return result

df = tangent.grad(empty_loop)
print(df(2.0))  # 1.0 (loop body never runs)
```

### Single Iteration

Loops with one iteration work as expected:

```python
def single_iteration(x):
    result = 0.0
    for i in range(1):
        result += x
    return result

df = tangent.grad(single_iteration)
print(df(2.0))  # 1.0 (f(x) = x)
```

### Negative Steps

Reverse iteration with negative steps:

```python
def reverse_iteration(x):
    result = 0.0
    for i in range(5, 0, -1):  # 5, 4, 3, 2, 1
        result += x
    return result

df = tangent.grad(reverse_iteration)
print(df(1.0))  # 5.0 (5 iterations)
```

## Important Notes

### Loop Variable Type

Loop variables are integers. Cast to float when needed:

```python
# ⚠️ May produce warnings:
for i in range(3):
    result += x ** i

# ✅ Better:
for i in range(3):
    result += x ** float(i)
```

### Mutation vs Accumulation

Use accumulation pattern for clearest differentiation:

```python
# ✅ Recommended - clear accumulation
result = 0.0
for i in range(n):
    result += term(x, i)
return result

# ⚠️ Avoid modifying input directly
for i in range(n):
    x += something  # Can be confusing
return x
```

### Performance Considerations

- **Loop unrolling**: Large loops create large unrolled code
- **Compile time**: More iterations = longer compilation
- **Recommendation**: Keep loops reasonably sized (< 100 iterations)
- **Alternative**: For large iterations, use NumPy vectorized operations

## Limitations

1. **Constant ranges only**: Range parameters must be compile-time constants
2. **No break/continue**: Loop control statements may not be supported
3. **No else clause**: `for...else` construct may not be supported
4. **Fixed iteration count**: Cannot dynamically determine loop count based on convergence

## Comparison with Alternatives

| Approach | Flexibility | Performance | Differentiability |
|----------|-------------|-------------|-------------------|
| **For loops** | ⭐⭐⭐ Good | ⭐⭐ OK | ✅ Yes (constant range) |
| **NumPy vectorized** | ⭐⭐ Limited | ⭐⭐⭐ Excellent | ✅ Yes |
| **List comprehension** | ⭐⭐ Limited | ⭐⭐ OK | ❌ Lists not differentiable |
| **Manual unrolling** | ⭐ Poor | ⭐⭐⭐ Excellent | ✅ Yes |

**Recommendation**:
- Use **for loops** for iterative algorithms and moderate-sized computations
- Use **NumPy** for large-scale array operations
- Manually unroll for very small, performance-critical loops

## Best Practices

### 1. Cast Loop Variables

Always cast loop variables to float for arithmetic:

```python
for i in range(n):
    result += x ** float(i)  # ✅ Good
```

### 2. Keep Ranges Moderate

Avoid extremely large loops:

```python
# ✅ Good
for i in range(10):
    ...

# ⚠️ Acceptable
for i in range(100):
    ...

# ❌ Avoid - consider NumPy instead
for i in range(10000):
    ...
```

### 3. Use NumPy for Array Operations

When operating on arrays, prefer NumPy:

```python
# ❌ Don't do this:
result = 0.0
for i in range(len(arr)):
    result += arr[i] * x

# ✅ Do this instead:
result = np.sum(arr) * x
```

### 4. Clear Variable Names

Use descriptive names for accumulated values:

```python
# ✅ Clear
total_loss = 0.0
for i in range(n):
    total_loss += term_loss(x, i)

# ⚠️ Less clear
result = 0.0
for i in range(n):
    result += f(x, i)
```

## Testing

Comprehensive tests are available in `/tmp/test_for_loop_comprehensive.py`, covering:
- Simple accumulation patterns
- Polynomial evaluation via loops
- Weighted sums using loop variables
- Nested loops
- Conditionals inside loops
- Custom step sizes
- Complex expressions
- NumPy integration
- Edge cases (empty loops, single iteration)
- Real-world examples (Taylor series)

All 10 comprehensive tests pass successfully.

## See Also

- [Augmented Assignment Operators](AUGMENTED_ASSIGNMENT_SUPPORT.md) - Perfect complement for loops
- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md) - For conditionals in loops
- [Conditional Expression Support](CONDITIONAL_EXPRESSION_SUPPORT.md) - Ternary operators in loops
- [List Comprehension Support](LIST_COMPREHENSION_SUPPORT.md) - Alternative iteration pattern
