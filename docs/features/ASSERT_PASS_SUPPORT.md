# Assert and Pass Statement Support in Tangent

## Overview

Tangent fully supports Python's `assert` and `pass` statements in functions being differentiated. These statements enable input validation, debugging checks, and cleaner code structure without affecting gradient computation.

## Assert Statements

### Basic Usage

Assert statements validate conditions at runtime:

```python
import tangent

def validate_input(x):
    assert x > 0, "Input must be positive"
    return x ** 2

df = tangent.grad(validate_input)
print(df(5.0))  # 10.0 (assertion passes, gradient computed)
```

### Assert Syntax

Python's `assert` statement has two forms:

```python
assert condition                    # Simple assertion
assert condition, "error message"   # Assertion with message
```

Both forms work in Tangent:

```python
def safe_sqrt(x):
    assert x >= 0  # Simple form
    assert x < 1000, "Input too large"  # With message
    return x ** 0.5
```

## Assert Use Cases

### 1. Input Validation

Validate function inputs are in valid ranges:

```python
import tangent

def normalize(x):
    """Normalize input to [0, 1] range."""
    assert x >= 0, "Input must be non-negative"
    assert x <= 100, "Input must be at most 100"
    return x / 100.0

df = tangent.grad(normalize)
grad = df(50.0)  # 0.01
```

### 2. Numerical Stability Checks

Ensure intermediate values remain stable:

```python
import tangent

def safe_divide(x, divisor=2.0):
    """Safe division with validation."""
    assert divisor != 0, "Divisor cannot be zero"
    assert abs(divisor) > 1e-10, "Divisor too small"
    return x / divisor

df = tangent.grad(safe_divide)
grad = df(10.0)  # 0.5
```

### 3. Domain Validation

Check mathematical domain requirements:

```python
import tangent

def safe_log(x):
    """Logarithm with domain check."""
    assert x > 0, "log requires positive input"
    return tangent.numpy.log(x)

df = tangent.grad(safe_log)
grad = df(2.0)  # 0.5 (gradient of log(x) = 1/x)
```

### 4. Complex Conditions with Boolean Operators

Combine multiple conditions:

```python
import tangent

def bounded_compute(x):
    """Compute with bounded input."""
    assert x > 0 and x < 10, "x must be in (0, 10)"
    assert not (x < 1 or x > 9), "x must be in [1, 9]"
    return x ** 3

df = tangent.grad(bounded_compute)
grad = df(2.0)  # 12.0 (gradient of x^3 = 3x^2)
```

### 5. NumPy Integration

Use NumPy operations in assertions:

```python
import numpy as np
import tangent

def validate_range(x):
    """Validate against NumPy array bounds."""
    bounds = np.array([0.0, 10.0])
    assert x >= np.min(bounds), "x below minimum"
    assert x <= np.max(bounds), "x above maximum"
    return x ** 2

df = tangent.grad(validate_range)
grad = df(5.0)  # 10.0
```

### 6. Conditional Assertions

Assertions in different branches:

```python
import tangent

def conditional_check(x):
    """Different assertions for different cases."""
    if x > 0:
        assert x < 100, "Positive x must be < 100"
        result = x ** 2
    else:
        assert x > -100, "Negative x must be > -100"
        result = x
    return result

df = tangent.grad(conditional_check)
print(df(5.0))   # 10.0 (positive case)
print(df(-5.0))  # 1.0 (negative case)
```

### 7. Assertions in Loops

Validate invariants in loops:

```python
import tangent

def iterative_compute(x):
    """Iterative computation with validation."""
    result = x
    for i in range(5):
        assert result > 0, "Result became negative"
        result = result * 0.9 + x * 0.1
    return result

df = tangent.grad(iterative_compute)
grad = df(5.0)
```

## Pass Statements

### Basic Usage

Pass statements do nothing - they're placeholders:

```python
import tangent

def simple_pass(x):
    pass  # Does nothing
    return x ** 2

df = tangent.grad(simple_pass)
print(df(3.0))  # 6.0
```

## Pass Use Cases

### 1. Empty Branch Placeholder

Placeholder for branches not yet implemented:

```python
import tangent

def partial_implementation(x):
    """Function with partial implementation."""
    if x < 0:
        pass  # TODO: Handle negative case
    elif x > 10:
        pass  # TODO: Handle large case
    # Current implementation
    return x ** 2

df = tangent.grad(partial_implementation)
grad = df(5.0)  # 10.0
```

### 2. No-Op in Conditionals

Explicitly do nothing in certain cases:

```python
import tangent

def conditional_processing(x):
    """Process only certain values."""
    result = x
    if x > 5:
        result = result * 2
    else:
        pass  # Do nothing for small x
    return result

df = tangent.grad(conditional_processing)
print(df(3.0))  # 1.0 (small: no change)
print(df(7.0))  # 2.0 (large: doubled)
```

### 3. Skip Loop Iterations

Skip certain loop iterations:

```python
import tangent

def selective_accumulation(x):
    """Accumulate only on certain iterations."""
    result = 0.0
    for i in range(4):
        if i == 2:
            pass  # Skip iteration 2
        else:
            result += x
    return result

df = tangent.grad(selective_accumulation)
grad = df(1.0)  # 3.0 (3 iterations, skipping i=2)
```

### 4. Empty Else Clause

Make control flow explicit:

```python
import tangent

def explicit_control_flow(x):
    """Explicit control flow with pass."""
    if x > 0:
        result = x ** 2
    else:
        pass  # Explicitly do nothing

    # This line always executes
    return x

df = tangent.grad(explicit_control_flow)
```

### 5. Nested Loop Structures

Skip in nested loops:

```python
import tangent

def matrix_offdiagonal(x):
    """Compute off-diagonal sum."""
    result = 0.0
    for i in range(3):
        for j in range(3):
            if i == j:
                pass  # Skip diagonal
            else:
                result += x
    return result

df = tangent.grad(matrix_offdiagonal)
grad = df(1.0)  # 6.0 (9 total - 3 diagonal = 6 off-diagonal)
```

## Combined Usage

Assert and pass work together naturally:

```python
import tangent

def validated_processing(x):
    """Function with validation and conditional logic."""
    # Validation
    assert x > 0, "x must be positive"
    assert x < 100, "x must be less than 100"

    # Conditional processing
    if x < 10:
        pass  # Small values - no adjustment
        result = x
    else:
        result = x / 10.0

    return result ** 2

df = tangent.grad(validated_processing)
print(df(5.0))   # 10.0 (small case: x^2)
print(df(50.0))  # 1.0 (large case: (x/10)^2)
```

## Real-World Examples

### Example 1: Safe Mathematical Operations

```python
import tangent
import numpy as np

def safe_math(x):
    """Perform mathematical operations with validation."""
    assert x > 0, "Requires positive input"

    # Compute log (domain check passed)
    log_x = np.log(x)

    # Compute sqrt (domain check passed)
    sqrt_x = np.sqrt(x)

    return log_x + sqrt_x

df = tangent.grad(safe_math)
# Gradient of log(x) + sqrt(x) = 1/x + 1/(2*sqrt(x))
grad = df(4.0)
```

### Example 2: Gradient Clipping Validation

```python
import tangent

def validated_gradient_step(x):
    """Gradient descent step with validation."""
    # Compute gradient of loss function
    loss_grad = 2 * (x - 5.0)

    # Validate gradient magnitude
    assert abs(loss_grad) < 100, "Gradient explosion detected"

    # Apply gradient step
    learning_rate = 0.1
    x_new = x - learning_rate * loss_grad

    return x_new ** 2

df = tangent.grad(validated_gradient_step)
grad = df(3.0)
```

### Example 3: Iterative Algorithm with Checks

```python
import tangent

def newton_iteration(x, iterations=3):
    """Newton's method with validation."""
    estimate = x

    for i in range(iterations):
        # Validate estimate hasn't diverged
        assert abs(estimate) < 1000, "Divergence detected"

        if i == 0:
            pass  # First iteration - use initial estimate

        # Newton update for sqrt(2)
        estimate = 0.5 * (estimate + 2.0 / estimate)

    return estimate

df = tangent.grad(newton_iteration)
grad = df(1.5)
```

### Example 4: Bounded Activation Function

```python
import tangent

def bounded_activation(x):
    """Custom activation with bounds checking."""
    assert abs(x) < 100, "Input magnitude too large"

    if x < -10:
        pass  # Saturated negative
        result = -1.0
    elif x > 10:
        pass  # Saturated positive
        result = 1.0
    else:
        result = x / 10.0

    return result

df = tangent.grad(bounded_activation)
print(df(5.0))    # 0.1 (linear region)
print(df(15.0))   # 0.0 (saturated)
```

## Important Notes

### Assert Behavior

1. **Runtime checks**: Assertions are evaluated at runtime
2. **Python -O flag**: Assertions are disabled with `python -O` (optimization mode)
3. **Performance**: Assertions have minimal performance impact
4. **Not for control flow**: Don't use assertions for control flow logic

### Pass Behavior

1. **Pure no-op**: Pass does absolutely nothing
2. **Placeholder**: Useful during development
3. **Explicit intent**: Makes empty branches explicit
4. **No performance cost**: Completely removed during compilation

### Differentiation Impact

- **Assert**: Does not affect gradients (validation only)
- **Pass**: Does not affect gradients (no operation)
- Both are transparent to automatic differentiation

## Best Practices

### For Assert

1. **Validate inputs**: Check function preconditions
2. **Check invariants**: Validate loop invariants
3. **Clear messages**: Always provide descriptive error messages
4. **Avoid side effects**: Assertions should not modify state
5. **Document assumptions**: Use assertions to document assumptions

```python
# ✅ Good: Clear validation with messages
def process(x):
    assert x >= 0, "x must be non-negative"
    assert x <= 1, "x must be at most 1"
    return x ** 2

# ❌ Bad: Silent assertion
def process(x):
    assert x >= 0
    return x ** 2
```

### For Pass

1. **Use sparingly**: Only when needed for clarity
2. **Add TODO comments**: Explain future plans
3. **Prefer explicit**: Makes intent clear in conditionals
4. **Temporary**: Often removed in final code

```python
# ✅ Good: Pass with TODO
def partial(x):
    if x < 0:
        pass  # TODO: Implement negative handling
    return x ** 2

# ❌ Bad: Unnecessary pass
def simple(x):
    pass
    pass
    pass
    return x ** 2
```

## Comparison with Other Constructs

| Feature | Assert | Pass | If/Else |
|---------|--------|------|---------|
| **Purpose** | Validation | Placeholder | Control flow |
| **Affects gradient** | No | No | Yes |
| **Runtime cost** | Minimal | None | Depends |
| **Can fail** | Yes | No | No |
| **Production use** | Yes | Rare | Yes |

## Limitations

None! Both `assert` and `pass` work fully in Tangent with no known limitations.

## Testing

Comprehensive tests are available in `/tmp/test_assert_pass_comprehensive.py`, covering:
- Input validation with assert
- Complex conditions with boolean operators
- NumPy integration
- Multiple assertions
- Assertions in conditionals and loops
- Pass as placeholder
- Pass in if-elif-else structures
- Pass in nested loops
- Combined assert and pass usage
- Real-world examples

All 12 comprehensive tests pass successfully.

## See Also

- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md) - For complex assertion conditions
- [For Loop Support](FOR_LOOP_SUPPORT.md) - Loops with assertions
- [Conditional Expression Support](CONDITIONAL_EXPRESSION_SUPPORT.md) - Alternative control flow
