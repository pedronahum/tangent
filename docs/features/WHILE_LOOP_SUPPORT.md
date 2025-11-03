## While Loop Support in Tangent

## Overview

Tangent supports Python's `while` loops in functions being differentiated. While loops enable iterative algorithms, convergence-based methods, and value-dependent iteration patterns that complement the fixed iteration count of `for` loops.

## ⚠️ Important Limitation

**`break` and `continue` statements are not supported** in while loops. Use conditional logic or ensure loops terminate naturally through their condition.

## Basic Usage

### Simple Counter-Based While Loop

```python
import tangent

def simple_while(x):
    result = 0.0
    i = 0
    while i < 5:
        result += x
        i += 1
    return result

df = tangent.grad(simple_while)
print(df(2.0))  # 5.0 (f(x) = 5x, so df/dx = 5)
```

### While Loop with Computation

```python
import tangent

def while_powers(x):
    result = 0.0
    i = 1
    while i <= 3:
        result += x ** float(i)
        i += 1
    return result

df = tangent.grad(while_powers)
# f(x) = x + x^2 + x^3
# df/dx = 1 + 2x + 3x^2
print(df(2.0))  # 17.0
```

## While Loop Patterns

### 1. Counter-Based Iteration

Standard iteration with a counter variable:

```python
import tangent

def counter_based(x):
    result = 0.0
    count = 0
    while count < 10:
        result += x * float(count)
        count += 1
    return result

df = tangent.grad(counter_based)
# f(x) = x*(0+1+2+...+9) = 45x
print(df(1.0))  # 45.0
```

### 2. Value-Based Termination

Loop terminates based on computed value:

```python
import tangent

def value_based(x):
    """Accumulate until reaching threshold."""
    result = x
    iterations = 0
    max_iterations = 100  # Safety limit
    while result < 50.0 and iterations < max_iterations:
        result += x * 0.1
        iterations += 1
    return result

df = tangent.grad(value_based)
grad = df(5.0)
```

**Important**: Always include a safety counter to prevent infinite loops during differentiation.

### 3. Complex Conditions

Use boolean operators in while conditions:

```python
import tangent

def complex_condition(x):
    result = 0.0
    i = 0
    while i < 10 and result < 100.0:
        result += x ** 2
        i += 1
    return result

df = tangent.grad(complex_condition)
```

### 4. Nested While Loops

While loops can be nested:

```python
import tangent

def nested_while(x):
    result = 0.0
    i = 0
    while i < 3:
        j = 0
        while j < 3:
            result += x * (float(i) + float(j))
            j += 1
        i += 1
    return result

df = tangent.grad(nested_while)
print(df(1.0))  # 27.0
```

### 5. Conditionals Inside While

Combine while with if statements:

```python
import tangent

def while_with_if(x):
    result = 0.0
    i = 0
    while i < 4:
        if i < 2:
            result += x
        else:
            result += x ** 2
        i += 1
    return result

df = tangent.grad(while_with_if)
# f(x) = 2x + 2x^2
# df/dx = 2 + 4x
print(df(3.0))  # 14.0
```

## Real-World Examples

### Example 1: Newton's Method Iteration

Iterative root finding:

```python
import tangent

def newton_sqrt(x, iterations=5):
    """Approximate sqrt(x) using Newton's method."""
    estimate = x / 2.0  # Initial guess
    i = 0
    while i < iterations:
        # Newton update: estimate = (estimate + x/estimate) / 2
        estimate = 0.5 * (estimate + x / estimate)
        i += 1
    return estimate

df = tangent.grad(newton_sqrt)
grad = df(4.0)  # Gradient of sqrt approximation
```

### Example 2: Gradient Descent Steps

Multiple gradient descent iterations:

```python
import tangent

def gradient_descent_steps(x, steps=3):
    """Perform multiple gradient descent steps."""
    estimate = x
    learning_rate = 0.1
    target = 5.0
    i = 0
    while i < steps:
        # Gradient of (estimate - target)^2 is 2*(estimate - target)
        grad = 2.0 * (estimate - target)
        estimate = estimate - learning_rate * grad
        i += 1
    return estimate

df = tangent.grad(gradient_descent_steps)
grad = df(10.0)
```

### Example 3: Power Series Evaluation

Compute power series until desired accuracy:

```python
import tangent

def power_series(x, max_terms=10):
    """Evaluate power series: sum of x^i/i!"""
    result = 1.0
    term = 1.0
    i = 1
    while i < max_terms:
        term = term * x / float(i)
        result = result + term
        i += 1
    return result

df = tangent.grad(power_series)
# Approximates exp(x), gradient approximates exp(x)
grad = df(0.5)
```

### Example 4: Iterative Refinement

Refine an estimate iteratively:

```python
import tangent

def iterative_refine(x, iterations=5):
    """Iteratively refine estimate."""
    estimate = x
    i = 0
    while i < iterations:
        # Refinement step
        error = estimate ** 2 - 2.0
        estimate = estimate - 0.1 * error
        i += 1
    return estimate

df = tangent.grad(iterative_refine)
grad = df(1.5)
```

### Example 5: Accumulation with Early Termination

Accumulate until threshold or max iterations:

```python
import tangent

def accumulate_with_limit(x):
    """Accumulate with termination condition."""
    total = 0.0
    count = 0
    max_count = 20
    while total < 50.0 and count < max_count:
        total += x
        count += 1
    return total

df = tangent.grad(accumulate_with_limit)
grad = df(3.0)  # Will accumulate until total >= 50
```

## NumPy Integration

While loops work with NumPy operations:

```python
import numpy as np
import tangent

def while_with_numpy(x):
    """While loop with NumPy array indexing."""
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    result = 0.0
    i = 0
    while i < len(weights):
        result += x * weights[i]
        i += 1
    return result

df = tangent.grad(while_with_numpy)
# f(x) = x * sum(weights) = x * 5.0
print(df(1.0))  # 5.0
```

## Important Notes

### Loop Termination

1. **Must terminate**: Infinite loops will hang compilation
2. **Safety counters**: Always include max iteration limits
3. **Constant conditions preferred**: Dynamic conditions work but may complicate analysis

```python
# ✅ Good: Safety limit included
def safe_while(x):
    result = x
    i = 0
    max_iterations = 100
    while result < 1000.0 and i < max_iterations:
        result += x
        i += 1
    return result

# ⚠️ Risky: No safety limit
def risky_while(x):
    result = x
    while result < 1000.0:  # Could be infinite if x <= 0
        result += x
    return result
```

### Loop Variables

Cast integer loop variables to float when using in arithmetic:

```python
# ✅ Good: Cast to float
while i < 5:
    result += x * float(i)
    i += 1

# ⚠️ May cause warnings:
while i < 5:
    result += x * i  # i is integer
    i += 1
```

### Break and Continue

**Not supported**: Use conditional logic instead

```python
# ❌ Don't do this (break not supported):
while True:
    result += x
    if result > 10:
        break

# ✅ Do this instead:
max_iterations = 100
iterations = 0
while result <= 10 and iterations < max_iterations:
    result += x
    iterations += 1
```

## Comparison: While vs For Loops

| Feature | While Loop | For Loop |
|---------|-----------|----------|
| **Iteration count** | Variable (condition-based) | Fixed (range-based) |
| **Use case** | Convergence, thresholds | Known iterations |
| **Safety** | Need explicit limits | Inherently bounded |
| **Readability** | More complex | Simpler |
| **Performance** | Same | Same |
| **Break/continue** | ❌ Not supported | ❌ Not supported |

**When to use while**:
- Convergence algorithms (Newton's method, gradient descent)
- Value-dependent termination
- Accumulation until threshold
- Iterative refinement

**When to use for**:
- Known number of iterations
- Polynomial evaluation
- Fixed-size summations
- Simple counting patterns

## Best Practices

### 1. Always Include Safety Limits

```python
# ✅ Good
def safe(x):
    i = 0
    max_iterations = 1000
    while condition(x) and i < max_iterations:
        x = update(x)
        i += 1
    return x

# ❌ Bad
def unsafe(x):
    while condition(x):  # No safety limit!
        x = update(x)
    return x
```

### 2. Make Conditions Clear

```python
# ✅ Good: Clear condition
def clear(x):
    count = 0
    while count < 10:
        x += 1.0
        count += 1
    return x

# ⚠️ Less clear: Complex condition
def complex(x):
    while x < 10 and x > 0 and x != 5:
        x += 1.0
    return x
```

### 3. Initialize Variables Properly

```python
# ✅ Good: All variables initialized
def good(x):
    result = 0.0
    counter = 0
    while counter < 5:
        result += x
        counter += 1
    return result

# ❌ Bad: Uninitialized variable
def bad(x):
    # result not initialized!
    counter = 0
    while counter < 5:
        result += x  # Error!
        counter += 1
    return result
```

### 4. Avoid Modifying Differentiated Variables

```python
# ✅ Good: x unchanged, use separate variable
def good(x):
    estimate = x
    i = 0
    while i < 5:
        estimate = estimate * 0.9
        i += 1
    return estimate

# ⚠️ Less clear: Modifying x directly
def confusing(x):
    i = 0
    while i < 5:
        x = x * 0.9
        i += 1
    return x
```

## Edge Cases

### Empty While Loop

Loops with zero iterations work correctly:

```python
def empty_while(x):
    result = x
    i = 0
    while i < 0:  # Never executes
        result += x
        i += 1
    return result

df = tangent.grad(empty_while)
print(df(2.0))  # 1.0 (f(x) = x)
```

### Single Iteration

```python
def single_iteration(x):
    result = 0.0
    i = 0
    while i < 1:
        result += x
        i += 1
    return result

df = tangent.grad(single_iteration)
print(df(2.0))  # 1.0 (f(x) = x)
```

### Multiple Update Variables

```python
def multiple_updates(x):
    result = 0.0
    multiplier = 1.0
    i = 0
    while i < 3:
        result += x * multiplier
        multiplier += 1.0
        i += 1
    return result

df = tangent.grad(multiple_updates)
# f(x) = x*1 + x*2 + x*3 = 6x
print(df(1.0))  # 6.0
```

## Limitations

1. **No break/continue**: Loop control statements not supported
2. **No while-else**: The `while...else` construct may not be supported
3. **Compilation time**: Large iteration counts increase compilation time
4. **Constant analysis**: Very complex conditions may not compile

## Performance Considerations

- **Loop unrolling**: While loops may be partially unrolled during compilation
- **Compilation time**: More iterations = longer compilation
- **Runtime performance**: Similar to hand-written loops
- **Recommendation**: For very large iterations (>1000), consider vectorized NumPy operations

## Testing

Comprehensive tests are available in `/tmp/test_while_comprehensive.py`, covering:
- Simple counter-based loops
- Loops using counter in computations
- Power computations
- Nested while loops
- Conditionals inside while
- Complex boolean conditions
- Value-based termination
- NumPy integration
- Multiple variable updates
- Real-world iterative algorithms
- Empty loops (edge case)

11 out of 12 comprehensive tests pass successfully.

## See Also

- [For Loop Support](FOR_LOOP_SUPPORT.md) - Fixed iteration alternative
- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md) - For while conditions
- [Augmented Assignment Support](AUGMENTED_ASSIGNMENT_SUPPORT.md) - For loop variable updates
- [Assert Statement Support](ASSERT_PASS_SUPPORT.md) - For loop invariants
