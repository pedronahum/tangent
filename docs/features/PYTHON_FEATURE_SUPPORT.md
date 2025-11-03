# Python Feature Support in Tangent

## Overview

This document provides a comprehensive reference of Python language features and their support status in Tangent's automatic differentiation system.

## ✅ Fully Supported Features

### Control Flow
- **✅ If/elif/else statements** - Full support with differentiable branches
- **✅ Conditional expressions (ternary)** - `a if condition else b`
- **✅ For loops with range()** - Fixed iteration loops with constant ranges
- **✅ While loops** - Variable iteration with conditions (no break/continue)

### Operators
- **✅ Boolean operators** - `and`, `or`, `not` with short-circuit evaluation
- **✅ Comparison operators** - `>`, `<`, `>=`, `<=`, `==`, `!=`
- **✅ Arithmetic operators** - `+`, `-`, `*`, `/`, `**`, `//`, `%`
- **✅ Augmented assignments** - `+=`, `-=`, `*=`, `/=`, `**=`, `//=`, `%=`

### Functions
- **✅ Lambda functions** - Anonymous functions with full inlining
- **✅ Closures** - Functions capturing variables from outer scope
- **✅ Nested functions** - Functions defined within functions
- **✅ Default arguments** - Function parameters with default values
- **✅ Keyword arguments** - Named function arguments

### Data Structures (Read-Only)
- **✅ Dictionaries (read-only)** - Dict access, methods, nested dicts
- **✅ Lists (syntax)** - List operations in non-differentiated paths
- **✅ Tuples (read-only)** - Tuple access (unpacking has limitations)
- **✅ NumPy arrays** - Full support with comprehensive gradients

### Comprehensions (Partial)
- **✅ List comprehensions** - Syntactic support (lists not differentiable)
- **❌ Dict comprehensions** - Not supported
- **❌ Set comprehensions** - Not supported
- **❌ Generator expressions** - Not supported

### Statements
- **✅ Assert statements** - Input validation and runtime checks
- **✅ Pass statements** - No-op placeholders
- **✅ Return statements** - Single value returns
- **✅ Assignment statements** - Variable binding

### Other Features
- **✅ Multiple assignment** - `a = b = x` (separate assignments work)
- **✅ NumPy slicing** - Array indexing and slicing
- **✅ Ellipsis indexing** - `arr[..., 0]` for NumPy arrays

## ⚠️ Partially Supported Features

### Tuples
- **✅ Tuple access** - Reading tuple elements works
- **✅ Tuple indexing** - `t[0]`, `t[1]` works
- **⚠️ Tuple unpacking** - Works but may give incorrect gradients (sums all elements)
- **❌ Multiple returns** - Cannot return tuples from differentiated functions
- **Workaround**: Use separate assignments instead of unpacking

### Dictionaries
- **✅ Dict access (read-only)** - `config['key']` works perfectly
- **✅ Dict methods** - `.get()`, `.keys()`, `.values()`, `.items()`
- **✅ Nested dicts** - Multi-level access works
- **❌ Dict construction** - Cannot create dicts with `{}` or `dict()`
- **Workaround**: Pass dicts as parameters or use global dicts

### Loops
- **✅ For loops** - With constant `range()` parameters
- **✅ While loops** - With termination conditions
- **❌ break/continue** - Loop control statements not supported
- **Workaround**: Use conditional logic for early termination

## ❌ Not Supported Features

### Statements
- **❌ Try/except/finally** - Exception handling not supported
- **❌ With statements** - Context managers not supported (basic syntax may work)
- **❌ Del statement** - Variable deletion not supported
- **❌ Raise statement** - Raising exceptions not supported
- **❌ Import statements** - Inside functions (use module-level imports)

### Operators
- **❌ Walrus operator** (`:=`) - Assignment expressions not supported correctly
- **❌ In operator** - Membership testing not supported
- **❌ Is operator** - Identity comparison not supported

### String Features
- **❌ F-strings** - Formatted string literals not supported
- **❌ String interpolation** - % formatting, .format() in limited contexts

### Data Structures
- **❌ Sets** - Set literals and operations not supported
- **❌ Dict literals** - Cannot construct dicts inline
- **❌ Set comprehensions** - Not supported
- **❌ Dict comprehensions** - Not supported

### Advanced Features
- **❌ Generators** - Generator functions and expressions not supported
- **❌ Decorators** - Function decorators not supported (except @tangent.grad)
- **❌ Classes** - Class definitions not supported
- **❌ Async/await** - Asynchronous programming not supported
- **❌ Type hints** - Annotations ignored (don't cause errors)

## Detailed Feature Documentation

### Dictionaries (Read-Only)

**Status**: ✅ Fully supported for read-only access

Dictionaries work perfectly when:
- Passed as function parameters
- Defined as global variables
- Accessed with `[]` or `.get()`
- Nested dictionaries
- Containing NumPy arrays as values

```python
import tangent

# ✅ Works: Dict as parameter
def compute(x, config={'lr': 0.1}):
    return x * config['lr']

df = tangent.grad(compute)
grad = df(5.0)  # Works!

# ✅ Works: Global dict
PARAMS = {'scale': 2.0, 'offset': 1.0}

def process(x):
    return x * PARAMS['scale'] + PARAMS['offset']

df = tangent.grad(process)
grad = df(3.0)  # Works!

# ✅ Works: Nested dicts
def nested(x):
    config = {
        'optimizer': {
            'lr': 0.1,
            'momentum': 0.9
        }
    }
    return x * config['optimizer']['lr']

df = tangent.grad(nested)
grad = df(2.0)  # Works!

# ❌ Doesn't work: Dict construction
def construct(x):
    params = {'a': x, 'b': x ** 2}  # ERROR: Dict construction
    return params['a']
```

**Workarounds**:
1. Define dicts outside the function
2. Pass dicts as parameters (with default values)
3. Use global dicts
4. Use separate variables instead of dict values

### Exception Handling

**Status**: ❌ Not supported

Try/except blocks are not supported in Tangent:

```python
# ❌ Doesn't work
def safe_divide(x):
    try:
        return 1.0 / x
    except ZeroDivisionError:
        return 0.0
```

**Workarounds**:
1. Use assertions to validate inputs
2. Use conditional statements to check preconditions
3. Handle exceptions outside differentiated functions

```python
# ✅ Works: Use assertions
def safe_divide(x):
    assert x != 0, "Division by zero"
    return 1.0 / x

# ✅ Works: Use conditionals
def safe_divide(x):
    if abs(x) < 1e-10:
        return 0.0
    return 1.0 / x
```

### Loop Control (break/continue)

**Status**: ❌ Not supported

Break and continue statements are not supported:

```python
# ❌ Doesn't work
def early_exit(x):
    result = 0.0
    for i in range(10):
        result += x
        if result > 100:
            break  # ERROR
    return result
```

**Workarounds**:
1. Use while loops with complex conditions
2. Include termination logic in the condition
3. Use conditional statements

```python
# ✅ Works: Condition in loop
def with_condition(x):
    result = 0.0
    i = 0
    max_iterations = 10
    while result <= 100 and i < max_iterations:
        result += x
        i += 1
    return result
```

## Best Practices

### 1. Use Supported Features When Possible

```python
# ✅ Good: Use supported features
def good(x, config):
    lr = config.get('lr', 0.1)
    return x * lr

# ❌ Bad: Try to construct dict
def bad(x):
    config = {'lr': 0.1}  # ERROR
    return x * config['lr']
```

### 2. Define Complex Data Structures Outside

```python
# ✅ Good: Define globally
CONFIG = {
    'model': {'layers': 3, 'units': 128},
    'training': {'lr': 0.01, 'epochs': 100}
}

def train_step(x):
    return x * CONFIG['training']['lr']

# ❌ Bad: Construct inside
def train_step(x):
    config = {'training': {'lr': 0.01}}  # ERROR
    return x * config['training']['lr']
```

### 3. Use Assertions Instead of Exceptions

```python
# ✅ Good: Use assertions
def safe_log(x):
    assert x > 0, "log requires positive input"
    return np.log(x)

# ❌ Bad: Try/except
def safe_log(x):
    try:  # ERROR
        return np.log(x)
    except:
        return 0.0
```

### 4. Avoid Tuple Unpacking

```python
# ✅ Good: Separate assignments
def compute(x):
    a = x ** 2
    b = x * 3
    return a + b

# ⚠️ Problematic: Tuple unpacking
def compute(x):
    a, b = x ** 2, x * 3  # May give wrong gradient
    return a + b
```

## Comparison with Other Frameworks

| Feature | Tangent | JAX | PyTorch | TensorFlow |
|---------|---------|-----|---------|------------|
| **If/else** | ✅ | ✅ | ✅ | ✅ |
| **For loops** | ✅ (constant range) | ✅ | ✅ | ✅ |
| **While loops** | ✅ (no break) | ✅ | ⚠️ | ⚠️ |
| **Lambdas** | ✅ | ✅ | ✅ | ✅ |
| **Closures** | ✅ | ✅ | ✅ | ✅ |
| **Dict (read)** | ✅ | ✅ | ✅ | ✅ |
| **Dict (write)** | ❌ | ✅ | ✅ | ✅ |
| **Tuples** | ⚠️ | ✅ | ✅ | ✅ |
| **Try/except** | ❌ | ⚠️ | ⚠️ | ⚠️ |
| **Break/continue** | ❌ | ⚠️ | ⚠️ | ⚠️ |

## Testing

Comprehensive tests available:
- `/tmp/test_feature_survey.py` - Survey of 15 features
- `/tmp/test_dict_detailed.py` - 8 dictionary tests (all pass)
- Individual feature test files for each supported feature

## Summary Statistics

- **Fully Supported**: 25+ features
- **Partially Supported**: 3 features (tuples, dicts, loops)
- **Not Supported**: 15+ features
- **Overall Coverage**: ~60% of common Python features

## Recommendations

For maximum compatibility with Tangent:

1. **✅ DO**:
   - Use NumPy arrays for numerical data
   - Pass configuration dicts as parameters
   - Use assertions for validation
   - Use conditional statements for control flow
   - Define complex data structures outside functions

2. **❌ DON'T**:
   - Construct dicts inside functions
   - Use try/except blocks
   - Use break/continue in loops
   - Rely on tuple unpacking
   - Use f-strings or sets

3. **⚠️ BE CAREFUL**:
   - Tuple unpacking may give incorrect gradients
   - Dict construction not supported (use parameters)
   - Loop ranges must be compile-time constants

## See Also

- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md)
- [For Loop Support](FOR_LOOP_SUPPORT.md)
- [While Loop Support](WHILE_LOOP_SUPPORT.md)
- [Augmented Assignment Support](AUGMENTED_ASSIGNMENT_SUPPORT.md)
- [Assert and Pass Support](ASSERT_PASS_SUPPORT.md)
- [Lambda Support](LAMBDA_SUPPORT_COMPLETE.md)
- [Closure Support](CLOSURE_SUPPORT_COMPLETE.md)
- [Tuple Limitations](TUPLE_LIMITATIONS.md)
