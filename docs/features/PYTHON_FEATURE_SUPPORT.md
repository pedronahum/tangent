# Python Feature Support in Tangent

## Overview

This document provides a comprehensive reference of Python language features and their support status in Tangent's automatic differentiation system.

## ✅ Fully Supported Features

### Control Flow
- **✅ If/elif/else statements** - Full support with differentiable branches
- **✅ Conditional expressions (ternary)** - `a if condition else b` (reverse and forward mode)
- **✅ For loops with range()** - Fixed iteration loops with constant ranges
- **✅ For loops with enumerate()** - `for i, v in enumerate(seq)` (desugared to an indexed loop)
- **✅ For loops with zip()** - `for a, b in zip(xs, ys)` (desugared to an indexed loop)
- **✅ While loops** - Variable iteration with conditions (no break/continue)

### Operators
- **✅ Boolean operators** - `and`, `or`, `not` with short-circuit evaluation
- **✅ Comparison operators** - `>`, `<`, `>=`, `<=`, `==`, `!=`
- **✅ Membership operators** - `in`, `not in` (as non-differentiable branch guards)
- **✅ Identity operators** - `is`, `is not`
- **✅ Arithmetic operators** - `+`, `-`, `*`, `/`, `**`, `//`, `%`
- **✅ Augmented assignments** - `+=`, `-=`, `*=`, `/=`, `**=`, `//=`, `%=` (on variables; not on subscripts/attributes such as `a[i] += x`)

### Functions
- **✅ Lambda functions** - Anonymous functions with full inlining
- **✅ Closures** - Functions capturing variables from outer scope
- **✅ Nested functions** - Functions defined within functions
- **✅ Default arguments** - Function parameters with default values
- **✅ Keyword arguments** - Named function arguments
- **✅ Built-in `abs`, `min`, `max`** - `min`/`max` in two-argument form, e.g. `max(x, 0.0)` (ReLU) — reverse mode

### Data Structures (Read-Only)
- **✅ Dictionaries (read-only)** - Dict access, methods, nested dicts
- **✅ Lists (syntax)** - List operations in non-differentiated paths
- **✅ Tuples** - Tuple access and unpacking fully supported
- **✅ NumPy arrays** - Full support with comprehensive gradients

### Comprehensions (Partial)
- **✅ List comprehensions** - Syntactic support (lists not differentiable)
- **✅ Dict comprehensions** - Over a constant `range(...)`/list/tuple (unrolled into a dict literal)
- **✅ Set comprehensions** - Over a constant `range(...)`/list/tuple (unrolled into a set literal)
- **❌ Generator expressions** - Not supported
- **Note**: Comprehensions with dynamic iterables or `if` filters are rejected with a clear error

### Statements
- **✅ Assert statements** - Input validation and runtime checks
- **✅ Pass statements** - No-op placeholders
- **✅ Return statements** - Including early returns in if/elif/else branches (lifted to single-exit form; returns inside loops are rejected)
- **✅ Assignment statements** - Variable binding

### Other Features
- **✅ Chained assignment** - `a = b = x` (desugared to `a = x; b = a`)
- **✅ NumPy slicing** - Array indexing and slicing
- **✅ Ellipsis indexing** - `arr[..., 0]` for NumPy arrays
- **✅ F-strings** - Non-differentiable debug/assert messages, e.g. `f"x = {x}"`
- **✅ Set literals** - Non-differentiable collections, e.g. `if x in {1, 2, 3}`
- **✅ Higher-order derivatives** - `grad(grad(f))` (see caveat for the
  low-level tape API)

## ⚠️ Partially Supported Features

### Tuples
- **✅ Tuple access** - Reading tuple elements works
- **✅ Tuple indexing** - `t[0]`, `t[1]` works
- **✅ Tuple unpacking** - Works correctly in assignments (e.g., `a, b = x**2, x*3`)
- **✅ Tuple returns (Multi-Output)** - Full support with `output_index` and `output_weights` parameters
- **✅ Individual output gradients** - Use `output_index` to differentiate specific outputs
- **✅ Weighted output gradients** - Use `output_weights` for custom linear combinations

### Dictionaries
- **✅ Dict access (read-only)** - `config['key']` works perfectly
- **✅ Dict construction (string keys)** - Single- and multi-key dicts with string keys work
- **✅ Local dicts in forward mode** - Construction and subscript (parameter dicts: reverse mode only)
- **✅ Dict `.get()`** - `d.get(k)` and `d.get(k, default)`
- **✅ `sum(d.values())`** - Folded over local dict literals (keys known statically)
- **❌ Dict methods** - `.keys()`, `.items()`, and general `.values()` iteration not supported
- **✅ Nested dicts (parameters)** - Multi-level access works when dict is passed as parameter
- **✅ Dict comprehensions** - Over a constant range/list/tuple (unrolled)
- **❌ dict() constructor** - Not supported

### Loops
- **✅ For loops** - `range(...)`, NumPy arrays, and constant collections
- **✅ While loops** - With termination conditions
- **❌ break/continue** - Loop control statements not supported
- **❌ Iterating differentiated collections** - `for v in [x, y]`, `for v in d.values()`, etc. are rejected (the loop variable is not differentiated); use `range(len(...))` with indexing, a NumPy array, or `sum(d.values())`
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

### String Features
- **❌ String interpolation** - % formatting, .format() in limited contexts

### Data Structures
- **❌ Set operations** - Union/intersection/etc. not supported (literals as membership guards are supported)
- **❌ Generator expressions** - Not supported
- **⚠️ Set/dict comprehensions** - Only over constant ranges/literals (dynamic iterables and `if` filters rejected)

### Advanced Features
- **❌ Generators** - Generator functions and expressions not supported
- **❌ Decorators** - Function decorators not supported (except @tangent.grad)
- **❌ Classes** - Class definitions not supported
- **❌ Async/await** - Asynchronous programming not supported
- **❌ Type hints** - Annotations ignored (don't cause errors)

## Detailed Feature Documentation

### Dictionaries (Limited Support)

**Status**: ✅ Construction and read access supported (string keys); `.get()`, `sum(.values())`, and constant-range comprehensions supported; other methods not supported

**What Works:**
- ✅ Dicts passed as function parameters
- ✅ Dicts defined as global variables
- ✅ Subscript access `dict['key']` on parameter/global dicts
- ✅ Nested dicts (when passed as parameters)
- ✅ Local dict construction with string keys (single- and multi-key)
- ✅ `.get(key)` and `.get(key, default)`
- ✅ `sum(d.values())` over local dict literals

**What Doesn't Work:**
- ❌ `.keys()`, `.items()`, and general `.values()` iteration (only `sum(d.values())` is folded)
- ❌ Dict comprehensions over dynamic iterables or with `if` filters (constant ranges are unrolled)
- ❌ `dict()` constructor
- ❌ Modifying dict values (empty dict + assignments)

```python
import tangent

# ✅ RECOMMENDED: Dict as parameter
def compute(x, config={'lr': 0.1, 'momentum': 0.9}):
    return x * config['lr'] + x * config['momentum']

df = tangent.grad(compute)
grad = df(5.0)  # Works perfectly!

# ✅ RECOMMENDED: Global dict
PARAMS = {'scale': 2.0, 'offset': 1.0}

def process(x):
    return x * PARAMS['scale'] + PARAMS['offset']

df = tangent.grad(process)
grad = df(3.0)  # Works!

# ✅ Works: Single-key dict
def single_key(x):
    d = {'a': x}  # OK - single key
    return d['a']

df = tangent.grad(single_key)
grad = df(2.0)  # Works!

# ✅ Works: Multi-key dict with differentiated values (string keys)
def multi_key(x):
    d = {'a': x, 'b': x ** 2}
    return d['a'] + d['b']

df = tangent.grad(multi_key)
grad = df(2.0)  # = 5.0, works!

# ✅ Works: .get() with or without a default (reverse mode)
def dict_get(x):
    d = {'a': x}
    return d.get('a', 0.0)  # desugars to d['a'] if 'a' in d else 0.0

df = tangent.grad(dict_get)
grad = df(2.0)  # = 1.0, works!

# ✅ Works: sum(d.values()) over a local dict literal
def dict_values_sum(x):
    d = {'a': x, 'b': x ** 2}
    return sum(d.values())  # desugars to d['a'] + d['b']

df = tangent.grad(dict_values_sum)
grad = df(2.0)  # = 5.0, works!

# ❌ BROKEN: iteration over keys/items
def dict_methods(x):
    d = {'a': x, 'b': x ** 2}
    total = 0.0
    for k in d.keys():        # ERROR: .keys() iteration not supported
        total = total + d[k]
    return total
```

**Best Practices:**
1. **Use string keys** when constructing dicts locally
2. **Pass dicts as parameters** or use global dicts for configuration that
   doesn't depend on inputs
3. **Prefer `d['key']`, `d.get('key')`, or `sum(d.values())`** over `.keys()`/`.items()` iteration

**Note:** Local dict construction is fully supported, with string or numeric
keys. A prior bug produced undefined `_` placeholders / a DictConstructionError
whenever a *local* dict variable was named `d` (which collided with Tangent's
internal `d[x]` gradient-operator sentinel). A local variable named `d` is now
alpha-renamed before differentiation, so any key type works regardless of the
variable's name.

### Tuple Returns (Multi-Output Functions)

**Status**: ✅ Fully supported with `output_index` and `output_weights` parameters

Tangent now has **full support for multi-output functions**! You can:
1. Get the gradient of a specific output
2. Get a weighted combination of output gradients
3. Use the default (sum of all outputs) for backward compatibility

#### Option 1: Gradient of Specific Output (NEW!)

```python
import tangent

def f(x):
    return x ** 2, x * 3  # Returns (output1, output2)

# Gradient of FIRST output only
df_first = tangent.grad(f, output_index=0)
grad1 = df_first(2.0)  # d/dx(x^2) = 2x = 4.0

# Gradient of SECOND output only
df_second = tangent.grad(f, output_index=1)
grad2 = df_second(2.0)  # d/dx(3x) = 3.0
```

#### Option 2: Weighted Combination (NEW!)

```python
# Custom weighting of outputs
df_weighted = tangent.grad(f, output_weights=(0.7, 0.3))
result = df_weighted(2.0)
# Computes: d/dx(0.7*x^2 + 0.3*3x) = 0.7*2x + 0.3*3 = 1.4x + 0.9 = 3.7
```

#### Option 3: Default (Sum of All Outputs)

```python
# Default: sum all outputs (backward compatible)
df_sum = tangent.grad(f)
result = df_sum(2.0)  # d/dx(x^2 + 3x) = 2x + 3 = 7.0
```

**Comparison:**

```python
# Tuple return (auto-summed)
def f_tuple(x):
    return x ** 2, x * 3

df_tuple = tangent.grad(f_tuple)
grad_tuple = df_tuple(2.0)  # = 7.0 (sum of gradients)

# Explicit sum (same result)
def f_sum(x):
    return x ** 2 + x * 3

df_sum = tangent.grad(f_sum)
grad_sum = df_sum(2.0)  # = 7.0 (identical)

assert grad_tuple == grad_sum  # True!
```

**When is this useful?**
- Machine learning: `total_loss = prediction_loss + regularization_loss`
- Multi-objective optimization where you want combined gradient
- Physics simulations with multiple energy terms

**See also**:
- `tests/test_multi_output_grad.py` - Multi-output gradient examples with `output_index` and `output_weights`
- `tests/test_tuple_return_behavior.py` - Comprehensive tuple return behavior examples

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

**Status**: ❌ Not supported (rejected with a clear error)

Break and continue statements are rejected at parse time with an actionable
error. They cannot be differentiated correctly: the reverse-mode loop tape
records one entry per completed iteration, but `break`/`continue` alter the
control flow mid-iteration, which would silently produce incorrect gradients.
Tangent therefore refuses them up front rather than returning a wrong result.

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

### Higher-Order Differentiation (Second Derivatives)

**Status**: ✅ Supported — differentiate a gradient function again

Tangent can differentiate a generated gradient function, so second derivatives
(and Hessian-vector products) work by nesting `grad`:

```python
import tangent

def f(x):
    return x ** 3

ddf = tangent.grad(tangent.grad(f))
ddf(2.0)  # 12.0  (d²/dx² x³ = 6x)
```

This works in both optimized and unoptimized modes, and for array-valued
functions (e.g. `sum(tanh(x))` yields the elementwise `tanh''`, the Hessian
diagonal).

**The low-level tape API under higher-order AD:**

Functions that call Tangent's internal tape API directly — `tangent.push`,
`tangent.pop`, `tangent.push_stack`, `tangent.pop_stack`, `tangent.Stack` —
differentiate correctly at first order, and their second derivatives are
correct in **unoptimized** mode. With `optimized=True` the optimization
passes eliminate tape push/pop pairs and corrupt the second derivative:

```python
def uses_tape(a):
    _stack = tangent.Stack()
    b = a * a
    tangent.push(_stack, b, 'id')
    b = tangent.pop(_stack, 'id')
    return b            # b == a**2; d²/da² should be 2

tangent.grad(tangent.grad(uses_tape), optimized=False)(3.0)  # 2.0 (correct)
tangent.grad(tangent.grad(uses_tape), optimized=True)(3.0)   # wrong
```

These primitives are bookkeeping that ordinary code never calls — Tangent
inserts them into generated gradient code itself, and that generated code
differentiates correctly at second order with `optimized=True` (that is how
second derivatives of ordinary functions work). The limitation only affects
hand-written use of the tape API inside a function you intend to
differentiate twice with optimizations enabled. **Workaround:** pass
`optimized=False` to the outer `grad`, or express the computation with
ordinary Python/NumPy operations and let Tangent manage the tape.

## Best Practices

### 1. Use Supported Features When Possible

```python
# ✅ Good: Construct a dict with string keys and index it
def good(x):
    config = {'lr': 0.1}
    return x * config['lr']

# ✅ Good: .get() works too (with or without a default)
def good_get(x):
    config = {'lr': 0.1}
    return x * config.get('lr', 0.01)

# ✅ Good: sum(d.values()) over a local dict literal
def good_sum(x):
    terms = {'data': x ** 2, 'reg': 0.1 * x}
    return sum(terms.values())

# ❌ Bad: key/item iteration is not supported
def bad(x, config):
    total = 0.0
    for k, v in config.items():  # ERROR: .items() iteration not supported
        total = total + v
    return total
```

### 2. Define Complex Data Structures Outside

```python
# ✅ Good: Define large / static structures globally for clarity
CONFIG = {
    'model': {'layers': 3, 'units': 128},
    'training': {'lr': 0.01, 'epochs': 100}
}

def train_step(x):
    return x * CONFIG['training']['lr']

# ✅ Also works: Construct (even nested) dicts with string keys inside
def train_step(x):
    config = {'training': {'lr': x}}
    return config['training']['lr']
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

### 4. ~~Avoid Tuple Unpacking~~ ✅ FIXED - Tuple Unpacking Works!

**Update**: Tuple unpacking now works correctly in all tested scenarios.

```python
# ✅ Works correctly: Tuple unpacking
def compute(x):
    a, b = x ** 2, x * 3  # Gradients computed correctly!
    return a + b

# ✅ Also works: Multiple unpacking
def compute(x):
    a, b = x ** 2, x * 3
    c, d = a + 1, b * 2
    return c + d

# ✅ Even works: Unpacking from function calls
def helper(x):
    return x ** 2, x * 3

def compute(x):
    a, b = helper(x)  # Works correctly!
    return a + b
```

All tuple unpacking patterns have been tested and produce correct gradients.

## Comparison with Other Frameworks

| Feature | Tangent | JAX | PyTorch | TensorFlow |
|---------|---------|-----|---------|------------|
| **If/else** | ✅ | ✅ | ✅ | ✅ |
| **For loops** | ✅ (constant range) | ✅ | ✅ | ✅ |
| **While loops** | ✅ (no break) | ✅ | ⚠️ | ⚠️ |
| **Lambdas** | ✅ | ✅ | ✅ | ✅ |
| **Closures** | ✅ | ✅ | ✅ | ✅ |
| **Dict (read)** | ✅ | ✅ | ✅ | ✅ |
| **Dict (construct)** | ✅ (string keys) | ✅ | ✅ | ✅ |
| **Dict (mutate)** | ❌ | ✅ | ✅ | ✅ |
| **Tuples** | ✅ | ✅ | ✅ | ✅ |
| **Try/except** | ❌ | ⚠️ | ⚠️ | ⚠️ |
| **Break/continue** | ❌ | ⚠️ | ⚠️ | ⚠️ |

## Testing

Comprehensive tests available:
- `tests/test_dict_construction.py` - Local dict construction (string keys)
- `tests/test_membership_operators.py` - `in` / `not in` operators
- Individual feature test files for each supported feature

## Summary Statistics

- **Fully Supported**: 34+ features (including early returns, tuples, membership/identity operators, f-strings, dict `.get()`, `sum(d.values())`, set literals, and constant-range set/dict comprehensions!)
- **Partially Supported**: 1 feature (some loops)
- **Not Supported**: 10+ features
- **Overall Coverage**: ~62% of common Python features

## Recommendations

For maximum compatibility with Tangent:

1. **✅ DO**:
   - Use NumPy arrays for numerical data
   - Pass configuration dicts as parameters
   - Use assertions for validation
   - Use conditional statements for control flow
   - Define complex data structures outside functions

2. **❌ DON'T**:
   - Iterate dict `.keys()`/`.items()` (only `sum(d.values())` is supported)
   - Use try/except blocks
   - Use break/continue in loops
   - Use set operations (union/intersection); set literals and constant-range comprehensions are fine

3. **⚠️ BE CAREFUL**:
   - Set/dict comprehensions must range over a constant `range(...)`/list/tuple (no dynamic iterables or `if` filters)
   - Loop ranges must be compile-time constants
   - Early returns in if/elif/else are supported; returns inside loops are not

## See Also

- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md)
- [For Loop Support](FOR_LOOP_SUPPORT.md)
- [While Loop Support](WHILE_LOOP_SUPPORT.md)
- [Augmented Assignment Support](AUGMENTED_ASSIGNMENT_SUPPORT.md)
- [Assert and Pass Support](ASSERT_PASS_SUPPORT.md)
- [Lambda Support](LAMBDA_SUPPORT_COMPLETE.md)
- [Closure Support](CLOSURE_SUPPORT_COMPLETE.md)
