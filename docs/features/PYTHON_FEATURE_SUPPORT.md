# Python Feature Support in Tangent

## Overview

This document provides a comprehensive reference of Python language features and their support status in Tangent's automatic differentiation system.

## ✅ Fully Supported Features

### Control Flow
- **✅ If/elif/else statements** - Full support with differentiable branches
- **✅ Conditional expressions (ternary)** - `a if condition else b` (reverse and forward mode)
- **✅ For loops with range()** - Constant or runtime bounds (`range(len(x))` included)
- **✅ For loops over sequences** - `for v in xs` over arrays, lists (built dynamically included), list/tuple literals of active values, and computed expressions like `for v in x * 2.0` (the iterable is hoisted and indexed; tuple-unpacking targets like `for a, b in pairs` work)
- **✅ For loops with enumerate()** - `for i, v in enumerate(seq)` (desugared to an indexed loop)
- **✅ For loops with zip()** - `for a, b in zip(xs, ys)` (desugared to an indexed loop)
- **✅ While loops** - Variable iteration with conditions
- **✅ break/continue** - Lowered into guard flags (`while` folds the break flag into its condition; a `for` wraps its body), so gradients count exactly the executed iterations. Loops with an `else` clause remain rejected
- **✅ return inside loops** - Lowered into assign + flag + `break`, propagating the exit past each enclosing loop; combined with branch-return lifting, any mix of early returns differentiates. Only loops with an `else` clause reject returns

### Operators
- **✅ Boolean operators** - `and`, `or`, `not` with short-circuit evaluation
- **✅ Comparison operators** - `>`, `<`, `>=`, `<=`, `==`, `!=`
- **✅ Membership operators** - `in`, `not in` (as non-differentiable branch guards)
- **✅ Identity operators** - `is`, `is not`
- **✅ Arithmetic operators** - `+`, `-`, `*`, `/`, `**`, `//`, `%`
- **✅ Augmented assignments** - `+=`, `-=`, `*=`, `/=`, `**=`, `//=`, `%=` on variables and subscripts (e.g. `a[i] += x`, expanded to `a[i] = a[i] + x`); not on attributes (`obj.attr += x` is rejected)

### Functions
- **⚠️ Lambda functions** - Supported when assigned to a variable and called (`sq = lambda y: y*y; sq(x)`) — the lambda is inlined. An inline lambda *call* (`(lambda y: y*y)(x)`) is rejected
- **✅ Closures (external)** - A closure created *outside* the differentiated function (e.g. returned by a factory) is supported; its captured variables are made available to the gradient
- **❌ Nested function definitions** - A `def` written *inside* the body of the differentiated function is rejected with a clear error (hoist it to module level, or use an assigned lambda)
- **❌ Recursion** - Rejected with a clear error
- **✅ Default arguments** - Function parameters with default values
- **✅ Keyword arguments** - Named function arguments
- **❌ Variadic `*args` / `**kwargs`** - Rejected with a clear error
- **✅ Built-in `abs`, `min`, `max`** - `min`/`max` in two-argument form, e.g. `max(x, 0.0)` (ReLU) — reverse mode

### Data Structures (Read-Only)
- **✅ Dictionaries (read-only)** - Dict access, methods, nested dicts
- **✅ Lists** - Reading, and differentiable building with `xs.append(v)` / `v = xs.pop()` (desugared into rebindings through `tangent.list_append`/`list_last`/`list_init`; works in loops, forward mode and higher order). Other in-place mutators (`extend`/`insert`/`remove`/`sort`/`reverse`, or append through an attribute/subscript) are rejected with a clear error - they used to silently drop gradients
- **✅ Tuples** - Tuple access and unpacking fully supported
- **✅ NumPy arrays** - Full support with comprehensive gradients
- **✅ Pytree arguments and return values** - Tuples, lists and (nested) dicts
  of arrays as arguments (gradients come back in the same structure) and as
  return values: the default seed is a matching pytree of ones (the gradient of
  the sum of all leaves), a caller-supplied seed of the same structure is used
  as the cotangent, and a structurally mismatched seed raises a `ValueError`

### Comprehensions (Partial)
- **✅ List comprehensions** - Over a constant `range(...)`/list/tuple (unrolled into a list literal) **and over dynamic iterables** (lowered into an indexed loop built on the differentiable `tangent.list_append` rebinding), including runtime `if` filters and nested comprehensions. Single generator with a plain-name target only; multiple generators and tuple targets are rejected with a clear error
- **✅ Dict comprehensions** - Over a constant `range(...)`/list/tuple (unrolled into a dict literal)
- **✅ Set comprehensions** - Over a constant `range(...)`/list/tuple (unrolled into a set literal)
- **❌ Generator expressions** - Not supported
- **Note**: Set/dict comprehensions with `if` filters or dynamic iterables are rejected with a clear error

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
- **✅ For loops** - `range(...)`, sequences (arrays, lists, active literals), and computed iterable expressions (hoisted and indexed)
- **✅ While loops** - With termination conditions
- **✅ break/continue and return** - Lowered into guard flags; exact gradients through early exits
- **❌ Iterating dict views and set literals** - `for v in d.values()` and `for v in {x, y}` are rejected (no stable positional order); use `sum(d.values())` or a list/tuple
- **Workaround**: Use conditional logic for early termination

## ❌ Not Supported Features

### Statements
- **❌ Try/except/finally** - Exception handling not supported
- **❌ With statements** - Context managers not supported (basic syntax may work)
- **❌ Del statement** - Variable deletion not supported
- **❌ Raise statement** - Raising exceptions not supported
- **❌ Import statements** - Inside functions (use module-level imports)

### Operators
- **❌ Walrus operator** (`:=`) - Rejected with a clear error (the bound name is not tracked, so it used to silently return a zero gradient)

### String Features
- **❌ String interpolation** - % formatting, .format() in limited contexts

### Data Structures
- **❌ Set operations** - Union/intersection/etc. not supported (literals as membership guards are supported)
- **❌ Generator expressions** - Not supported
- **⚠️ Comprehensions** - List comprehensions over both constant and dynamic iterables (with filters); set/dict comprehensions only over constant ranges/literals and without `if` filters

### Advanced Features
- **❌ Generators** - Generator functions and expressions not supported
- **❌ Decorators** - Function decorators not supported (except @tangent.grad)
- **⚠️ Classes** - Module-level classes used inside a differentiated function
  are supported via method inlining (see [Classes and Inheritance](#classes-and-inheritance)
  below); a `class` definition *inside* the differentiated function is not
- **❌ Nested function definitions / recursion** - A `def` inside the differentiated function, and recursion, are rejected with a clear error (external closures are supported; see Functions)
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

### Classes and Inheritance

**Status**: ✅ Supported for module-level classes, via method inlining

Tangent differentiates functions that instantiate and use user-defined
classes. It does not add OOP support to the AD core; instead, a desugaring
pass resolves the class from the function's globals, tracks instance-variable
assignments, and inlines method bodies at their call sites (substituting
`self.attr` with the tracked constructor values) before differentiation.

```python
class Calculator:
    def __init__(self, scale):
        self.scale = scale

    def square(self, x):
        return x * x * self.scale

def f(x):
    calc = Calculator(2.0)
    return calc.square(x)

df = tangent.grad(f)
df(3.0)   # 12.0
```

**What works:**
- ✅ Instantiating a module-level class and calling its methods
- ✅ Constructor arguments and instance attributes (`self.attr` read in methods)
- ✅ Methods calling other methods of the same instance; chained method calls
- ✅ Multiple methods, multiple parameters, `wrt=` selection
- ✅ NumPy operations inside methods
- ✅ **Inheritance**: inherited methods are resolved through the MRO
  (including multi-level hierarchies and grandparent methods), method
  overriding, attribute inheritance through `super().__init__()`, and derived
  methods calling inherited helpers

**What doesn't work:**
- ❌ A `class` definition *inside* the differentiated function
- ❌ `@property`, `@classmethod`, `@staticmethod`
- ❌ Calling the parent implementation from an overridden method
  (`super().method(x)` or `Parent.method(self, x)` in a method body;
  `super().__init__()` in constructors is fine)
- ❌ Methods that mutate instance state; dynamic attribute access
  (`getattr(self, name)`)
- ⚠️ Multiple inheritance (diamond patterns) and abstract base classes are
  untested

See `tests/test_classes.py` and `tests/test_inheritance.py` for the full set
of supported patterns.

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

### Loop Control (break/continue/return)

**Status**: ✅ Supported (lowered into guard flags)

`break` and `continue` are desugared before differentiation
(`tangent/loop_exit_desugar.py`): `continue` becomes a per-iteration skip flag
guarding the rest of the body; `break` additionally sets a loop-level flag that
a `while` folds into its condition and a `for` uses to skip all remaining
iterations. A `return` inside a loop lowers into an assignment plus a returning
flag plus `break`, propagated past each enclosing loop. Every construct
produced is one the AD core differentiates in both modes, so the gradient
counts exactly the iterations that executed - including data-dependent exits
(`if total > 5.0: break`).

```python
# ✅ Works
def early_exit(x):
    result = 0.0
    for i in range(10):
        result += x
        if result > 100:
            break
    return result
```

Note: after a `break` out of a `for`, the loop still spins through its
remaining iterations with an empty body (exact semantics, wasted spins) - an
early break out of a very long `range` is correct but not fast. Loops with an
`else` clause are still rejected (`for`/`else` semantics depend on how the
loop exited).
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

### Higher-Order Differentiation (Second and Third Derivatives)

**Status**: ✅ Supported — differentiate a gradient function again (second
and, for ordinary NumPy functions, third derivatives)

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

**Third derivatives** also work for ordinary NumPy functions — nesting `grad`
a third time differentiates the second-order adjoint code. This relies on
adjoints registered for Tangent's own accumulation helpers (`tangent.unreduce`,
`tangent.unreduce_like`, `tangent.unbroadcast`, `tangent.add_grad`, ...), so
the third pass does not step into their type-dispatch bodies. Fourth and higher
orders are not yet reliable: the unoptimized path reaches the low-level tape
machinery (see below) and the optimized path can return incorrect values.

**The low-level tape API under higher-order AD:**

Functions that call Tangent's internal tape API directly — `tangent.push`,
`tangent.pop`, `tangent.push_stack`, `tangent.pop_stack`, `tangent.Stack` —
differentiate correctly at first and second order, in both optimized and
unoptimized modes:

```python
def uses_tape(a):
    _stack = tangent.Stack()
    b = a * a
    tangent.push(_stack, b, 'id')
    b = tangent.pop(_stack, 'id')
    return b            # b == a**2; d²/da² should be 2

tangent.grad(tangent.grad(uses_tape), optimized=False)(3.0)  # 2.0
tangent.grad(tangent.grad(uses_tape), optimized=True)(3.0)   # 2.0
```

This used to be wrong with `optimized=True` (a documented limitation), and
the mechanism is worth recording because it constrains the optimizer. Tape
pushes and pops are paired by an `op_id` argument, unique per generated pair.
Differentiating gradient code **again** duplicates op ids: the new primal
re-executes an old push and pop, and the new adjoint mirrors them (the
adjoint of a push is a pop and vice versa), so one op id then names several
distinct runtime pairs. Dead-code elimination used to look pairings up in
annotations that kept only the *last-seen* push/pop per op id, so it removed
a dead pop together with the **wrong** push — the tape stayed balanced in
count but crossed in dataflow, and a second-order gradient accumulator read a
primal value (or a pop hit a mismatched op id).

The fix (`tangent.optimization._tape_pairings`) makes dead-code elimination
tape-aware: it pairs pushes with the pops that consume them (unique op ids
pair directly; duplicated op ids within one function pair like parentheses in
program order; anything ambiguous becomes a barrier that is never removed)
and only ever removes a push and pop **together**. Balanced removal keeps the
stack consistent, so genuinely dead tape traffic — including the loop-counter
and condition pushes of control-flow gradients — is still eliminated, while
tape entries that a higher-order derivative needs survive. The same analysis
lets checkpointed loops (`tangent.grad(f, checkpoint=True)`) run with
optimizations enabled: the checkpoint bookkeeping either survives as a pair
or is removed as a pair, instead of leaving pops without pushes.

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
| **While loops** | ✅ | ✅ | ⚠️ | ⚠️ |
| **Lambdas** | ⚠️ (assigned only) | ✅ | ✅ | ✅ |
| **Closures (external)** | ✅ | ✅ | ✅ | ✅ |
| **Nested defs inside fn** | ❌ | ✅ | ✅ | ✅ |
| **Dict (read)** | ✅ | ✅ | ✅ | ✅ |
| **Dict (construct)** | ✅ (string keys) | ✅ | ✅ | ✅ |
| **Dict (mutate)** | ❌ | ✅ | ✅ | ✅ |
| **Tuples** | ✅ | ✅ | ✅ | ✅ |
| **Try/except** | ❌ | ⚠️ | ⚠️ | ⚠️ |
| **Break/continue** | ✅ | ✅ | ✅ | ✅ |

## Testing

Comprehensive tests available:
- `tests/test_dict_construction.py` - Local dict construction (string keys)
- `tests/test_membership_operators.py` - `in` / `not in` operators
- Individual feature test files for each supported feature

## Summary Statistics

- **Fully Supported**: 34+ features (including early returns, tuples, membership/identity operators, f-strings, dict `.get()`, `sum(d.values())`, set literals, constant-range list/set/dict comprehensions, and subscript assignment such as `a[i] = x` / `a[sl] = x`)
- **Partially Supported**: 1 feature (some loops)
- **Not Supported**: 10+ features
- **Overall Coverage**: ~62% of common Python features

> Note: features that cannot be differentiated are rejected with a clear,
> actionable error rather than silently returning a wrong gradient. Nested
> function definitions inside the differentiated function and recursion fall in
> this category (they crashed before); assigned lambdas, external closures and
> constant-range comprehensions are the supported alternatives.

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
   - Use set operations (union/intersection); set literals and constant-range comprehensions are fine
   - Define nested functions or use recursion inside a differentiated function (hoist helpers to module level); assigned lambdas and external closures are fine

3. **⚠️ BE CAREFUL**:
   - List comprehensions may range over runtime iterables (single generator, plain-name target); set/dict comprehensions must range over a constant `range(...)`/list/tuple and don't support `if` filters
   - List building uses `.append()`/argument-less `.pop()` on plain variables; other in-place list mutators are rejected
   - Early returns in if/elif/else are supported; returns inside loops are not

## See Also

- [Boolean Operator Support](BOOLEAN_OPERATOR_SUPPORT.md)
- [For Loop Support](FOR_LOOP_SUPPORT.md)
- [While Loop Support](WHILE_LOOP_SUPPORT.md)
- [Augmented Assignment Support](AUGMENTED_ASSIGNMENT_SUPPORT.md)
- [Assert and Pass Support](ASSERT_PASS_SUPPORT.md)
- [Tuple Support](TUPLE_SUPPORT.md)
- [Error Messages](ERROR_MESSAGES.md)
