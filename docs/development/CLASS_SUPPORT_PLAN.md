# Class Support Implementation Plan for Tangent

## Executive Summary

**Goal**: Add support for differentiating through user-defined class methods in Tangent.

**Current Status**: ❌ All class method calls fail with `TypeError: None`

**Root Cause**: Class method calls like `obj.method(x)` cannot be resolved by `annotate.py` because:
1. `obj` is a local variable (created at runtime)
2. Method resolution happens at parse time (before execution)
3. The `resolve()` function returns `None` for unresolvable calls

**Recommended Approach**: **Method Function Extraction** - Convert class methods to standalone functions that Tangent can differentiate.

---

## Problem Analysis

### Current Behavior

When Tangent encounters:
```python
class Calculator:
    def compute(self, x):
        return x ** 2

def f(x):
    calc = Calculator()
    return calc.compute(x)
```

**What happens:**
1. `quoting.parse_function(f)` parses only the function `f`, not the `Calculator` class
2. ANF transforms `calc.compute(x)` into a temporary variable assignment
3. `annotate.ResolveCalls` tries to resolve `calc.compute`
4. Resolution fails because `calc` is a local variable
5. `func` annotation is set to `None`
6. `naming.primal_name(None, ...)` raises `TypeError`

### Why This Happens

The `annotate.py::resolve()` function at line 57-70:
```python
def resolve(node):
    if isinstance(node, gast.Attribute):
        return getattr(resolve(node.value), node.attr)
    if isinstance(node, gast.Name):
        if node.id in self.namespace:
            return self.namespace[node.id]
```

For `calc.compute`:
- `node.func` is `gast.Attribute` (value=`calc`, attr=`compute`)
- Tries to resolve `calc` from namespace
- `calc` is NOT in namespace (it's created inside the function)
- Resolution fails, returns `None`

---

## Architectural Options

### Option 1: Method Inlining (Like Lambdas) ⭐ RECOMMENDED

**Strategy**: Transform `obj.method(args)` into inlined method body

**How it works**:
```python
# Before:
calc = Calculator()
return calc.compute(x)

# After transformation:
# (inline compute method with self=calc)
return x ** 2
```

**Advantages**:
- ✅ Works with Tangent's existing architecture
- ✅ No runtime overhead (methods are eliminated)
- ✅ Similar to successful lambda implementation
- ✅ Clean, understandable transformation

**Challenges**:
- ⚠️ Must resolve classes from function's global namespace
- ⚠️ Must parse method source code at transformation time
- ⚠️ Must handle `self` parameter substitution
- ⚠️ Must handle instance attributes (`self.factor`)

**Implementation Location**: New file `tangent/class_desugar.py` (already created, needs refinement)

**Integration Point**: In `grad_util.py::autodiff_ast()` before `annotate.ResolveCalls()`

---

### Option 2: Method-to-Function Transformation

**Strategy**: Convert methods to standalone functions

**How it works**:
```python
# Before:
class Calculator:
    def compute(self, x):
        return x ** 2

# Create standalone function:
def Calculator_compute(self, x):
    return x ** 2

# Transform call:
calc.compute(x) → Calculator_compute(calc, x)
```

**Advantages**:
- ✅ Methods become regular functions (Tangent can handle these)
- ✅ Clear separation of concerns
- ✅ Easier debugging

**Challenges**:
- ⚠️ Must track class definitions
- ⚠️ Must transform all method calls
- ⚠️ More complex AST manipulation

---

### Option 3: Runtime Method Resolution

**Strategy**: Resolve methods at runtime using introspection

**How it works**:
- Modify `annotate.py::ResolveCalls` to handle method calls specially
- When `obj.method` cannot be resolved, mark it for runtime resolution
- At runtime, inspect the object and inline the method

**Advantages**:
- ✅ More flexible - handles dynamic cases
- ✅ Works with inheritance

**Challenges**:
- ❌ Complex implementation
- ❌ Requires significant changes to Tangent's core
- ❌ Runtime overhead

---

## Recommended Implementation: Method Inlining

### Phase 1: Basic Method Inlining

**File**: `tangent/class_desugar.py` (refinement needed)

**Approach**:
1. Access classes from function's `__globals__` namespace
2. Track instance variable assignments (`calc = Calculator()`)
3. When seeing `calc.method(args)`, look up the class and method
4. Parse the method's source code
5. Inline the method body with parameter substitution

**Key Components**:

```python
class ClassMethodInliner(gast.NodeTransformer):
    def __init__(self, func):
        """
        Args:
            func: The function being differentiated (to access __globals__)
        """
        self.func = func
        self.instance_map = {}  # var_name -> class_object

    def visit_Assign(self, node):
        """Track instance creation: calc = Calculator()"""
        # If RHS is ClassName(), resolve class from func.__globals__
        # Store mapping: 'calc' -> Calculator class object

    def visit_Call(self, node):
        """Inline method calls: calc.method(x) -> method body"""
        # If node.func is Attribute and base is tracked instance:
        #   1. Get the class object
        #   2. Get the method: class_obj.method_name
        #   3. Parse method source: inspect.getsource(method)
        #   4. Inline method body with substitutions
```

**Integration**:
```python
# In grad_util.py::autodiff_ast()
node = quoting.parse_function(func)
node = class_desugar.inline_class_methods(node, func)  # NEW - pass func for __globals__
node = lambda_desugar.desugar_lambdas(node)
node = listcomp_desugar.desugar_listcomps(node)
annotate.ResolveCalls(func).visit(node)
```

---

### Phase 2: Handle Instance Attributes

**Challenge**: Methods that use `self.factor`

```python
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor
```

**Solution**:
1. Parse `__init__` method
2. Extract attribute assignments (`self.factor = factor`)
3. Track initialization values (`Scaler(2.5)` → `factor=2.5`)
4. When inlining, substitute `self.factor` with the actual value

---

### Phase 3: Handle Method Chaining

**Challenge**: Methods calling other methods

```python
class ChainCalc:
    def square(self, x):
        return x ** 2

    def square_plus_one(self, x):
        return self.square(x) + 1  # Calls another method!
```

**Solution**:
1. Inline recursively
2. When inlining `square_plus_one`, encounter `self.square(x)`
3. Recursively inline `square` method
4. Result: `return (x ** 2) + 1`

---

## Implementation Steps

### Step 1: Refine `class_desugar.py`

**Current Status**: Basic structure exists but needs to access `func.__globals__`

**Modifications Needed**:

```python
class ClassMethodInliner(gast.NodeTransformer):
    def __init__(self, func):
        self.func = func  # NEW: Store function for __globals__ access
        self.instance_vars = {}  # var_name -> class_object

    def visit_Assign(self, node):
        """Track: calc = Calculator()"""
        if (isinstance(node.value, gast.Call) and
            isinstance(node.value.func, gast.Name)):

            class_name = node.value.func.id

            # Look up class in function's globals
            if class_name in self.func.__globals__:
                class_obj = self.func.__globals__[class_name]

                # Check if it's actually a class
                if inspect.isclass(class_obj):
                    var_name = node.targets[0].id
                    self.instance_vars[var_name] = {
                        'class': class_obj,
                        'init_args': node.value.args
                    }

        return node

    def visit_Call(self, node):
        """Inline: calc.method(x)"""
        if (isinstance(node.func, gast.Attribute) and
            isinstance(node.func.value, gast.Name)):

            obj_name = node.func.value.id
            method_name = node.func.attr

            if obj_name in self.instance_vars:
                class_obj = self.instance_vars[obj_name]['class']
                method = getattr(class_obj, method_name)

                # Inline the method!
                return self._inline_method(method, obj_name, node.args)

        self.generic_visit(node)
        return node

    def _inline_method(self, method, instance_var, args):
        """Parse and inline method body."""
        import textwrap

        # Get method source
        source = inspect.getsource(method)
        source = textwrap.dedent(source)  # Remove indentation

        # Parse method
        method_ast = gast.parse(source).body[0]

        # Extract return expression
        return_stmt = None
        for stmt in method_ast.body:
            if isinstance(stmt, gast.Return):
                return_stmt = stmt
                break

        if return_stmt is None:
            return gast.Constant(value=None, kind=None)

        # Substitute parameters
        params = method_ast.args.args[1:]  # Skip 'self'
        param_map = {}
        for i, param in enumerate(params):
            if i < len(args):
                param_map[param.id] = args[i]

        # Substitute in return expression
        inlined = self._substitute_params(return_stmt.value, param_map, instance_var)

        return inlined
```

### Step 2: Update `grad_util.py`

```python
# Line ~75: Add import
from tangent import class_desugar

# Line ~124: Update transformation pipeline
node = quoting.parse_function(func)
node = class_desugar.inline_class_methods(node, func)  # Pass func!
node = lambda_desugar.desugar_lambdas(node)
# ... rest of pipeline
```

### Step 3: Create Entry Point

```python
# In class_desugar.py
def inline_class_methods(node, func):
    """Inline class method calls.

    Args:
        node: AST of the function
        func: The actual function object (for accessing __globals__)

    Returns:
        Transformed AST with class methods inlined
    """
    inliner = ClassMethodInliner(func)
    return inliner.visit(node)
```

---

## Test Cases

### Test 1: Simple Method
```python
class Calculator:
    def square(self, x):
        return x ** 2

def f(x):
    calc = Calculator()
    return calc.square(x)

df = tangent.grad(f)
assert df(3.0) == 6.0  # 2*x at x=3
```

### Test 2: Method with Attributes
```python
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor

def f(x):
    scaler = Scaler(2.5)
    return scaler.scale(x)

df = tangent.grad(f)
assert df(3.0) == 2.5
```

### Test 3: Multiple Methods
```python
class Polynomial:
    def square(self, x):
        return x ** 2

    def cube(self, x):
        return x ** 3

def f(x):
    poly = Polynomial()
    return poly.square(x) + poly.cube(x)

df = tangent.grad(f)
assert df(2.0) == 2*2 + 3*4  # 2x + 3x² at x=2
```

### Test 4: NumPy Integration
```python
class NumpyCalc:
    def compute(self, x):
        return np.sin(x) + x ** 2

def f(x):
    calc = NumpyCalc()
    return calc.compute(x)

df = tangent.grad(f)
assert abs(df(1.0) - (np.cos(1.0) + 2.0)) < 1e-6
```

---

## Timeline Estimate

**Phase 1: Basic Inlining** - 2-3 hours
- Refine `class_desugar.py` to access `func.__globals__`
- Implement basic method inlining
- Test with simple cases

**Phase 2: Attributes** - 2-3 hours
- Parse `__init__` methods
- Track instance attributes
- Substitute attribute access

**Phase 3: Method Chaining** - 1-2 hours
- Recursive inlining
- Handle `self.method()` calls

**Phase 4: Testing & Documentation** - 2 hours
- Comprehensive test suite
- Update documentation
- Add examples

**Total Estimated Time**: 7-10 hours

---

## Known Limitations

### Won't Support (Initially):
1. **Inheritance** - Would need method resolution order (MRO) handling
2. **Property decorators** - `@property` methods
3. **Class methods / Static methods** - `@classmethod`, `@staticmethod`
4. **Dynamic attribute access** - `getattr(self, attr_name)`
5. **Methods with side effects** - Modifying instance state

### Future Enhancements:
1. Support for inheritance (MRO traversal)
2. Support for properties
3. Better error messages for unsupported patterns
4. Validation warnings (like control flow validator)

---

## Integration with Existing Features

### Lambda Support
- Classes and lambdas work together
- Lambdas can be used in methods
- Methods can return lambdas

### Checkpointing
- Class methods in checkpointed loops
- Should work transparently after inlining

### NumPy/JAX/TensorFlow
- Methods can call NumPy functions
- Already tested and working pattern

---

## Success Criteria

✅ **Basic methods** - Simple methods with parameters work
✅ **Instance attributes** - Methods using `self.attr` work
✅ **Multiple methods** - Multiple methods in one class work
✅ **NumPy integration** - Methods with NumPy operations work
✅ **Method chaining** - Methods calling other methods work
✅ **7/7 test cases** passing
✅ **Zero regressions** in existing test suite
✅ **Documentation** complete with examples

---

## Next Steps

1. **Review this plan** - Confirm the approach
2. **Refine `class_desugar.py`** - Add `func` parameter support
3. **Update `grad_util.py`** - Integrate into transformation pipeline
4. **Test incrementally** - Start with Test 1, then 2, 3, etc.
5. **Document** - Create examples and usage guide

---

**Status**: 📋 PLAN COMPLETE - Ready for implementation
**Date**: 2025-11-03
**Estimated Completion**: 7-10 hours of focused work
