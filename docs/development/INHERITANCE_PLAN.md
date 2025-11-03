# Inheritance Support Implementation Plan

## Goal
Extend Tangent's class support to handle inheritance, allowing differentiation through methods that use Python's class hierarchy.

## Current Limitation
The current implementation (`class_desugar.py`) only handles:
- Direct method calls on class instances
- Instance attributes from `__init__`
- Method chaining within the same class

**It does NOT handle:**
- Methods inherited from parent classes
- Method overriding
- `super()` calls
- Multiple inheritance

## Examples to Support

### Example 1: Simple Inheritance
```python
class Shape:
    def area_squared(self, x):
        return self.area(x) ** 2

class Circle(Shape):
    def area(self, radius):
        return 3.14159 * radius ** 2

def f(r):
    circle = Circle()
    return circle.area_squared(r)  # Should work!

df = tangent.grad(f)  # Should compute gradient
```

### Example 2: Method Overriding
```python
class Base:
    def compute(self, x):
        return x ** 2

class Derived(Base):
    def compute(self, x):
        return x ** 3  # Overrides base method

def f(x):
    obj = Derived()
    return obj.compute(x)

df = tangent.grad(f)  # Should use Derived's version
```

### Example 3: Parent Attributes
```python
class Vehicle:
    def __init__(self, speed_factor):
        self.speed_factor = speed_factor

class Car(Vehicle):
    def __init__(self, speed_factor, efficiency):
        super().__init__(speed_factor)
        self.efficiency = efficiency

    def cost(self, distance):
        return distance * self.speed_factor / self.efficiency

def f(d):
    car = Car(1.5, 0.8)
    return car.cost(d)

df = tangent.grad(f)  # Should handle parent attributes
```

## Implementation Strategy

### Phase 1: Method Resolution Order (MRO)
Python uses the C3 linearization algorithm to determine method resolution order. We can use Python's built-in `inspect.getmro(cls)` to get the correct order.

**Key insight**: When looking up a method, search through MRO until we find it.

```python
def find_method_in_hierarchy(class_obj, method_name):
    """Find method by searching through MRO."""
    for base_class in inspect.getmro(class_obj):
        if hasattr(base_class, method_name):
            method = getattr(base_class, method_name)
            if inspect.ismethod(method) or inspect.isfunction(method):
                return method, base_class
    return None, None
```

### Phase 2: Attribute Resolution
Need to track attributes from parent `__init__` methods.

**Approach**:
1. When processing `__init__`, check if it calls `super().__init__(...)`
2. If yes, recursively process parent `__init__` methods
3. Merge attributes from parent and child

### Phase 3: Handle `super()` Calls
`super()` is tricky because it's a runtime construct. For differentiation purposes:
- Detect `super().__init__(...)` patterns
- Inline parent's `__init__` with appropriate arguments
- Track which class's method we're currently inlining to get context for `super()`

## Technical Challenges

### Challenge 1: `super()` is Dynamic
`super()` behavior depends on:
- The class it's called from
- The instance type
- The MRO

**Solution**: Since we're doing static transformation, we can use the class definition context to resolve `super()` statically.

### Challenge 2: Multiple Inheritance
Python supports multiple inheritance with C3 linearization.

**Solution**: Use `inspect.getmro()` which handles this correctly. Search through MRO in order.

### Challenge 3: Attribute Conflicts
Parent and child might both set the same attribute.

**Solution**: Child attributes override parent attributes (follows Python semantics).

## Implementation Steps

### Step 1: Enhance `_extract_instance_attrs` (30 min)
```python
def _extract_instance_attrs(self, var_name, class_obj, init_args, init_keywords):
    """Extract attributes from class hierarchy."""
    attrs = {}

    # Process parent classes first (in reverse MRO order)
    for base_class in reversed(inspect.getmro(class_obj)[1:]):  # Skip object
        if hasattr(base_class, '__init__'):
            parent_attrs = self._extract_attrs_from_init(
                base_class, init_args, init_keywords
            )
            attrs.update(parent_attrs)  # Parent attrs

    # Process current class (overrides parent)
    current_attrs = self._extract_attrs_from_init(
        class_obj, init_args, init_keywords
    )
    attrs.update(current_attrs)  # Child attrs override

    self.instance_attrs[var_name] = attrs
```

### Step 2: Enhance `visit_Call` for MRO Lookup (20 min)
```python
def visit_Call(self, node):
    """Inline method calls using MRO."""
    if (isinstance(node.func, gast.Attribute) and
        isinstance(node.func.value, gast.Name)):

        obj_name = node.func.value.id
        method_name = node.func.attr

        if obj_name in self.instance_vars:
            class_obj = self.instance_vars[obj_name]['class']

            # NEW: Use MRO to find method
            method, defining_class = self._find_method_in_mro(
                class_obj, method_name
            )

            if method:
                return self._inline_method(method, obj_name, node.args,
                                          node.keywords)

    self.generic_visit(node)
    return node

def _find_method_in_mro(self, class_obj, method_name):
    """Find method using MRO."""
    for base_class in inspect.getmro(class_obj):
        if base_class is object:
            continue
        if hasattr(base_class, method_name):
            attr = getattr(base_class, method_name)
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                return attr, base_class
    return None, None
```

### Step 3: Handle `super().__init__()` (40 min)
Detect and inline `super().__init__(...)` calls in `__init__` methods.

```python
def _process_super_init_call(self, stmt, class_obj, param_map):
    """Detect and process super().__init__() calls."""
    # Check if stmt is: super().__init__(...)
    if (isinstance(stmt, gast.Expr) and
        isinstance(stmt.value, gast.Call)):

        call = stmt.value
        # Check for super().__init__ pattern
        if (isinstance(call.func, gast.Attribute) and
            call.func.attr == '__init__' and
            isinstance(call.func.value, gast.Call) and
            isinstance(call.func.value.func, gast.Name) and
            call.func.value.func.id == 'super'):

            # Get parent class
            mro = inspect.getmro(class_obj)
            if len(mro) > 1:
                parent_class = mro[1]
                # Recursively extract parent attrs
                return self._extract_attrs_from_init(
                    parent_class, call.args, call.keywords
                )
    return {}
```

### Step 4: Comprehensive Testing (30 min)
Create `tests/test_inheritance.py` with:
- Simple inheritance (1 level)
- Multi-level inheritance (A -> B -> C)
- Method overriding
- Attribute inheritance
- `super()` calls
- Multiple inheritance

## Test Cases

### Test 1: Simple Inheritance
```python
class Base:
    def square(self, x):
        return x ** 2

class Derived(Base):
    pass  # Inherits square

def f(x):
    obj = Derived()
    return obj.square(x)

df = tangent.grad(f)
assert abs(df(3.0) - 6.0) < 1e-10  # 2*x
```

### Test 2: Method Override
```python
class Base:
    def compute(self, x):
        return x ** 2

class Derived(Base):
    def compute(self, x):
        return x ** 3

def f(x):
    obj = Derived()
    return obj.compute(x)

df = tangent.grad(f)
assert abs(df(2.0) - 12.0) < 1e-10  # 3*x^2 = 12
```

### Test 3: Parent Attributes
```python
class Vehicle:
    def __init__(self, factor):
        self.factor = factor

class Car(Vehicle):
    def __init__(self, factor):
        super().__init__(factor)

    def compute(self, x):
        return x * self.factor

def f(x):
    car = Car(2.5)
    return car.compute(x)

df = tangent.grad(f)
assert abs(df(3.0) - 2.5) < 1e-10
```

### Test 4: Multi-level Inheritance
```python
class A:
    def base_method(self, x):
        return x ** 2

class B(A):
    def middle_method(self, x):
        return self.base_method(x) + x

class C(B):
    def top_method(self, x):
        return self.middle_method(x) * 2

def f(x):
    obj = C()
    return obj.top_method(x)  # 2*(x^2 + x)

df = tangent.grad(f)
# d/dx[2(x^2 + x)] = 2(2x + 1) = 4x + 2
assert abs(df(3.0) - 14.0) < 1e-10  # 4*3 + 2 = 14
```

## Timeline

- **Step 1**: Enhance attribute extraction (30 min)
- **Step 2**: MRO method lookup (20 min)
- **Step 3**: Handle `super()` (40 min)
- **Step 4**: Testing (30 min)
- **Documentation**: (20 min)

**Total**: ~2.5 hours

## Success Criteria

✅ Simple inheritance works (method inherited from parent)
✅ Method overriding works (child method overrides parent)
✅ Attribute inheritance works (parent attributes accessible in child methods)
✅ Multi-level inheritance works (A -> B -> C)
✅ `super().__init__()` calls work
✅ All tests pass
✅ Documentation updated

## Future Enhancements (Not in Scope)

- Multiple inheritance (beyond simple diamond)
- Abstract base classes
- Classmethods and staticmethods in inheritance
- Property inheritance
- Metaclasses

---

**Status**: 📋 READY TO IMPLEMENT
**Date**: 2025-11-03
**Estimated Time**: 2.5 hours
