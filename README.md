# Tangent - Source-to-Source Automatic Differentiation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)](tests/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/tangent_tutorial.ipynb)

**A modernized Python library for automatic differentiation with readable source code, educational visualizations, and multi-backend support.**

Originally developed by Google Research, now maintained and enhanced by [@pedronahum](https://github.com/pedronahum) with modern ML framework integrations and powerful visualization tools.

<p align="center">
  <img src="assets/gradient_flow.png" alt="Gradient Flow Visualization" width="70%">
  <br>
  <em>Visualize how gradients flow through your computations</em>
</p>

---

## 🌟 What Makes Tangent Unique?

Tangent performs **source-to-source** automatic differentiation - it transforms your Python code directly into gradient code that you can read, debug, and understand. Unlike other autodiff libraries:

- **📖 Readable**: Generated gradient code is pure Python you can inspect
- **🔍 Debuggable**: Step through gradient computation line by line
- **🎨 Visual**: Interactive computation graphs and gradient flow diagrams
- **⚡ Fast**: No tape overhead, compiled gradients run at full speed
- **🔧 Flexible**: Works with NumPy, JAX, and TensorFlow 2.x

![Autodiff Tool Space](docs/toolspace.png "Autodiff Tool Space")

---

## 🎨 Gallery of Gradients: See the Magic

**The killer feature: Readable gradient code!** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Gallery_of_Gradients.ipynb)

Unlike black-box autodiff libraries, Tangent shows you **exactly** how gradients are computed. Our curated gallery demonstrates this with 8 beautiful examples:

```python
# Your function
def f(x):
    result = 0.0
    for i in range(1, 6):
        result += x ** i
    return result

# See the gradient code!
df = tangent.grad(f, verbose=1)  # Prints the generated code
```

**What you'll see:**
- 🔢 Polynomial derivatives (chain rule basics)
- 🔄 For loops that run **in reverse** during backprop
- 🌀 While loops with stack-based tape recording
- 🔀 Conditional branching (if/else)
- 📊 NumPy array operations and broadcasting
- 📦 Nested function inlining
- 🔢 Matrix operations with colon slicing
- ⚡ Optimization comparison (before/after)

Each example shows: **Original function → Generated gradient code → Why it looks that way → Verification**

**Perfect for:**
- 🎓 Learning how autodiff really works
- 🐛 Debugging gradient computations
- 👨‍🏫 Teaching calculus or ML concepts
- 🔬 Research and algorithm development

**[→ Explore the Gallery](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Gallery_of_Gradients.ipynb)** | [📖 Documentation](examples/README_GALLERY.md)

---

## 🚀 Quick Start: Building Energy Optimization

**Try it now in Colab!** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Building_Energy_Optimization_with_Tangent.ipynb)

See Tangent in action with a real-world example - optimizing building heating to minimize energy costs.

> **Based on**: PassiveLogic's [Breaking the AI Speed Barrier](https://passivelogic.com/blog/?post=breaking-ai-speed-barrier-blog) and their [Differentiable Swift Examples](https://github.com/PassiveLogic/differentiable-swift-examples/tree/main/Benchmarks/BuildingSimulation). This example demonstrates how Tangent achieves similar performance in Python.

```python
import tangent
import numpy as np

# Physical simulation: building temperature dynamics
def simulate_building(heating_schedule, outdoor_temp, electricity_price, params):
    T = params['T_initial']
    total_cost = 0.0

    for t in range(len(heating_schedule)):
        # Temperature dynamics with heating and solar gain
        dT_dt = (outdoor_temp[t] - T) / (params['R'] * params['C']) + \
                heating_schedule[t] / params['C']
        T = T + params['dt'] * dT_dt

        # Cost = energy cost + comfort penalty
        total_cost += electricity_price[t] * heating_schedule[t] + \
                     params['lambda_comfort'] * (T - params['T_target']) ** 2

    return total_cost

# 🎯 Automatic differentiation - ONE LINE!
grad_simulate = tangent.grad(simulate_building, optimized=True)

# Compute gradient to optimize heating schedule
gradient = grad_simulate(heating_schedule, outdoor_temp, electricity_price, params)

# Use gradients for optimization (gradient descent)
heating_schedule -= learning_rate * gradient
```

**What you'll learn:**
- 🏢 Real-world optimization problem (building thermal control)
- 📖 Inspect generated gradient code (see what Tangent creates!)
- ⚡ Compare unoptimized vs optimized performance
- 📊 Visualize gradients and optimization convergence
- 🎓 Perfect for teaching automatic differentiation!

**[→ Open Interactive Notebook](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Building_Energy_Optimization_with_Tangent.ipynb)**

---

## 🐍 Comprehensive Python Support

Tangent supports a remarkably complete subset of Python for numerical computing:

**✅ Control Flow**: if/elif/else, for loops, while loops, ternary operator
**✅ Operators**: Boolean (`and`, `or`, `not`), arithmetic, comparisons, augmented assignments (`+=`, `-=`, etc.)
**✅ Functions**: Lambdas, closures, nested functions, default/keyword arguments
**✅ Classes**: User-defined classes with method inlining, instance attributes, method chaining, inheritance
**✅ Data**: Dictionaries (read-only), NumPy arrays, lists (syntactic), tuples (read-only)
**✅ Statements**: assert, pass, return, assignment

**📊 Feature Coverage**: ~60% of common Python features fully supported

**📖 Complete Reference**: [Python Feature Support Guide](docs/features/PYTHON_FEATURE_SUPPORT.md)

---

## ⚡ Advanced Optimization Pipeline

Tangent includes a sophisticated multi-pass optimization system that generates **production-grade gradient code**. When you call `tangent.grad(f, optimized=True)`, your gradients go through 7 optimization passes:

### 🔧 Optimization Passes

#### 1. **Constant Folding**
Evaluates constant expressions at compile time:
```python
# Before
result = 2 * 3.14159 * x

# After
result = 6.28318 * x
```

#### 2. **Dead Code Elimination (DCE)**
Removes unused computations using activity analysis and control flow:
```python
# Before
def df(x):
    temp1 = x ** 2
    temp2 = x * 3  # Never used!
    return temp1 + 5

# After
def df(x):
    temp1 = x ** 2
    return temp1 + 5
```

**Advanced DCE** uses backward slicing to eliminate entire chains of unused computations - typically removes 30-50% of generated code!

#### 3. **Assignment Propagation**
Inlines single-use variables to reduce overhead:
```python
# Before
a = x * 2
b = a + 3
return b

# After
return x * 2 + 3
```

#### 4. **Strength Reduction**
Replaces expensive operations with cheaper equivalents:
```python
# Before
result = x ** 2        # Expensive: power operation
cost = x / 3.14159     # Expensive: division

# After
result = x * x         # Cheap: multiplication (5-25× faster!)
cost = x * 0.318310    # Cheap: multiply by reciprocal (2-3× faster!)
```

**Common transformations**:
- `x ** 2` → `x * x` (25× faster)
- `x ** 3` → `x * x * x` (15× faster)
- `x ** 0.5` → `sqrt(x)` (5× faster)
- `x / constant` → `x * reciprocal` (2× faster)

#### 5. **Common Subexpression Elimination (CSE)**
Identifies and reuses repeated computations:
```python
# Before
a = np.sin(x) ** 2 + np.cos(x) ** 2
b = np.sin(x) ** 2 - np.cos(x) ** 2  # sin(x)**2 and cos(x)**2 computed twice!

# After
_cse_1 = np.sin(x) ** 2
_cse_2 = np.cos(x) ** 2
a = _cse_1 + _cse_2
b = _cse_1 - _cse_2
```

Typical savings: **15-30% reduction** in gradient computation time.

#### 6. **Algebraic Simplification**
Uses SymPy to apply mathematical identities:
```python
# Before
result = np.sin(x) ** 2 + np.cos(x) ** 2

# After
result = 1.0  # Pythagorean identity!
```

**Mathematical identities applied**:
- `sin²(x) + cos²(x)` → `1`
- `log(exp(x))` → `x`
- `x * 1` → `x`
- `x + 0` → `x`
- `x * 0` → `0`

#### 7. **Fixed-Point Iteration**
Repeats passes 1-6 until no more optimizations are possible. Each pass creates opportunities for others:
- Constant folding → enables more DCE
- DCE → creates single-use variables for propagation
- Strength reduction → enables more CSE
- CSE → creates new constants for folding

### 📊 Performance Impact

**Real-world results** from the Building Energy Optimization example:

| Configuration | Gradient Time | Speedup |
|---------------|--------------|---------|
| Unoptimized | 100ms | 1.0× |
| Basic DCE only | 51ms | **1.95×** |
| Strength Reduction | 45ms | **2.22×** |
| Full Pipeline (all 7 passes) | 42.5ms | **2.35×** |

**Per-operation speedups**:
- Power operations: **5-25× faster** (x²→x*x)
- Division operations: **2-3× faster** (x/c→x*reciprocal)
- Redundant expressions: **eliminated entirely**

### 🎯 Usage

```python
import tangent

# Default: optimized=True (recommended for production)
df = tangent.grad(f, optimized=True)

# Unoptimized: see all intermediate steps (debugging/education)
df_unopt = tangent.grad(f, optimized=False)

# See what optimizations did
df_verbose = tangent.grad(f, optimized=True, verbose=1)
# Prints:
# DCE: Eliminated 15 statements (42 → 27)
# Strength Reduction: Applied 8 transformations
# CSE: Eliminated 5 redundant subexpressions
```

### 🔬 Optimization Deep Dive

Want to learn more? Check out our comprehensive documentation:

- **[Symbolic Optimizations Guide](docs/optimizations/SYMBOLIC_OPTIMIZATIONS_COMPLETE.md)** - CSE & algebraic simplification details
- **[Strength Reduction Guide](docs/optimizations/STRENGTH_REDUCTION_COMPLETE.md)** - Power/division optimization patterns
- **[Performance Analysis](docs/optimizations/PERFORMANCE_ANALYSIS.md)** - Benchmark results and analysis
- **[DCE Implementation](tangent/optimizations/dce.py)** - Activity analysis algorithm
- **[Gallery Example #8](examples/Gallery_of_Gradients.ipynb)** - See optimization in action!

### 💡 Why Optimizations Matter

1. **Production Performance**: 2-3× faster gradients in real workloads
2. **Readable AND Fast**: Optimized code is still pure Python
3. **No Runtime Overhead**: Optimizations happen once at `tangent.grad()` time
4. **Compiler-Grade**: Uses techniques from production compilers (LLVM, GCC)
5. **Educational**: Compare unoptimized vs optimized to understand performance

**The best part?** Optimizations are applied to readable Python code, so you can still inspect, debug, and modify the generated gradients!

---

## 🆕 What's New in This Fork

This modernized version includes major enhancements:

### ✅ **Lambda Function Support** 🆕
Lambda functions are now fully supported! Write concise, Pythonic code and let Tangent handle the differentiation:

```python
import tangent

def neural_net(x):
    # Use lambdas for activations!
    relu = lambda z: np.maximum(0, z)
    hidden = relu(x * 0.5)
    return np.sum(hidden ** 2)

# Gradients work seamlessly
df = tangent.grad(neural_net)
gradient = df(2.0)  # ✅ Works!
```

**How it works**: Tangent automatically inlines lambda functions at their call sites, preserving mathematical correctness while eliminating the lambda construct. Supports:
- ✅ Simple lambdas: `g = lambda x: x ** 2`
- ✅ Multi-argument lambdas: `f = lambda a, b: a * b`
- ✅ Nested calls: `g(h(x))` where both are lambdas
- ✅ NumPy/JAX/TensorFlow operations inside lambdas

📖 [See full documentation](docs/features/LAMBDA_SUPPORT_COMPLETE.md) with 7 comprehensive test cases!

### ✅ **Class Support** 🆕 NEW!
Full support for user-defined classes with automatic method inlining! Write object-oriented code and let Tangent handle differentiation:

```python
import tangent
import numpy as np

# Define classes with methods
class Polynomial:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        """Evaluate polynomial: a*x^2 + b*x + c"""
        return self.a * x ** 2 + self.b * x + self.c

def loss_function(x):
    # Create instance and call method
    poly = Polynomial(2.0, 3.0, 1.0)  # 2x^2 + 3x + 1
    return poly.evaluate(x)

# Gradients work seamlessly!
df = tangent.grad(loss_function)
gradient = df(5.0)  # 4*5 + 3 = 23.0 ✅ Works!
```

**How it works**: Tangent automatically inlines class methods at their call sites, substituting `self` and instance attributes. Methods are transformed into their computational content while preserving mathematical correctness.

**Supported patterns**:
- ✅ Simple methods: `def square(self, x): return x ** 2`
- ✅ Instance attributes: Methods using `self.factor`, `self.weight`, etc.
- ✅ Multiple methods: Multiple method calls in one class
- ✅ Method chaining: Methods calling other methods (`self.helper(x)`)
- ✅ NumPy/JAX/TensorFlow operations inside methods
- ✅ Multi-parameter methods: `def compute(self, x, y)`
- ✅ Different instances: Multiple objects of the same class

**Real-world examples**:

```python
# Scaler with instance attributes
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        return x * self.factor

def scaled_loss(x):
    scaler = Scaler(2.5)
    return scaler.scale(x) ** 2

df = tangent.grad(scaled_loss)
print(df(3.0))  # 15.0 ✅ Correct!

# Method chaining
class ChainedCalculator:
    def square(self, x):
        return x ** 2

    def double(self, x):
        return x * 2

    def process(self, x):
        return self.double(self.square(x))  # Chains methods!

def chained_computation(x):
    calc = ChainedCalculator()
    return calc.process(x)  # Returns 2x^2

df = tangent.grad(chained_computation)
print(df(3.0))  # 12.0 (gradient of 2x^2 is 4x) ✅

# NumPy integration
class NumpyModel:
    def predict(self, x):
        return np.sin(x) + x ** 2

def model_loss(x):
    model = NumpyModel()
    return model.predict(x)

df = tangent.grad(model_loss)
# Gradient: cos(x) + 2x
```

**Technical approach**: Uses runtime class resolution from `func.__globals__`, parses method source with `inspect.getsource()`, and performs multi-pass inlining for method chaining.

**Limitations** (currently):
- ⚠️ No property decorators (`@property`)
- ⚠️ No class/static methods (`@classmethod`, `@staticmethod`)
- ⚠️ Methods must be accessible via `inspect.getsource()`

📖 [See full documentation](docs/features/CLASS_SUPPORT_COMPLETE.md) with 14 comprehensive test cases covering all patterns!

### ✅ **Class Inheritance Support** 🆕 NEW!
Full support for Python class inheritance including `super()` calls! Build complex class hierarchies and differentiate through them:

```python
import tangent

# Base class
class Vehicle:
    def __init__(self, speed_factor):
        self.speed_factor = speed_factor

# Derived class using super()
class Car(Vehicle):
    def __init__(self, speed_factor, efficiency):
        super().__init__(speed_factor)  # Calls parent __init__
        self.efficiency = efficiency

    def cost(self, distance):
        # Uses both parent and child attributes!
        return distance * self.speed_factor / self.efficiency

def f(distance):
    car = Car(speed_factor=1.5, efficiency=0.5)
    return car.cost(distance)

# Gradients work perfectly with inheritance!
df = tangent.grad(f)
print(df(10.0))  # 3.0 ✅ Works!
```

**Supported inheritance patterns**:
- ✅ Simple inheritance: Child inherits parent methods
- ✅ Method overriding: Child method overrides parent method
- ✅ `super().__init__()`: Attribute inheritance from parent
- ✅ Multi-level inheritance: Grandparent → Parent → Child
- ✅ Method chaining across hierarchy: Derived methods calling inherited methods
- ✅ NumPy/JAX/TensorFlow in inherited methods

**Real-world example - Multi-level inheritance**:

```python
class Shape:
    def area_squared(self, x):
        return self.area(x) ** 2

class Circle(Shape):
    def area(self, radius):
        return 3.14159 * radius ** 2

def f(r):
    circle = Circle()
    return circle.area_squared(r)  # Calls inherited method!

df = tangent.grad(f)
print(df(2.0))  # ✅ Works through inheritance!
```

**How it works**: Tangent uses Python's Method Resolution Order (MRO) to locate inherited methods, detects `super().__init__()` patterns to extract parent attributes recursively, and performs multi-pass inlining to handle method chaining across inheritance levels.

📖 [See full inheritance documentation](docs/features/INHERITANCE_SUPPORT_COMPLETE.md) with 12 comprehensive test cases!

### ✅ **Conditional Expressions (Ternary Operator)** 🆕 NEW!
Full support for Python's ternary operator makes code cleaner and more Pythonic:

```python
import tangent

# ReLU activation using ternary operator
def relu(x):
    return x if x > 0 else 0.0

# Clipping function
def clip(x, min_val=0.0, max_val=1.0):
    return min_val if x < min_val else (max_val if x > max_val else x)

# Gradients work perfectly!
df_relu = tangent.grad(relu)
print(df_relu(5.0))   # 1.0 (positive input)
print(df_relu(-3.0))  # 0.0 (negative input)
```

**Supported**:
- ✅ Simple ternaries: `a if condition else b`
- ✅ Nested ternaries: `a if c1 else (b if c2 else c)`
- ✅ All comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
- ✅ Complex expressions in branches: `x**2 if x > 0 else 2*x`
- ✅ Multiple ternaries in sequence

**Common use cases**:
- **Activation functions**: ReLU, Leaky ReLU, ELU
- **Clipping/bounding**: `min(max(x, lo), hi)` → `lo if x < lo else (hi if x > hi else x)`
- **Piecewise functions**: Different formulas for different ranges
- **Sign functions**: `1.0 if x > 0 else -1.0`

📖 [See full documentation](docs/features/CONDITIONAL_EXPRESSION_SUPPORT.md) with 8 comprehensive test cases!

### ✅ **List Comprehension Support** 🆕 NEW!
Python's list comprehensions now work syntactically! While Python lists themselves aren't differentiable (use NumPy arrays for that), list comprehensions can be used in non-differentiated code:

```python
import tangent
import numpy as np

# List comprehensions work for data preparation
def process_batch(x):
    # Build list (not differentiated)
    samples = [x * i for i in range(5)]
    # Use NumPy for differentiated operations
    result = x ** 2
    return result

# Recommended pattern: Use NumPy directly
def efficient_version(x):
    coeffs = np.array([1.0, 2.0, 3.0])
    return np.sum(x * coeffs)  # ✅ Fully differentiable!

df = tangent.grad(efficient_version)
print(df(2.0))  # 6.0
```

**What works**:
- ✅ All list comprehension syntax: `[expr for x in iter if cond]`
- ✅ Nested comprehensions: `[[x*y for y in range(2)] for x in range(2)]`
- ✅ Multiple generators: `[x+y for x in a for y in b]`
- ✅ List comprehensions in non-differentiated code paths

**Limitation**: Python lists aren't differentiable. For differentiated operations, use NumPy/JAX arrays (which is the recommended pattern anyway for numerical computing).

📖 [See full documentation](docs/features/LIST_COMPREHENSION_SUPPORT.md) with technical details and best practices!

### ✅ **Boolean Operator Support** 🆕 NEW!
Python's boolean operators (`and`, `or`, `not`) now work seamlessly in differentiated code! Write natural, Pythonic conditions:

```python
import tangent

# Range checking with 'and'
def clip_gradient(x):
    return x if x > 0 and x < 10 else x ** 2

# Boundary detection with 'or'
def boundary_penalty(x):
    return x ** 2 if x < 0 or x > 10 else x

# Condition inversion with 'not'
def safe_range(x):
    out_of_bounds = x < 0 or x > 10
    return x if not out_of_bounds else 0.0

# Gradients work perfectly!
df_clip = tangent.grad(clip_gradient)
print(df_clip(5.0))   # 1.0 (in range [0,10])
print(df_clip(15.0))  # 30.0 (out of range, returns x^2)
```

**Supported**:
- ✅ `and` operator with short-circuit evaluation
- ✅ `or` operator with short-circuit evaluation
- ✅ `not` operator for negation
- ✅ Complex chained expressions: `(x > 0 and x < 5) or (x > 10 and x < 15)`
- ✅ Multiple operators in sequence: `x > 0 and x < 10 and x != 5`
- ✅ Nested boolean expressions: `not (x < 0 or x > 10)`
- ✅ Boolean operators with NumPy comparisons

**Common use cases**:
- **Range validation**: Check if values are within bounds
- **Boundary conditions**: Detect edge cases in numerical algorithms
- **Composite conditions**: Combine multiple criteria
- **Piecewise functions**: Different behavior for different regions
- **Guard clauses**: Protect against invalid inputs

📖 [See full documentation](docs/features/BOOLEAN_OPERATOR_SUPPORT.md) with 8 comprehensive test cases!

### ✅ **Augmented Assignment Operators** 🆕 NEW!
Python's augmented assignment operators (`+=`, `-=`, `*=`, `/=`, `**=`) work perfectly! Write natural, concise accumulation patterns:

```python
import tangent

# Accumulator pattern (common in training loops)
def loss_accumulator(x):
    total_loss = 0.0
    total_loss += (x - 1.0) ** 2
    total_loss += (x - 2.0) ** 2
    total_loss += (x - 3.0) ** 2
    return total_loss

# Chained operations
def chained(x):
    result = 1.0
    result += x       # result = 1 + x
    result *= 2.0     # result = 2(1 + x)
    result += x ** 2  # result = 2 + 2x + x^2
    return result

# Gradient descent step
def update_step(x):
    grad = 2 * (x - 5.0)
    learning_rate = 0.1
    x -= learning_rate * grad
    return x ** 2

# All work seamlessly!
df_loss = tangent.grad(loss_accumulator)
df_chain = tangent.grad(chained)
df_update = tangent.grad(update_step)
```

**Supported operators**:
- ✅ `+=` (addition)
- ✅ `-=` (subtraction)
- ✅ `*=` (multiplication)
- ✅ `/=` (division)
- ✅ `**=` (exponentiation)
- ✅ `//=` (floor division)
- ✅ `%=` (modulo)

**Common use cases**:
- **Loss accumulation**: Sum multiple loss terms
- **Gradient descent**: `param -= lr * grad`
- **Momentum updates**: `velocity *= momentum; velocity += grad`
- **Running sums**: Accumulate values in loops
- **Weighted combinations**: `result *= coefficient`

📖 [See full documentation](docs/features/AUGMENTED_ASSIGNMENT_SUPPORT.md) with 10 comprehensive test cases!

### ✅ **For Loop Support** 🆕 NEW!
Python's `for` loops with `range()` are fully supported! Write iterative algorithms naturally:

```python
import tangent

# Polynomial evaluation
def polynomial(x):
    result = 0.0
    for i in range(4):
        result += x ** float(i)  # 1 + x + x^2 + x^3
    return result

# Taylor series approximation
def taylor_exp(x):
    result = 1.0
    term = 1.0
    for i in range(1, 5):
        term *= x / float(i)
        result += term
    return result

# Nested loops
def nested_sum(x):
    result = 0.0
    for i in range(2):
        for j in range(3):
            result += x * (float(i) + float(j))
    return result

# All work perfectly!
df_poly = tangent.grad(polynomial)
df_exp = tangent.grad(taylor_exp)
df_nested = tangent.grad(nested_sum)

print(df_poly(2.0))  # 17.0 (1 + 2x + 3x^2 = 1 + 4 + 12)
```

**Supported patterns**:
- ✅ `range(stop)` - iterate from 0 to stop-1
- ✅ `range(start, stop)` - iterate from start to stop-1
- ✅ `range(start, stop, step)` - iterate with custom step
- ✅ Using loop variable in computations
- ✅ Nested loops (loop within loop)
- ✅ Conditionals inside loops
- ✅ Complex expressions per iteration
- ✅ NumPy operations in loops
- ✅ Empty loops (zero iterations)

**Common use cases**:
- **Polynomial evaluation**: Sum terms like `Σ aᵢxⁱ`
- **Taylor series**: Approximate functions like exp, sin, cos
- **Iterative algorithms**: Refinement, convergence methods
- **Accumulation**: Sum multiple contributions
- **Nested computations**: Multi-dimensional operations

**Note**: Loop ranges must be compile-time constants (not function parameters).

📖 [See full documentation](docs/features/FOR_LOOP_SUPPORT.md) with 10 comprehensive test cases!

### ✅ **Assert and Pass Statements** 🆕 NEW!
Python's `assert` and `pass` statements work perfectly! Add input validation and clean code structure:

```python
import tangent

# Input validation with assert
def safe_sqrt(x):
    assert x >= 0, "Input must be non-negative"
    assert x < 1000, "Input too large"
    return x ** 0.5

# Placeholder with pass
def partial_implementation(x):
    if x < 0:
        pass  # TODO: Handle negative case
    return x ** 2

# Combined usage
def validated_processing(x):
    assert x > 0 and x < 100, "x must be in (0, 100)"

    if x < 10:
        pass  # Small values - no adjustment
        result = x
    else:
        result = x / 10.0

    return result ** 2

# All work seamlessly!
df_sqrt = tangent.grad(safe_sqrt)
df_partial = tangent.grad(partial_implementation)
df_validated = tangent.grad(validated_processing)
```

**Supported patterns**:
- ✅ Simple assertions: `assert condition`
- ✅ Assertions with messages: `assert condition, "message"`
- ✅ Complex conditions: `assert x > 0 and x < 10`
- ✅ NumPy in assertions: `assert x >= np.min(arr)`
- ✅ Assertions in loops and conditionals
- ✅ Pass as no-op placeholder
- ✅ Pass in if/elif/else branches
- ✅ Pass in loops

**Common use cases**:
- **Input validation**: Check preconditions and ranges
- **Domain checks**: Ensure mathematical validity (e.g., positive for log)
- **Numerical stability**: Validate intermediate values
- **Debug assertions**: Runtime sanity checks
- **Code structure**: Clean placeholder with pass
- **Empty branches**: Explicit no-ops in conditionals

📖 [See full documentation](docs/features/ASSERT_PASS_SUPPORT.md) with 12 comprehensive test cases!

### ✅ **While Loop Support** 🆕 NEW!
Python's `while` loops are fully supported! Write convergence algorithms and iterative methods naturally:

```python
import tangent

# Newton's method iteration
def newton_sqrt(x, iterations=5):
    estimate = x / 2.0
    i = 0
    while i < iterations:
        estimate = 0.5 * (estimate + x / estimate)
        i += 1
    return estimate

# Value-based termination
def accumulate_until(x):
    result = x
    iterations = 0
    max_iterations = 100  # Safety limit
    while result < 50.0 and iterations < max_iterations:
        result += x * 0.1
        iterations += 1
    return result

# Nested while loops
def nested_iteration(x):
    result = 0.0
    i = 0
    while i < 3:
        j = 0
        while j < 3:
            result += x
            j += 1
        i += 1
    return result

# All work seamlessly!
df_newton = tangent.grad(newton_sqrt)
df_accum = tangent.grad(accumulate_until)
df_nested = tangent.grad(nested_iteration)
```

**Supported patterns**:
- ✅ Counter-based iteration: `while i < n`
- ✅ Value-based termination: `while result < threshold`
- ✅ Complex conditions: `while i < n and result < max`
- ✅ Nested while loops
- ✅ Conditionals inside while
- ✅ NumPy operations in loops
- ✅ Multiple variable updates
- ✅ Empty loops (zero iterations)

**Common use cases**:
- **Newton's method**: Iterative root finding
- **Gradient descent**: Multiple optimization steps
- **Convergence algorithms**: Iterate until criteria met
- **Power series**: Sum until desired accuracy
- **Accumulation**: Build up until threshold

**Note**: `break` and `continue` statements are not supported. Always include safety iteration limits to prevent infinite loops.

📖 [See full documentation](docs/features/WHILE_LOOP_SUPPORT.md) with 12 comprehensive test cases!

### ✅ **Closure and Captured Variable Support** 🆕
Factory functions and closures now work seamlessly! Create parameterized functions using the factory pattern:

```python
import tangent

# Factory function returning a closure
def make_loss(target):
    def loss(prediction):
        return (prediction - target) ** 2  # Captures 'target'
    return loss

# Create loss functions for different targets
loss_5 = make_loss(5.0)
loss_10 = make_loss(10.0)

# Compute gradients - closures work perfectly!
dloss_5 = tangent.grad(loss_5)
dloss_10 = tangent.grad(loss_10)

gradient_5 = dloss_5(4.0)   # ✅ Works! Returns: -2.0
gradient_10 = dloss_10(8.0) # ✅ Works! Returns: -4.0
```

**Supported patterns**:
- ✅ Factory functions with captured variables
- ✅ Nested closures (closure of closure)
- ✅ Lambdas with captured variables
- ✅ NumPy arrays in closures
- ✅ Multiple captured variables
- ✅ Mixed captures (parameters + locals)

**Real-world use cases**:
- **Loss factories**: `make_mse_loss(target)` for different targets
- **Optimizer builders**: `make_optimizer(learning_rate, decay)`
- **Activation factories**: `make_leaky_relu(alpha)`
- **Regularization factories**: `make_l2_loss(lambda_reg)`

📖 [See full documentation](docs/features/CLOSURE_SUPPORT_COMPLETE.md) with 8 comprehensive test cases!

### ✅ **Extended NumPy Support** (27 new gradient definitions) 🆕
Comprehensive NumPy gradient coverage bringing it to near-parity with JAX:
- **Element-wise**: abs, square, reciprocal, negative
- **Logarithmic**: log10, log2, log1p, expm1
- **Reductions**: min, max, prod
- **Linear Algebra**: matmul, linalg.inv, outer, trace
- **Shape Operations**: squeeze, expand_dims, concatenate, stack
- **Comparison**: minimum, clip, where
- **Utilities**: sign, floor, ceil
- **Statistics**: var, std

📚 [See examples](examples/numpy_extended/) with 8 real-world use cases and comprehensive tests!

### ✅ **JAX Integration** (54 gradient definitions)
Full support for Google's JAX with comprehensive gradient definitions for:
- Neural network activations (ReLU, Sigmoid, ELU, Leaky ReLU, Softplus)
- Math functions (exp, log, sqrt, sin, cos, tanh, power)
- Linear algebra (dot, matmul)
- Reductions (sum, mean, max)
- Element-wise operations (maximum, minimum, negative)
- Broadcasting operations

### ✅ **Extended TensorFlow Support** (25 new gradient definitions) 🆕
Comprehensive TensorFlow 2.x gradient coverage bringing it to parity with NumPy:
- **Element-wise**: abs, square, sqrt, sign, floor, ceil, round, reciprocal, minimum, clip_by_value
- **Logarithmic**: log10, log2, log1p, expm1
- **Reductions**: reduce_min, reduce_prod
- **Trigonometric**: sin, cos, tan, asin, acos, atan
- **Neural Networks**: relu, sigmoid, softmax
- **Linear Algebra**: linalg.inv, linalg.trace, transpose
- **Shape Operations**: concat, stack

📊 **52+ operations** (up from 27) - now at parity with NumPy!

### ✅ **Visualization Tools** 🎨
**NEW!** Educational visualization suite for understanding autodiff:

<table>
<tr>
<td width="50%">

**Computation Graph**
```python
import tangent
import matplotlib.pyplot as plt

def f(x):
    y = x * x
    z = y + x
    return z

fig = tangent.visualize(f, mode='graph')
plt.show()
```

Shows function structure as a directed graph with:
- 🔵 Blue: Input nodes
- 🟢 Green: Operations
- 🔴 Red: Output nodes

![Computation Graph](assets/computation_graph.png)

</td>
<td width="50%">

**Gradient Flow**
```python
def f(x):
    return x * x + 2.0 * x + 1.0

fig = tangent.visualize(
    f,
    mode='flow',
    inputs=(2.0,)
)
plt.show()
```

Displays forward and backward passes:
- ⬆️ Forward: Function evaluation
- ⬇️ Backward: Gradient propagation
- Shows actual numerical values

![Gradient Flow](assets/gradient_flow.png)

</td>
</tr>
<tr>
<td width="50%">

**Gradient Comparison**
```python
import numpy as np

def f(x):
    return np.sum(x**3 - 2*x**2 + x)

fig = tangent.compare_gradients(
    f,
    (np.array([1.0, 2.0, 3.0]),)
)
plt.show()
```

Compares autodiff vs numerical:
- 📊 Side-by-side bar charts
- ✅ Error quantification
- Educational validation

![Gradient Comparison](assets/gradient_comparison.png)

</td>
<td width="50%">

**Code Inspection**
```python
def f(x):
    y = x * x
    z = y + x
    return z

tangent.show_gradient_code(f)
```

Pretty-prints:
- Original function code
- Generated gradient function
- Formatted with headers
- Easy to understand

```
GRADIENT CODE FOR: f
────────────────────────────────────────
ORIGINAL FUNCTION:
def f(x):
    y = x * x
    z = y + x
    return z

GENERATED GRADIENT FUNCTION:
def dfdx(x, bz=1.0):
    y = x * x
    z = y + x
    # Backward pass
    by = bz
    bx = by * x + by * x + bz
    return bx
```

</td>
</tr>
</table>

### ✅ **Comprehensive Testing**
- **111 comprehensive tests** across all features
- Classes: 14 tests (100% passing) ✅
- Inheritance: 12 tests (100% passing) ✅ NEW!
- JAX: 34 tests
- TensorFlow: 22 tests
- Visualization: 28 tests

### ✅ **Enhanced Error Messages**
Clear, helpful error messages with suggestions for fixes

### ✅ **Function Caching**
Automatic caching with 1000x+ speedup for repeated gradient calls

---

## 🚀 Quick Start

### Installation

```bash
# Install from GitHub
pip install git+https://github.com/pedronahum/tangent.git

# With JAX support
pip install git+https://github.com/pedronahum/tangent.git jax jaxlib

# With TensorFlow support
pip install git+https://github.com/pedronahum/tangent.git tensorflow

# With visualization tools
pip install git+https://github.com/pedronahum/tangent.git matplotlib networkx

# Full installation (recommended)
pip install git+https://github.com/pedronahum/tangent.git jax jaxlib tensorflow matplotlib networkx
```

### Basic Usage

```python
import tangent
import numpy as np

# Define your function
def f(x):
    return x ** 3 - 2 * x ** 2 + 3 * x - 1

# Get the gradient function
df = tangent.grad(f)

# Compute gradient at x=2
gradient = df(2.0)
print(f"f'(2) = {gradient}")
```

**Output:**
```
f'(2) = 11.0
```

### Multi-Backend Support

<table>
<tr>
<td width="33%">

**NumPy**
```python
import numpy as np
import tangent

def f(x):
    return np.sum(x ** 2)

df = tangent.grad(f)
x = np.array([1, 2, 3])
grad = df(x)
print(grad)
```

**Output:**
```
[2 4 6]
```

</td>
<td width="33%">

**JAX**
```python
import jax.numpy as jnp
import jax
import tangent

def f(x):
    return jnp.sum(
        jax.nn.relu(x)
    )

df = tangent.grad(f)
x = jnp.array([-1, 0, 1])
grad = df(x)
print(grad)
```

**Output:**
```
[0. 0. 1.]
```

</td>
<td width="33%">

**TensorFlow**
```python
import tensorflow as tf
import tangent

def f(x):
    return tf.reduce_sum(
        tf.tanh(x)
    )

df = tangent.grad(f)
x = tf.constant([0.0, 1.0, 2.0])
grad = df(x)
print(grad.numpy())
```

**Output:**
```
[1.         0.41997433 0.07065082]
```

</td>
</tr>
</table>

---

## 📚 Interactive Tutorial

We've created a comprehensive Jupyter notebook tutorial that covers everything:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/tangent_tutorial.ipynb)

**Contents:**
1. Installation & Setup
2. Basic Concepts - Understanding source-to-source autodiff
3. NumPy Integration - Vector and matrix operations
4. TensorFlow 2.x Integration - Deep learning workflows
5. JAX Integration - High-performance computing
6. Advanced Features - Multiple gradients, result preservation
7. **Visualization & Debugging** - Interactive tools (NEW!)
8. Real-World Examples - Linear regression, logistic regression, neural networks

---

## 🎓 Educational Features

### Visualize How Autodiff Works

```python
import tangent
import matplotlib.pyplot as plt

# Define a polynomial
def polynomial(x):
    y = x * x
    z = y * x
    w = 2.0 * y
    return z - w + 3.0 * x - 1.0

# Visualize the computation graph
fig = tangent.visualize(polynomial, mode='graph')
plt.savefig('computation_graph.png', dpi=150)
plt.show()
```

This shows you:
- How your function is decomposed into operations
- The flow of data through your computation
- Dependencies between variables

### Understand Gradient Flow

```python
# Visualize how gradients propagate backward
fig = tangent.visualize(polynomial, mode='flow', inputs=(2.0,))
plt.show()
```

See:
- Forward pass with actual values
- Backward pass with gradient values
- Step-by-step gradient computation

### Verify Your Gradients

```python
import numpy as np

def f(x):
    return np.sum(x ** 3 - 2 * x ** 2 + x)

x = np.array([1.0, 2.0, 3.0])

# Compare autodiff vs numerical gradients
fig = tangent.compare_gradients(f, (x,))
plt.show()
```

Perfect for:
- Debugging gradient implementations
- Teaching autodiff concepts
- Validating complex derivatives

---

## 🔬 Advanced Features

### Multiple Gradients

Compute gradients with respect to multiple arguments:

```python
def f(x, y):
    return x * x * y + x * y * y

# Gradients w.r.t. both x and y
df = tangent.grad(f, wrt=(0, 1))
grad_x, grad_y = df(2.0, 3.0)

print(f"∂f/∂x = {grad_x}")  # Expected: 2xy + y² = 21
print(f"∂f/∂y = {grad_y}")  # Expected: x² + 2xy = 16
```

**Output:**
```
∂f/∂x = 21.0
∂f/∂y = 16.0
```

### Preserve Results

Get both the function value and gradient:

```python
def f(x):
    return np.sum(x ** 2)

df = tangent.grad(f, preserve_result=True)
gradient, result = df(np.array([1.0, 2.0, 3.0]))

print(f"f(x) = {result}")
print(f"∇f(x) = {gradient}")
```

**Output:**
```
f(x) = 14.0
∇f(x) = [2. 4. 6.]
```

### Inspect Generated Code

See exactly what Tangent generates:

```python
def f(x):
    y = x * x
    z = y + x
    return z

tangent.show_gradient_code(f)
```

Output:
```
================================================================================
GRADIENT CODE FOR: f
================================================================================

ORIGINAL FUNCTION:
--------------------------------------------------------------------------------
def f(x):
    y = x * x
    z = y + x
    return z

GENERATED GRADIENT FUNCTION:
--------------------------------------------------------------------------------
def dfdx(x, bz=1.0):
    # Forward pass
    y = x * x
    z = y + x

    # Backward pass
    by = bz
    bx = by * x + by * x + bz
    return bx
================================================================================
```

### Performance: Automatic Caching

Tangent automatically caches compiled gradient functions:

```python
import tangent
import time

def expensive_function(x):
    return x ** 10

# First call: compiles (~20-100ms)
start = time.time()
df = tangent.grad(expensive_function)
first_time = time.time() - start

# Subsequent calls: cached (~0.04ms)
start = time.time()
df = tangent.grad(expensive_function)
cached_time = time.time() - start

print(f"First call:  {first_time*1000:.2f}ms")
print(f"Cached call: {cached_time*1000:.2f}ms")
print(f"Speedup:     {first_time/cached_time:.0f}x")

# Check cache stats
stats = tangent.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

Benchmarks show:
- **1000x+ speedup** for cached retrieval
- **87x+ speedup** for 100 repeated calls
- **99% cache hit rate** in typical usage

---

## 📖 Examples

### 📓 Interactive Notebooks

**Building Energy Optimization Tutorial** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/tangent/blob/master/examples/Building_Energy_Optimization_with_Tangent.ipynb)

A comprehensive, pedagogical notebook demonstrating:
- ✅ **Real-world application**: Building thermal dynamics simulation
- ✅ **Generated code inspection**: See exactly what Tangent creates
- ✅ **Optimization levels**: Compare unoptimized vs optimized gradients
- ✅ **Performance benchmarks**: Tangent vs numerical differentiation vs PyTorch
- ✅ **Gradient-based optimization**: Use gradients to minimize energy costs
- ✅ **Visualization**: Beautiful plots of gradients, schedules, and convergence

Perfect for teaching automatic differentiation or understanding how Tangent works under the hood!

**Other Notebooks:**
- [Tangent Tutorial](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/tangent_tutorial.ipynb) - General introduction to Tangent

---

### Example 1: Linear Regression

```python
import tangent
import numpy as np

# Generate data
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.5

# Loss function
def mse_loss(w, b):
    predictions = w * X + b
    return np.mean((predictions - y) ** 2)

# Compute gradients
dmse_dw = tangent.grad(mse_loss, wrt=(0,))
dmse_db = tangent.grad(mse_loss, wrt=(1,))

# Gradient descent
w, b = 0.0, 0.0
learning_rate = 0.1

for epoch in range(50):
    grad_w = dmse_dw(w, b)
    grad_b = dmse_db(w, b)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if epoch % 10 == 0:
        loss = mse_loss(w, b)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")
```

### Example 2: Neural Network with JAX

```python
import tangent
import jax.numpy as jnp
import jax

def neural_network(W1, b1, W2, b2, X, y):
    """Two-layer neural network."""
    hidden = jax.nn.relu(jnp.dot(X, W1) + b1)
    output = jax.nn.sigmoid(jnp.dot(hidden, W2) + b2)
    loss = -jnp.mean(y * jnp.log(output) + (1 - y) * jnp.log(1 - output))
    return loss

# Compute gradients w.r.t. all parameters
dnn_dW1 = tangent.grad(neural_network, wrt=(0,))
dnn_db1 = tangent.grad(neural_network, wrt=(1,))
dnn_dW2 = tangent.grad(neural_network, wrt=(2,))
dnn_db2 = tangent.grad(neural_network, wrt=(3,))

# Training loop
for epoch in range(100):
    grad_W1 = dnn_dW1(W1, b1, W2, b2, X, y)
    grad_b1 = dnn_db1(W1, b1, W2, b2, X, y)
    grad_W2 = dnn_dW2(W1, b1, W2, b2, X, y)
    grad_b2 = dnn_db2(W1, b1, W2, b2, X, y)

    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
```

### Example 3: Visualization Demo

Run the complete visualization demo:

```bash
python examples/demo_visualization.py
```

This generates 6 PNG visualizations showing:
1. Computation graphs
2. Gradient flow diagrams
3. Multivariate gradient flow
4. Autodiff vs numerical comparison
5. Vector function gradients
6. Matrix operation gradients

---

## 🧪 Testing

All new features are thoroughly tested:

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_classes.py       # Class support (14 tests)
pytest tests/test_inheritance.py   # Inheritance support (12 tests)
pytest tests/test_jax.py          # JAX integration (34 tests)
pytest tests/test_tensorflow.py   # TensorFlow (22 tests)
pytest tests/test_visualization.py # Visualization (28 tests)

# Run with coverage
pytest tests/ --cov=tangent --cov-report=html
```

**Test Coverage**: 111 comprehensive tests across all features

---

## 📊 Repository Structure

```
tangent/
├── tangent/                     # Core library
│   ├── __init__.py
│   ├── grad_util.py            # Main autodiff engine
│   ├── class_desugar.py        # Class method inlining (NEW!)
│   ├── jax_extensions.py       # JAX support (51 gradients)
│   ├── tf_extensions.py        # TensorFlow 2.x support
│   ├── visualization.py        # Visualization tools
│   ├── function_cache.py       # Caching system
│   └── ...
├── tests/                       # Comprehensive test suite
│   ├── test_classes.py         # 14 class support tests
│   ├── test_inheritance.py     # 12 inheritance tests (NEW!)
│   ├── test_jax.py             # 34 JAX tests
│   ├── test_tensorflow.py      # 22 TensorFlow tests
│   ├── test_visualization.py   # 28 visualization tests
│   └── ...
├── examples/                    # Example scripts
│   ├── test_jax_basic.py       # JAX examples
│   ├── test_tf2_basic.py       # TensorFlow examples
│   ├── demo_visualization.py   # Visualization demos
│   └── demo_error_messages.py
├── notebooks/                   # Interactive tutorials
│   ├── tangent_tutorial.ipynb  # Comprehensive Colab notebook
│   └── README.md
├── benchmarks/                  # Performance benchmarks
│   └── benchmark_cache.py
└── docs/                        # Documentation
    ├── features/                   # Feature documentation
    │   ├── CLASS_SUPPORT_COMPLETE.md
    │   ├── INHERITANCE_SUPPORT_COMPLETE.md
    │   ├── LAMBDA_SUPPORT_COMPLETE.md
    │   ├── PYTHON_FEATURE_SUPPORT.md
    │   └── ... (all feature docs)
    └── development/                # Development & planning docs
        ├── ROADMAP_TO_GREATNESS.md
        └── ... (planning documents)
```

---

## 📚 Documentation

Comprehensive documentation organized by topic:

### 🏆 **Performance Benchmarks**
- **[Framework Comparison](docs/benchmarks/FRAMEWORK_COMPARISON.md)** - Tangent vs TensorFlow vs PyTorch
- **[Benchmark Summary](docs/benchmarks/BENCHMARK_SUMMARY.md)** - Executive summary
- **[Building Simulation](docs/benchmarks/BUILDING_SIMULATION_BENCHMARK.md)** - Real-world thermal simulation
- **[Correctness Verification](docs/benchmarks/CORRECTNESS_VERIFICATION.md)** - Mathematical validation

**Key Results**:
- ✅ **Matches TensorFlow** - 4.300ms vs 4.315ms (essentially tied!)
- ✅ **1.53× faster than PyTorch** for gradients
- ✅ **10.80× faster forward pass** than PyTorch
- ✅ **Mathematically correct** - verified to 7 significant figures

### ⚡ **Optimizations**

Tangent includes a **7-pass optimization pipeline** producing production-grade gradient code. See the **[Advanced Optimization Pipeline](#-advanced-optimization-pipeline)** section for full details.

**Quick Links**:
- **[Symbolic Optimizations](docs/optimizations/SYMBOLIC_OPTIMIZATIONS_COMPLETE.md)** - CSE & algebraic simplification
- **[Strength Reduction](docs/optimizations/STRENGTH_REDUCTION_COMPLETE.md)** - Power/division optimization
- **[Performance Analysis](docs/optimizations/PERFORMANCE_ANALYSIS.md)** - Optimization impact
- **[Future Improvements](docs/benchmarks/PERFORMANCE_IMPROVEMENT_STRATEGIES.md)** - Roadmap for 2-5× more speedup

**Key Results**: 2.35× speedup with full pipeline | 1.95× with DCE alone | 5-25× per strength reduction

### 📖 **Complete Index**
**[→ Full Documentation Index](docs/INDEX.md)** - All docs organized and searchable

---

## 🤝 Contributing

Contributions are welcome! This is an actively maintained fork with regular updates.

**Areas for contribution:**
- Additional gradient definitions for JAX/TensorFlow operations
- More visualization tools (3D plots, animations)
- Performance optimizations
- Documentation improvements
- Bug fixes

**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Submit a pull request

---

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

Original work Copyright 2017 Google Inc.
Modified work Copyright 2024 Pedro Nahum

---

## 🙏 Acknowledgments

- Original Tangent library by Google Research
- JAX team at Google for the excellent numerical computing library
- TensorFlow team for TensorFlow 2.x
- The Python scientific computing community

---

## 📬 Contact

- **Repository**: [github.com/pedronahum/tangent](https://github.com/pedronahum/tangent)
- **Issues**: [github.com/pedronahum/tangent/issues](https://github.com/pedronahum/tangent/issues)
- **Author**: [@pedronahum](https://github.com/pedronahum)

---

## 🌟 Star History

If you find Tangent useful, please consider starring the repository!

---

**Built with ❤️ for the machine learning and scientific computing communities**
