# Notebook Test Suite

This directory contains test files for validating the Tangent tutorial notebook.

## Test Files

### Main Test Suite
- **`test_notebook_cells.py`** - Comprehensive test suite covering all 20 executable notebook cells
  - Tests NumPy, TensorFlow, and JAX backends
  - Tests all Section 9 advanced Python features (lambdas, classes, inheritance, control flow)
  - Run with: `pytest test_notebook_cells.py -v`

### Debug/Investigation Files
- **`test_class_issue.py`** - Demonstrates class definition scope requirements
- **`test_tf_subscript.py`** - Tests TensorFlow tensor subscripting with gradients
- **`inspect_generated_code.py`** - Utility for inspecting generated gradient code

## Running Tests

### Run All Notebook Tests
```bash
cd /Users/pedro/programming/python/tangent
pytest tests/notebook/test_notebook_cells.py -v
```

### Run Specific Test
```bash
pytest tests/notebook/test_notebook_cells.py::test_cell_55_classes -v
```

### Run with Coverage
```bash
pytest tests/notebook/ --cov=tangent --cov-report=html
```

## Test Coverage

The test suite validates:

### Section 2: Basic Concepts (2 tests)
- ✅ Square function gradient
- ✅ Polynomial gradient with code inspection

### Section 3: NumPy Integration (3 tests)
- ✅ Vector norm squared gradient
- ✅ Matrix-vector operation gradient
- ✅ Sigmoid element-wise gradient

### Section 4: TensorFlow Integration (3 tests)
- ✅ TF import and version check
- ✅ TF quadratic gradient
- ✅ TF neural network layer gradient (with subscripting)

### Section 5: JAX Integration (3 tests)
- ✅ JAX import and version check
- ✅ JAX polynomial gradient
- ✅ JAX ReLU network gradient

### Section 6: Advanced Features (2 tests)
- ✅ Multivariate gradients
- ✅ Preserve result feature

### Section 9: Advanced Python Features (7 tests)
- ✅ Lambda functions with closures
- ✅ User-defined classes
- ✅ Class inheritance with super()
- ✅ For loops
- ✅ While loops
- ✅ Ternary operator
- ✅ OOP neural network

**Total: 20 test cases**

## Test Results

All tests passing (as of 2025-11-04):
```
✓ Passed:  20/20
✗ Failed:  0/20
⊘ Skipped: 0/20
Total:     100%
```

## Important Notes

### Class Definitions
Classes must be defined at **module level** for Tangent to access them:

**✅ Works** (module level):
```python
class MyClass:
    pass

def my_function(x):
    obj = MyClass()
    return obj.method(x)

grad_fn = tangent.grad(my_function)  # ✓ Works!
```

**✗ Fails** (local scope):
```python
def test_function():
    class MyClass:  # ← Local to function
        pass

    def my_function(x):
        obj = MyClass()  # ← Can't access!
        return obj.method(x)

    grad_fn = tangent.grad(my_function)  # ✗ Fails!
```

### Notebook vs Python Files
In Jupyter/Colab notebooks, classes defined in a cell become module-level for that cell's scope. This is why the notebook cells work correctly even though classes and usage code are in the same cell.

## Backend Compatibility

All tests work with:
- ✅ NumPy (100% compatible)
- ✅ JAX (100% compatible after fixes)
- ✅ TensorFlow (100% compatible after fixes)

### Fixes Applied
1. **Signature binding** - Module-prefixed functions (`jnp.sum`, `tf.reshape`)
2. **Array subscripting** - Immutable tensor updates (JAX, TensorFlow)
3. **Type casting** - JAX ReLU gradient with scalars

See `BACKEND_FIXES_COMPLETE.md` in the root directory for details.

## Continuous Integration

These tests should be run:
- Before any notebook changes
- After any Tangent core changes
- As part of CI/CD pipeline
- Before releases

## Contributing

When adding new notebook cells:
1. Add corresponding test case to `test_notebook_cells.py`
2. Follow the existing test pattern
3. Ensure test is independent and can run in isolation
4. Update this README with new test count

## Questions?

See the main test suite at `tests/test_notebook_examples.py` which uses pytest classes for better organization.
