"""Demo script showing improved error messages in Tangent.

This script demonstrates the new user-friendly error messages with:
- Source code context
- Helpful suggestions
- Clear formatting
"""

import tangent
import sys


print("=" * 80)
print("Tangent Error Message Demo")
print("=" * 80)
print()

# Demo 1: Source code not available (REPL function)
print("1. Demonstrating SourceCodeNotAvailableError")
print("-" * 80)
print("Trying to differentiate a function defined in exec()...")
print()

try:
    # Create a function dynamically (source not available)
    exec("def dynamic_func(x): return x * x")
    df = tangent.grad(dynamic_func)
except Exception as e:
    print(e)
    print()

# Demo 2: Forward mode not implemented
print("\n2. Demonstrating ForwardNotImplementedError")
print("-" * 80)
print("Trying to use forward mode on a function without forward gradient...")
print()

def my_custom_function(x):
    """A custom function without forward mode definition."""
    return x ** 5

try:
    # Try to use forward mode (not implemented for custom functions by default)
    from tangent.errors import ForwardNotImplementedError
    raise ForwardNotImplementedError(my_custom_function)
except Exception as e:
    print(e)
    print()

# Demo 3: Reverse mode not implemented
print("\n3. Demonstrating ReverseNotImplementedError")
print("-" * 80)
print("Trying to use a function without reverse gradient...")
print()

def another_custom_function(x):
    """Another custom function."""
    return x ** 7

try:
    from tangent.errors import ReverseNotImplementedError
    raise ReverseNotImplementedError(another_custom_function)
except Exception as e:
    print(e)
    print()

# Demo 4: Non-scalar output error
print("\n4. Demonstrating NonScalarOutputError")
print("-" * 80)
print("Trying to take gradient of vector-valued function...")
print()

try:
    from tangent.error_handlers import NonScalarOutputError
    import numpy as np
    raise NonScalarOutputError(output_shape=(3, 4))
except Exception as e:
    print(e)
    print()

# Demo 5: Type mismatch error
print("\n5. Demonstrating TypeMismatchError")
print("-" * 80)
print("Type mismatch during gradient computation...")
print()

try:
    from tangent.error_handlers import TypeMismatchError
    raise TypeMismatchError(
        expected_type='float64',
        actual_type='float32',
        context='matrix multiplication'
    )
except Exception as e:
    print(e)
    print()

# Demo 6: Gradient not found error
print("\n6. Demonstrating GradientNotFoundError")
print("-" * 80)
print("Custom function without gradient definition...")
print()

try:
    from tangent.error_handlers import GradientNotFoundError
    raise GradientNotFoundError(func_name='my_special_function')
except Exception as e:
    print(e)
    print()

print("=" * 80)
print("Demo Complete!")
print("=" * 80)
print()
print("Notice how each error provides:")
print("  ✓ Clear description of the problem")
print("  ✓ Helpful suggestions for fixing it")
print("  ✓ Code examples where applicable")
print("  ✓ Links to documentation")
print()
