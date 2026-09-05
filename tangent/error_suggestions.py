"""Helpful error suggestions for unsupported Python features in Tangent.

This module provides context-aware suggestions for common errors.
"""

# Dictionary mapping error patterns to helpful suggestions
UNSUPPORTED_FEATURE_SUGGESTIONS = {
    'Augmented assignment to an attribute': '''In-place update of an attribute
(e.g. obj.attr += x) is not supported: attribute gradients are not implemented.

Workarounds:
  ❌ obj.weight += x
  ✅ y = obj.weight + x        # use a new variable
  ✅ operate on whole arrays   # use NumPy arrays and subscript updates

Note: augmented assignment to a SUBSCRIPT (a[i] += x) IS supported, as is plain
augmented assignment on a variable (x += y).
''',

    'F-Strings': '''F-strings are not yet supported in Tangent.

Workaround:
  ❌ msg = f"Value is {x}"
  ✅ msg = "Value is " + str(x)  # Use string concatenation
  ✅ msg = "Value is %s" % x      # Use % formatting (limited support)

Note: String operations don't affect gradient computation, so this is purely syntactic.
''',

    'Dictionary Comprehensions': '''Dictionary comprehensions are not supported.

Workaround:
  ❌ d = {k: x ** i for i, k in enumerate(['a', 'b'])}

  ✅ Pass dict as parameter:
     def compute(x, config={'a': 1, 'b': 2}):
         return x * config['a']

  ✅ Use separate variables:
     a = x
     b = x ** 2
     # Use a, b directly instead of d['a'], d['b']

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#dictionaries
''',

    'Sets': '''Sets are not supported in Tangent.

Workaround:
  ❌ s = {1, 2, 3}
  ✅ Use lists: items = [1, 2, 3]
  ✅ Use tuples: items = (1, 2, 3)

Note: If you need set operations for control flow, consider restructuring your code.
''',

    'Set Comprehensions': '''Set comprehensions are not supported.

Workaround:
  ❌ s = {x ** 2 for x in range(10)}
  ✅ Use list comprehension: items = [x ** 2 for x in range(10)]
''',

    'Generator Expressions': '''Generator expressions are not supported.

Workaround:
  ❌ gen = (x ** 2 for x in range(10))
  ✅ Use list comprehension: items = [x ** 2 for x in range(10)]
  ✅ Use explicit loops with lists
''',

    'Try/Finally blocks': '''Try/except/finally blocks are not supported in Tangent.

Workarounds:
  1. Use assertions for validation:
     ✅ assert x != 0, "Division by zero"
        return 1.0 / x

  2. Use conditional statements:
     ✅ if abs(x) < 1e-10:
            return 0.0
        return 1.0 / x

  3. Handle exceptions outside differentiated function:
     ✅ try:
            result = differentiable_func(x)
        except ValueError:
            result = fallback_value

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#exception-handling
''',

    'Break statements': '''Break statements are not supported in Tangent loops.

Workarounds:
  1. Use while loop with condition:
     ❌ for i in range(10):
            result += x
            if result > 100:
                break

     ✅ i = 0
        while result <= 100 and i < 10:
            result += x
            i += 1

  2. Use conditional logic:
     ✅ for i in range(10):
            if result <= 100:
                result += x

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#loop-control
''',

    'Continue statements': '''Continue statements are not supported in Tangent loops.

Workarounds:
  1. Use conditional logic instead:
     ❌ for i in range(10):
            if i % 2 == 0:
                continue
            result += x * i

     ✅ for i in range(10):
            if i % 2 != 0:
                result += x * i

  2. Invert the condition to avoid continue
''',

    'In operator': '''The 'in' operator for membership testing is not supported.

Workarounds:
  1. For constant checks, use comparisons:
     ❌ if x in [1, 2, 3]:
            return x

     ✅ if x == 1 or x == 2 or x == 3:
            return x

  2. For control flow, restructure logic:
     ✅ Use conditional expressions based on actual values
''',

    'Not In operator': '''The 'not in' operator is not supported.

Workaround:
  ❌ if x not in [1, 2, 3]:
         return x ** 2

  ✅ if x != 1 and x != 2 and x != 3:
         return x ** 2
''',

    'Import statements': '''Import statements inside functions are not supported.

Workaround:
  ❌ def compute(x):
         import numpy as np
         return np.sin(x)

  ✅ import numpy as np  # Import at module level

     def compute(x):
         return np.sin(x)
''',

    'Import/From statements': '''Import/from statements inside functions are not supported.

Workaround:
  Place all imports at the module level (top of file).
''',

    'MatMult operator': '''The @ (matrix multiplication) operator is supported for backends that
register matmul gradients (NumPy, tinygrad, JAX, TensorFlow). If you see a
"No `@` (matmul) gradient registered" error, your backend has no registration.

Workaround:
  ❌ result = A @ B
  ✅ result = np.dot(A, B)
  ✅ result = np.matmul(A, B)   (or the backend's matmul function/method)
''',

    'Floor Div operator': '''The // (floor division) operator has limited support.

If you encounter issues:
  ❌ result = x // y
  ✅ result = np.floor(x / y)
''',

    'Bitwise Or operator': '''Bitwise operators are not supported.

For logical operations:
  ❌ result = a | b
  ✅ result = a or b  # For boolean logic
''',

    'Bitwise And operator': '''Bitwise operators are not supported.

For logical operations:
  ❌ result = a & b
  ✅ result = a and b  # For boolean logic
''',

    'Bitwise Xor operator': '''Bitwise XOR operator is not supported.

Workaround: Use equivalent logical operations if applicable.
''',

    'Left Shift operator': '''Bitwise shift operators are not supported.

Workaround:
  ❌ result = x << 2
  ✅ result = x * (2 ** 2)  # Equivalent to left shift
''',

    'Right Shift operator': '''Bitwise shift operators are not supported.

Workaround:
  ❌ result = x >> 2
  ✅ result = x // (2 ** 2)  # Equivalent to right shift
''',

    'Walrus operator': '''The walrus operator (:=) is not supported.

Workaround:
  ❌ if (y := x ** 2) > 10:
         return y

  ✅ y = x ** 2
     if y > 10:
         return y
''',

    'Delete statements': '''Del statements are not supported.

Workaround:
  Variables in Tangent functions should not be deleted.
  Simply don't use the variable after a certain point.
''',

    'Deleting variables': '''Deleting variables is not supported.

Workaround:
  Simply stop using the variable instead of deleting it.
''',

    'Raise statements': '''Raise statements are not supported inside
differentiated functions.

Workaround:
  Validate inputs with assertions (supported) and keep exception handling
  outside the differentiated function.
  ❌ if x < 0: raise ValueError("negative")
  ✅ assert x >= 0, "x must be non-negative"
''',

    'Assignment expressions (the walrus operator ":=")': '''The walrus operator
(y := expr) binds a name in expression position, and that binding is not
tracked as an active intermediate, so the gradient of any branch that uses it
is silently dropped.

Workaround:
  Assign in a normal statement instead.
  ❌ if (y := x * 2) > 1: return y
  ✅ y = x * 2
     if y > 1:
         return y
''',

    'Variadic positional arguments (*args)': '''Variadic positional arguments
(*args) are not supported because activity analysis indexes arguments by
position, which is ill-defined for a variable-length pack.

Workaround:
  Use a fixed number of positional arguments, or pass a NumPy array / tuple
  and index into it.
  ❌ def f(*xs): return xs[0] ** 2
  ✅ def f(x): return x ** 2
''',

    'Variadic keyword arguments (**kwargs)': '''Variadic keyword arguments
(**kwargs) are not supported.

Workaround:
  Declare the keyword arguments you need explicitly (defaults are supported).
  ❌ def f(x, **kw): return x ** kw['p']
  ✅ def f(x, p=2.0): return x ** p
''',

    'Nested function definitions': '''Nested function definitions (a def inside
another def), closures, and recursion are not supported. The reverse pass
requires a single function with exactly one (normalized) return.

Workaround:
  ✅ Lambdas assigned to a variable and called are inlined and DO work:
       sq = lambda y: y ** 2
       return sq(x)
  ✅ Hoist helpers to module level and call them (they are differentiated as
     part of the call tree):
       def _square(y): return y ** 2
       def f(x): return _square(x)
''',

}


def get_suggestion(feature_name):
    """Get a helpful suggestion for an unsupported feature.

    Args:
        feature_name: Name of the unsupported feature

    Returns:
        A helpful suggestion string, or None if no specific suggestion exists
    """
    return UNSUPPORTED_FEATURE_SUGGESTIONS.get(feature_name)


def format_error_with_suggestion(feature_name, original_message):
    """Format an error message with a helpful suggestion.

    Args:
        feature_name: Name of the unsupported feature
        original_message: The original error message

    Returns:
        Enhanced error message with suggestion
    """
    suggestion = get_suggestion(feature_name)

    if suggestion:
        return f'''{original_message}

💡 Suggestion:
{suggestion}'''
    else:
        return original_message
