# Enhanced Error Messages in Tangent

## Overview

Tangent now provides helpful, context-aware error messages when you use unsupported Python features or encounter common issues. Each error includes:

- **Clear description** of what went wrong
- **💡 Suggestions** with workarounds and alternatives
- **✅/❌ Examples** showing correct and incorrect usage
- **📖 Documentation links** for more information

## Improved Error Categories

### 1. Unsupported Python Features

When you use Python features that Tangent doesn't support, you'll get helpful suggestions:

#### Dictionary Comprehensions

**Error:**
```
TangentParseError: Dictionary Comprehensions are not supported

💡 Suggestion:
Dictionary comprehensions are not supported.

Workaround:
  ❌ d = {k: x ** i for i, k in enumerate(['a', 'b'])}

  ✅ Pass dict as parameter:
     def compute(x, config={'a': 1, 'b': 2}):
         return x * config['a']

  ✅ Use separate variables:
     a = x
     b = x ** 2

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#dictionaries
```

#### F-Strings

**Error:**
```
TangentParseError: F-Strings are not supported

💡 Suggestion:
F-strings are not yet supported in Tangent.

Workaround:
  ❌ msg = f"Value is {x}"
  ✅ msg = "Value is " + str(x)  # Use string concatenation
  ✅ msg = "Value is %s" % x      # Use % formatting
```

#### Try/Except Blocks

**Error:**
```
TangentParseError: Try/Finally blocks are not supported

💡 Suggestion:
Try/except/finally blocks are not supported in Tangent.

Workarounds:
  1. Use assertions for validation:
     ✅ assert x != 0, "Division by zero"
        return 1.0 / x

  2. Use conditional statements:
     ✅ if abs(x) < 1e-10:
            return 0.0
        return 1.0 / x

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#exception-handling
```

#### Break/Continue Statements

**Error:**
```
TangentParseError: Break statements are not supported in strict mode

💡 Suggestion:
Break statements are not supported in Tangent loops.

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

📖 See: docs/features/PYTHON_FEATURE_SUPPORT.md#loop-control
```

#### In Operator

**Error:**
```
TangentParseError: In operator is not supported

💡 Suggestion:
The 'in' operator for membership testing is not supported.

Workarounds:
  ❌ if x in [1, 2, 3]:
         return x

  ✅ if x == 1 or x == 2 or x == 3:
         return x
```

### 2. Runtime Errors with Context

#### Dict Construction Bug

When you try to construct a dict with multiple keys containing differentiated values, Tangent now catches the resulting `NameError` and provides a clear explanation:

**Error:**
```
DictConstructionError:
================================================================================
Tangent Error: Dict Construction Bug
================================================================================

Multi-key dictionary construction with differentiated values is currently buggy.

The generated code contains undefined '_' placeholders, causing: NameError: name '_' is not defined

This is a known issue in Tangent. See: docs/bugs/DICT_CONSTRUCTION_BUG.md

💡 Workarounds:

1. **Pass dict as parameter (RECOMMENDED)**:
   ✅ def compute(x, config={'a': 1, 'b': 2}):
          return x * config['a'] + x * config['b']

   df = tangent.grad(compute)
   grad = df(5.0)  # Works perfectly!

2. **Use global dict**:
   ✅ CONFIG = {'a': 1, 'b': 2}

      def process(x):
          return x * CONFIG['a'] + x * CONFIG['b']

3. **Use separate variables**:
   ✅ def compute(x):
          a = x
          b = x ** 2
          return a + b

📖 Documentation: docs/features/PYTHON_FEATURE_SUPPORT.md#dictionaries
```

## Complete List of Enhanced Errors

| Feature | Error Type | Has Suggestion |
|---------|------------|----------------|
| F-Strings | TangentParseError | ✅ |
| Dictionary Comprehensions | TangentParseError | ✅ |
| Set Comprehensions | TangentParseError | ✅ |
| Generator Expressions | TangentParseError | ✅ |
| Sets | TangentParseError | ✅ |
| Try/Except/Finally | TangentParseError | ✅ |
| Break Statements | TangentParseError | ✅ |
| Continue Statements | TangentParseError | ✅ |
| In Operator | TangentParseError | ✅ |
| Not In Operator | TangentParseError | ✅ |
| Import Statements (in functions) | TangentParseError | ✅ |
| MatMult Operator (@) | TangentParseError | ✅ |
| Floor Div (//) | TangentParseError | ✅ |
| Bitwise Operators | TangentParseError | ✅ |
| Walrus Operator (:=) | TangentParseError | ✅ |
| Del Statement | TangentParseError | ✅ |
| Multi-key Dict Construction | DictConstructionError | ✅ |

## Implementation Details

### How It Works

1. **Feature Detection**: The `LanguageFence` class in `tangent/fence.py` walks the AST and detects unsupported features

2. **Suggestion Lookup**: When an unsupported feature is found, `tangent/error_suggestions.py` provides context-aware suggestions

3. **Error Enhancement**: The `_reject()` method automatically enhances error messages with suggestions

4. **Runtime Wrapping**: Gradient functions are wrapped to catch and enhance runtime errors (like the dict construction bug)

### Adding New Error Suggestions

To add suggestions for a new unsupported feature:

1. Edit `tangent/error_suggestions.py`
2. Add an entry to `UNSUPPORTED_FEATURE_SUGGESTIONS` dictionary:

```python
UNSUPPORTED_FEATURE_SUGGESTIONS = {
    'Feature Name': '''Feature description and explanation.

Workarounds:
  ❌ bad_example
  ✅ good_example

📖 See: docs/link
''',
}
```

3. The suggestion will automatically be included when that feature is rejected

## Examples

### Example 1: Trying to Use F-Strings

```python
import tangent

def compute(x):
    msg = f"Computing gradient for {x}"  # ❌ F-strings not supported
    return x ** 2

df = tangent.grad(compute)
```

**Output:**
```
TangentParseError: F-Strings are not supported

💡 Suggestion:
F-strings are not yet supported in Tangent.

Workaround:
  ❌ msg = f"Value is {x}"
  ✅ msg = "Value is " + str(x)
  ✅ msg = "Value is %s" % x

Note: String operations don't affect gradient computation.
```

### Example 2: Dict Construction Bug

```python
import tangent

def compute(x):
    d = {'a': x, 'b': x ** 2}  # ❌ Multi-key dict bug
    return d['a'] + d['b']

df = tangent.grad(compute)
result = df(2.0)  # Triggers enhanced error
```

**Output:**
```
DictConstructionError:
Multi-key dictionary construction with differentiated values is currently buggy.

💡 Workarounds:

1. Pass dict as parameter (RECOMMENDED):
   ✅ def compute(x, config={'a': 1, 'b': 2}):
          return x * config['a']

[... full suggestion ...]
```

## Testing

Run the error message demo to see all enhanced errors:

```bash
python examples/demo_error_messages.py
```

## Benefits

- **Faster debugging**: Users immediately understand what went wrong
- **Learning**: Users learn Tangent's limitations and best practices
- **Better UX**: Clear, actionable feedback instead of cryptic errors
- **Reduced support burden**: Users can self-solve common issues

## See Also

- [Python Feature Support](PYTHON_FEATURE_SUPPORT.md) - Complete feature matrix
- [Dict Construction Bug](../bugs/DICT_CONSTRUCTION_BUG.md) - Detailed bug report
- [Error Handlers](../../tangent/error_handlers.py) - Advanced error handling
- [Error Suggestions](../../tangent/error_suggestions.py) - Suggestion database
