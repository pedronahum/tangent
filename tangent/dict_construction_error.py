"""Special error handler for dict construction bug."""


class DictConstructionError(Exception):
    """Error raised when multi-key dict construction fails."""

    def __init__(self):
        message = '''
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

      df = tangent.grad(process)

3. **Use separate variables**:
   ✅ def compute(x):
          a = x
          b = x ** 2
          return a + b  # Instead of d['a'] + d['b']

      df = tangent.grad(compute)

📖 Documentation: docs/features/PYTHON_FEATURE_SUPPORT.md#dictionaries

================================================================================
'''
        super(DictConstructionError, self).__init__(message)


def is_dict_construction_error(error):
    """Check if an error is the dict construction bug.

    Args:
        error: The exception to check

    Returns:
        True if this is the dict construction NameError
    """
    if isinstance(error, NameError):
        error_msg = str(error)
        if "name '_' is not defined" in error_msg:
            return True
    return False


def enhance_dict_construction_error(error):
    """Wrap a dict construction NameError with helpful message.

    Args:
        error: The original NameError

    Returns:
        DictConstructionError with helpful suggestions
    """
    if is_dict_construction_error(error):
        return DictConstructionError()
    return error
