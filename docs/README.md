# Tangent Documentation

This directory contains all documentation for the Tangent automatic differentiation library.

## Directory Structure

### `features/` - Feature Documentation

Comprehensive guides for each supported Python feature:

**Language Features:**
- [Python Feature Support Guide](features/PYTHON_FEATURE_SUPPORT.md) - Complete reference of supported Python features
- [Lambda Functions](features/LAMBDA_SUPPORT_COMPLETE.md) - Lambda and closure support
- [Class Support](features/CLASS_SUPPORT_COMPLETE.md) - User-defined classes with method inlining
- [Inheritance Support](features/INHERITANCE_SUPPORT_COMPLETE.md) - Class inheritance and `super()`
- [Conditional Expressions](features/CONDITIONAL_EXPRESSION_SUPPORT.md) - Ternary operator support
- [Boolean Operators](features/BOOLEAN_OPERATOR_SUPPORT.md) - `and`, `or`, `not` operators
- [Augmented Assignment](features/AUGMENTED_ASSIGNMENT_SUPPORT.md) - `+=`, `-=`, `*=`, etc.
- [For Loops](features/FOR_LOOP_SUPPORT.md) - `for` loop with `range()`
- [While Loops](features/WHILE_LOOP_SUPPORT.md) - `while` loop support
- [Assert and Pass](features/ASSERT_PASS_SUPPORT.md) - Statement support
- [List Comprehensions](features/LIST_COMPREHENSION_SUPPORT.md) - Syntactic support
- [Closures](features/CLOSURE_SUPPORT_COMPLETE.md) - Factory functions and captured variables

**Backend Extensions:**
- [NumPy Extensions](features/NUMPY_EXTENSIONS_COMPLETE.md) - Extended NumPy gradient definitions
- [TensorFlow Extensions](features/TF_EXTENSIONS_COMPLETE.md) - TensorFlow 2.x gradient support

### `development/` - Development & Planning

Internal documentation for contributors and developers:

- [Roadmap to Greatness](development/ROADMAP_TO_GREATNESS.md) - Strategic roadmap and future plans
- [Class Support Plan](development/CLASS_SUPPORT_PLAN.md) - Implementation plan for classes
- [Inheritance Plan](development/INHERITANCE_PLAN.md) - Implementation plan for inheritance
- [Checkpointing TODO](development/CHECKPOINTING_TODO.md) - Checkpointing feature notes

## Quick Links

### For Users
- **Main README**: [../README.md](../README.md)
- **Python Feature Guide**: [features/PYTHON_FEATURE_SUPPORT.md](features/PYTHON_FEATURE_SUPPORT.md)
- **Tutorial Notebook**: [../notebooks/tangent_tutorial.ipynb](../notebooks/tangent_tutorial.ipynb)

### For Contributors
- **Contributing Guide**: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **Roadmap**: [development/ROADMAP_TO_GREATNESS.md](development/ROADMAP_TO_GREATNESS.md)

## Documentation Standards

All feature documentation follows this structure:

1. **Overview** - What the feature does
2. **Quick Examples** - Simple usage examples
3. **Supported Patterns** - What works and what doesn't
4. **How It Works** - Technical implementation details
5. **Test Coverage** - Description of test cases
6. **Limitations** - Current restrictions
7. **Usage Tips** - Best practices

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pedronahum/tangent/issues)
- **Examples**: [../examples/](../examples/)
- **Tests**: [../tests/](../tests/)
