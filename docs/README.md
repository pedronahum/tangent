# Tangent Documentation

This directory contains all documentation for the Tangent automatic
differentiation library. The **[full index](INDEX.md)** lists every document;
this page is the short version.

## Directory Structure

### `features/` - Feature Documentation

The canonical reference is the
**[Python Feature Support Guide](features/PYTHON_FEATURE_SUPPORT.md)** — a
complete matrix of supported and unsupported Python features, including
control flow, functions (lambdas, closures), classes and inheritance,
containers (pytrees), and higher-order derivatives.

Focused deep dives:

- [Conditional Expressions](features/CONDITIONAL_EXPRESSION_SUPPORT.md) - Ternary operator support
- [Boolean Operators](features/BOOLEAN_OPERATOR_SUPPORT.md) - `and`, `or`, `not` operators
- [Augmented Assignment](features/AUGMENTED_ASSIGNMENT_SUPPORT.md) - `+=`, `-=`, `*=`, etc.
- [For Loops](features/FOR_LOOP_SUPPORT.md) - `for` loop with `range()`
- [While Loops](features/WHILE_LOOP_SUPPORT.md) - `while` loop support
- [Assert and Pass](features/ASSERT_PASS_SUPPORT.md) - Statement support
- [List Comprehensions](features/LIST_COMPREHENSION_SUPPORT.md) - Comprehension support
- [Tuple Support](features/TUPLE_SUPPORT.md) - Tuple access, unpacking, and returns
- [Multi-Output Design](features/MULTI_OUTPUT_DESIGN.md) - `output_index`/`output_weights`
- [Error Messages](features/ERROR_MESSAGES.md) - Actionable error design

Backend coverage (NumPy, JAX, TensorFlow, PyTorch, Keras 3, tinygrad) is
documented in the main README's
[Backend Support](../README.md#-backend-support) section.

### Other directories

- `optimizations/` - Optimization deep dives (CSE, strength reduction, coarsening)
- `benchmarks/` - Benchmark documentation and results
- `bugs/` - Analyses of past bugs
- `development/`, `plans/` - Historical roadmaps and implementation plans
  (point-in-time; trust the feature guide and README over these)

## Quick Links

### For Users
- **Main README**: [../README.md](../README.md)
- **Python Feature Guide**: [features/PYTHON_FEATURE_SUPPORT.md](features/PYTHON_FEATURE_SUPPORT.md)
- **Checkpointing Guide**: [checkpointing_user_guide.md](checkpointing_user_guide.md)
- **Tutorial Notebook**: [../notebooks/tangent_tutorial.ipynb](../notebooks/tangent_tutorial.ipynb)

### For Contributors
- **Contributing Guide**: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **Docs layout**: [ORGANIZATION.md](ORGANIZATION.md)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pedronahum/tangent/issues)
- **Examples**: [../examples/](../examples/)
- **Tests**: [../tests/](../tests/)
