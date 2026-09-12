# Tangent Documentation Index

Quick navigation to all documentation files.

---

## 📚 Main Documentation

- [README](../README.md) - Main project documentation
- [CONTRIBUTING](../CONTRIBUTING.md) - Contributing guidelines
- [Organization](ORGANIZATION.md) - How the docs tree is laid out

---

## 🔬 Features

The canonical reference is the **[Python Feature Support Guide](features/PYTHON_FEATURE_SUPPORT.md)** —
a complete matrix of what Tangent can and cannot differentiate, including
control flow, functions, classes and inheritance, containers (pytrees), and
higher-order derivatives.

Focused deep dives in `features/`:

- [Assert and Pass Support](features/ASSERT_PASS_SUPPORT.md)
- [Augmented Assignment Support](features/AUGMENTED_ASSIGNMENT_SUPPORT.md)
- [Boolean Operator Support](features/BOOLEAN_OPERATOR_SUPPORT.md)
- [Conditional Expression Support](features/CONDITIONAL_EXPRESSION_SUPPORT.md)
- [Error Messages](features/ERROR_MESSAGES.md)
- [For Loop Support](features/FOR_LOOP_SUPPORT.md)
- [List Comprehension Support](features/LIST_COMPREHENSION_SUPPORT.md)
- [Multi-Output Design](features/MULTI_OUTPUT_DESIGN.md)
- [Tuple Support](features/TUPLE_SUPPORT.md)
- [While Loop Support](features/WHILE_LOOP_SUPPORT.md)

Backend coverage (NumPy, JAX, TensorFlow, PyTorch, Keras 3, tinygrad) is
documented in the main README's
[Backend Support](../README.md#-backend-support) section.

---

## 📖 Guides

- [Checkpointing User Guide](checkpointing_user_guide.md) — gradient checkpointing: what works and what doesn't

---

## ⚡ Optimizations

- [Symbolic Optimizations](optimizations/SYMBOLIC_OPTIMIZATIONS_COMPLETE.md) - CSE and algebraic simplification
- [Strength Reduction](optimizations/STRENGTH_REDUCTION_COMPLETE.md) - Power and division optimization
- [Performance Analysis](optimizations/PERFORMANCE_ANALYSIS.md) - Optimization impact analysis
- [Straight-Line Coarsening](optimizations/COARSENING.md) - Symbolic whole-segment VJPs (opt-in)

Dead code elimination is described in the main README's
[optimization pipeline](../README.md#-advanced-optimization-pipeline) section;
the implementation lives in `tangent/optimizations/dce.py`.

---

## 🏆 Benchmarks

- [Framework Gradient Benchmarks](benchmarks/FRAMEWORK_BENCHMARKS.md) - MLP / conv / scalar vs jax, torch, autograd, finite differences
- [Framework Comparison](benchmarks/FRAMEWORK_COMPARISON.md) - Tangent vs TensorFlow vs PyTorch
- [Building Simulation Benchmark](benchmarks/BUILDING_SIMULATION_BENCHMARK.md) - Thermal simulation: optimization stack + vs TF/PyTorch
- [Correctness Verification](benchmarks/CORRECTNESS_VERIFICATION.md) - Mathematical correctness validation
- [Performance Improvement Strategies](benchmarks/PERFORMANCE_IMPROVEMENT_STRATEGIES.md) - Future optimization opportunities

Benchmark numbers are point-in-time measurements from the environments named
in each document.

---

## 🔧 Development History & Plans

Historical planning documents, kept for context (they describe intent at the
time of writing, not necessarily current behavior — trust the feature guide
and README over these):

- `development/` - Roadmaps and implementation plans (classes, inheritance, checkpointing)
- `plans/` - Modernization roadmap, TF2 integration plan, caching notes
- `bugs/` - Analyses of past bugs (e.g. [dict construction](bugs/DICT_CONSTRUCTION_BUG.md), since fixed)

---

## 🚀 Getting Started

1. **New to Tangent?** Start with the [README](../README.md)
2. **What can I differentiate?** See the [Python Feature Support Guide](features/PYTHON_FEATURE_SUPPORT.md)
3. **Performance?** See the [Framework Gradient Benchmarks](benchmarks/FRAMEWORK_BENCHMARKS.md)
4. **Examples?** Browse [../examples/](../examples/README.md) and the notebooks in [../notebooks/](../notebooks/)

---

## 📁 Directory Structure

```
tangent/
├── README.md                    # Main documentation
├── CONTRIBUTING.md              # Contributing guidelines
├── docs/
│   ├── INDEX.md                 # This file
│   ├── README.md                # Short docs overview
│   ├── ORGANIZATION.md          # How the docs tree is laid out
│   ├── checkpointing_user_guide.md
│   ├── features/                # Feature reference (PYTHON_FEATURE_SUPPORT.md is canonical)
│   ├── optimizations/           # Optimization deep dives
│   ├── benchmarks/              # Benchmark documentation
│   ├── bugs/                    # Past bug analyses
│   ├── development/             # Historical plans/roadmaps
│   └── plans/                   # Historical modernization plans
├── benchmarks/                  # Benchmark implementations
├── examples/                    # Runnable examples and notebooks
├── tests/                       # Test suite
└── tangent/                     # Source code
```
