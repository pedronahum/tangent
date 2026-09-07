# Tangent Documentation Index

Quick navigation to all documentation files.

---

## 📚 Main Documentation

- [README](../README.md) - Main project documentation
- [CONTRIBUTING](../CONTRIBUTING.md) - Contributing guidelines
- [Organization](ORGANIZATION.md) - Documentation organization guide

---

## 🏆 Benchmarks

### Performance Comparison
- **[Framework Comparison](benchmarks/FRAMEWORK_COMPARISON.md)** - Detailed comparison: Tangent vs TensorFlow vs PyTorch
- **[Benchmark Summary](benchmarks/BENCHMARK_SUMMARY.md)** - Executive summary of benchmark results
- **[Benchmark Results Summary](benchmarks/BENCHMARK_RESULTS_SUMMARY.md)** - Quick reference guide

### Building Simulation
- **[Building Simulation Benchmark](benchmarks/BUILDING_SIMULATION_BENCHMARK.md)** - Thermal simulation benchmark details
- **[Correctness Verification](benchmarks/CORRECTNESS_VERIFICATION.md)** - Mathematical correctness validation
- **[Performance Improvement Strategies](benchmarks/PERFORMANCE_IMPROVEMENT_STRATEGIES.md)** - Future optimization opportunities

**Key Results**:
- ✅ Tangent matches TensorFlow (4.300ms vs 4.315ms)
- ✅ 1.53× faster than PyTorch
- ✅ 10.80× faster forward pass than PyTorch
- ✅ Mathematically correct (verified to 7 significant figures)

---

## ⚡ Optimizations

### Symbolic Optimizations
- **[Symbolic Optimizations Complete](optimizations/SYMBOLIC_OPTIMIZATIONS_COMPLETE.md)** - CSE and algebraic simplification
- **[Strength Reduction Complete](optimizations/STRENGTH_REDUCTION_COMPLETE.md)** - Power and division optimization
- **[Performance Analysis](optimizations/PERFORMANCE_ANALYSIS.md)** - Optimization impact analysis

### Dead Code Elimination (DCE)
- [DCE Phase 0](optimizations/DCE_PHASE0_COMPLETE.md) - Initial implementation
- [DCE Phase 1](optimizations/DCE_PHASE1_COMPLETE.md) - Enhanced analysis
- [DCE Phase 2](optimizations/DCE_PHASE2_COMPLETE.md) - Advanced features
- [DCE Phase 3](optimizations/DCE_PHASE3_COMPLETE.md) - Integration
- [DCE Phase 4](optimizations/DCE_PHASE4_COMPLETE.md) - Finalization

**Optimization Results**:
- DCE alone: 1.95× speedup
- All optimizations: 2.35× speedup
- Strength reduction: x**2 → x*x (10× faster per operation)
- CSE: Eliminates redundant computations

---

## 🔬 Features

Located in `features/` directory:

- [Assert Pass Support](../ASSERT_PASS_SUPPORT.md)
- [Augmented Assignment Support](../AUGMENTED_ASSIGNMENT_SUPPORT.md)
- [Boolean Operator Support](../BOOLEAN_OPERATOR_SUPPORT.md)
- [Closure Support Complete](../CLOSURE_SUPPORT_COMPLETE.md)
- [Conditional Expression Support](../CONDITIONAL_EXPRESSION_SUPPORT.md)
- [For Loop Support](../FOR_LOOP_SUPPORT.md)
- [Lambda Support Complete](../LAMBDA_SUPPORT_COMPLETE.md)
- [List Comprehension Support](../LIST_COMPREHENSION_SUPPORT.md)
- [NumPy Extensions Complete](../NUMPY_EXTENSIONS_COMPLETE.md)
- [TensorFlow Extensions Complete](../TF_EXTENSIONS_COMPLETE.md)
- [While Loop Support](../WHILE_LOOP_SUPPORT.md)

---

## 📖 Guides

- [Checkpointing User Guide](checkpointing_user_guide.md) — what works and what doesn't
- [Control Flow Guide](../CONTROL_FLOW_GUIDE.md)
- [Python Feature Support](../PYTHON_FEATURE_SUPPORT.md)

---

## 🔧 Development

Located in `development/` directory:

- Development plans and technical notes
- Integration documentation
- Python extensions

---

## 📊 Quick Stats

### Performance
- **Gradient computation**: 4.300ms (matches TensorFlow!)
- **Forward pass**: 0.315ms (10× faster than PyTorch)
- **Optimization speedup**: 2.35× vs unoptimized

### Correctness
- **Verification**: All frameworks produce identical results
- **Max difference**: 1.83×10⁻⁶°C (0.000005%)
- **Status**: ✅ Production ready

### Benchmarks
- **Configuration**: 100 trials, 20 timesteps
- **Simulation**: Building thermal system (HVAC)
- **Frameworks tested**: Tangent, TensorFlow, PyTorch

---

## 🚀 Getting Started

1. **New to Tangent?** Start with [README](../README.md)
2. **Performance comparison?** See [Benchmark Summary](benchmarks/BENCHMARK_SUMMARY.md)
3. **Optimization details?** Check [Symbolic Optimizations](optimizations/SYMBOLIC_OPTIMIZATIONS_COMPLETE.md)
4. **Future improvements?** Read [Performance Improvement Strategies](benchmarks/PERFORMANCE_IMPROVEMENT_STRATEGIES.md)

---

## 📁 Directory Structure

```
tangent/
├── README.md                    # Main documentation
├── CONTRIBUTING.md              # Contributing guidelines
├── docs/
│   ├── INDEX.md                # This file
│   ├── benchmarks/             # Benchmark documentation
│   │   ├── FRAMEWORK_COMPARISON.md
│   │   ├── BENCHMARK_SUMMARY.md
│   │   ├── BUILDING_SIMULATION_BENCHMARK.md
│   │   ├── CORRECTNESS_VERIFICATION.md
│   │   └── PERFORMANCE_IMPROVEMENT_STRATEGIES.md
│   ├── optimizations/          # Optimization documentation
│   │   ├── SYMBOLIC_OPTIMIZATIONS_COMPLETE.md
│   │   ├── STRENGTH_REDUCTION_COMPLETE.md
│   │   ├── PERFORMANCE_ANALYSIS.md
│   │   └── DCE_PHASE*.md
│   ├── features/               # Feature documentation
│   ├── development/            # Development notes
│   └── plans/                  # Future plans
├── benchmarks/                 # Benchmark implementations
│   ├── building_simulation_tangent.py
│   ├── building_simulation_tensorflow.py
│   ├── building_simulation_pytorch.py
│   ├── building_simulation_compare.py
│   ├── verify_correctness.py
│   └── benchmark_results.txt
└── tangent/                    # Source code
    └── optimizations/          # Optimization implementations
```

---

**Last Updated**: November 2025
**Status**: ✅ All documentation current and accurate
