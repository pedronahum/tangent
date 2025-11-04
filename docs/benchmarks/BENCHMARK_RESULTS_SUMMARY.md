# Benchmark Results Summary

## Quick Results

**Tangent vs TensorFlow vs PyTorch on Building Thermal Simulation**

| Framework | Gradient Time | Speedup vs PyTorch |
|-----------|---------------|-------------------|
| **Tangent** | **4.237ms** | **1.59×** |
| **TensorFlow** | 4.309ms | 1.56× |
| PyTorch | 6.729ms | Baseline |

**Tangent is 1.7% faster than TensorFlow and 1.59× faster than PyTorch!**

---

## Tangent Optimization Impact

**Building Simulation Benchmark**:

| Configuration | Time | Speedup |
|--------------|------|---------|
| No optimization | 10.028ms | Baseline |
| DCE only | 5.133ms | 1.95× |
| **All optimizations** | **4.237ms** | **2.35×** |

**Optimizations enabled**:
- ✅ Dead Code Elimination (DCE)
- ✅ Strength Reduction (`x**2` → `x*x`)
- ✅ Common Subexpression Elimination (CSE)
- ✅ Algebraic Simplification

---

## Files

### Benchmarks
- [`benchmarks/building_simulation_tangent.py`](benchmarks/building_simulation_tangent.py) - Tangent implementation
- [`benchmarks/building_simulation_tensorflow.py`](benchmarks/building_simulation_tensorflow.py) - TensorFlow implementation
- [`benchmarks/building_simulation_pytorch.py`](benchmarks/building_simulation_pytorch.py) - PyTorch implementation
- [`benchmarks/building_simulation_compare.py`](benchmarks/building_simulation_compare.py) - Run all benchmarks

### Documentation
- [`FRAMEWORK_COMPARISON.md`](FRAMEWORK_COMPARISON.md) - Detailed analysis of Tangent vs TensorFlow vs PyTorch
- [`BUILDING_SIMULATION_BENCHMARK.md`](BUILDING_SIMULATION_BENCHMARK.md) - Tangent optimization analysis
- [`STRENGTH_REDUCTION_COMPLETE.md`](STRENGTH_REDUCTION_COMPLETE.md) - Strength reduction implementation details
- [`SYMBOLIC_OPTIMIZATIONS_COMPLETE.md`](SYMBOLIC_OPTIMIZATIONS_COMPLETE.md) - CSE and algebraic simplification details

---

## How to Run

### Run individual benchmarks:
```bash
# Tangent (with optimizations)
python benchmarks/building_simulation_tangent.py

# TensorFlow
python benchmarks/building_simulation_tensorflow.py

# PyTorch
python benchmarks/building_simulation_pytorch.py
```

### Run comparison:
```bash
python benchmarks/building_simulation_compare.py
```

---

## Key Insights

1. **Tangent is competitive with TensorFlow**:
   - Actually 1.7% faster for gradient computation
   - 2.84× faster forward pass
   - Pure Python/NumPy with no runtime dependencies

2. **Symbolic optimizations work**:
   - 2.35× speedup over unoptimized Tangent
   - DCE provides most benefit (1.95×)
   - Strength reduction + CSE add another 20%

3. **Source-to-source transformation wins**:
   - Fastest forward pass (0.305ms)
   - No graph construction overhead
   - Generates readable Python code

4. **Both Tangent and TensorFlow outperform PyTorch**:
   - ~1.6× faster than PyTorch eager mode
   - Trade-off: PyTorch offers more flexibility

---

## Optimization Examples

### Strength Reduction
```python
# Before:
result = x ** 2  # Power operation (10 cycles)

# After:
result = x * x   # Multiplication (1 cycle)
```

### Common Subexpression Elimination
```python
# Before:
a = x * x
b = x * x  # Redundant

# After:
_cse_temp = x * x  # Computed once
a = _cse_temp
b = _cse_temp
```

### Dead Code Elimination
```python
# Before (gradient code):
def grad_f(x, by):
    # Full forward pass (20 timesteps)
    # ... many intermediate values ...
    # Backward pass uses only final values

# After DCE:
def grad_f(x, by):
    # Only necessary forward computations
    # No unused gradient accumulations
```

---

## Recommendations

**Use Tangent for**:
- ✅ Scientific computing with NumPy
- ✅ When you need fast forward pass
- ✅ When gradient speed matches TensorFlow
- ✅ When you want readable generated code

**Use TensorFlow for**:
- ✅ Production ML deployment
- ✅ GPU/TPU acceleration
- ✅ Large-scale systems

**Use PyTorch for**:
- ✅ Research and prototyping
- ✅ Maximum flexibility
- ✅ Easy debugging

---

**Date**: November 2025
**Status**: ✅ Complete
**Conclusion**: Tangent's source-to-source AD with symbolic optimizations is competitive with mature frameworks like TensorFlow while maintaining simplicity and generating readable code.
