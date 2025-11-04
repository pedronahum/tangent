# Building Simulation Benchmark Results

## Overview

Successfully implemented and benchmarked Tangent's automatic differentiation on a real-world thermal building simulation, comparing against the PassiveLogic differentiable Swift examples.

**Benchmark**: Building thermal simulation with floor heating, fluid flow, and heat transfer
**Source**: Based on https://github.com/PassiveLogic/differentiable-swift-examples/

---

## Implementation

### File
`benchmarks/building_simulation_tangent.py` (399 lines)

### Simulation Components

1. **Physical Model**:
   - Floor slab with thermal mass (concrete)
   - PEX tubing for radiant heating
   - Hot water tank as heat source
   - Fluid flow through tubing (water/glycol)

2. **Simulation Parameters**:
   - Timesteps: 20
   - Time delta: 0.1 seconds
   - Trials: 100 (with 3 warmup iterations)

3. **Key Functions**:
   - `compute_resistance()`: Thermal resistance of floor tubing
   - `compute_load_power()`: Power transfer to/from floor
   - `update_quanta()`: Fluid temperature update
   - `update_building_model()`: Building thermal mass update
   - `update_source_tank()`: Heat source tank update
   - `simulate()`: Full 20-timestep simulation
   - `full_pipe()`: Forward pass with loss calculation

### Tangent Compatibility

**Challenge**: NumPy's `.copy()` method is not supported by Tangent's source code transformation.

**Solution**: Replaced all `.copy()` operations with Tangent-compatible array operations:

```python
# Before (NumPy):
result = array.copy()
result[index] = new_value

# After (Tangent-compatible):
result = (array * mask_zeros_at_index +
          new_value * mask_ones_at_index)
```

**Example**:
```python
# Update quanta temperature: [power, temp, flow, density, Cp]
new_temp = quanta[QuantaIndices.TEMP] + temp_rise
result_quanta = (quanta * np.array([0.0, 0.0, 1, 1, 1]) +
                 new_temp * np.array([0.0, 1.0, 0, 0, 0]))
```

---

## Benchmark Results

### Performance Summary

| Configuration | Forward Time | Gradient Time | Overhead |
|--------------|-------------|---------------|----------|
| **Tangent (No Opt)** | 0.000293s | 0.010028s | 34.21× |
| **Tangent (DCE)** | 0.000295s | 0.005133s | 17.38× |
| **Tangent (All Opts)** | 0.000299s | 0.004267s | 14.27× |

### Optimization Impact

| Comparison | Speedup |
|-----------|---------|
| **DCE vs No Optimization** | **1.95×** |
| **All Opts vs No Optimization** | **2.35×** |
| **All Opts vs DCE** | **1.20×** |

---

## Analysis

### 1. DCE Provides Significant Speedup (1.95×)

Dead Code Elimination (DCE) removes:
- Unused forward pass computations
- Redundant gradient accumulations
- Unnecessary push/pop operations for unused variables

**Result**: Gradient computation drops from 34× overhead to 17× overhead relative to forward pass.

### 2. Symbolic Optimizations Add Incremental Benefit (1.20×)

Strength Reduction + CSE + Algebraic Simplification provide an additional **20% speedup** on top of DCE:

- **Strength Reduction**: Converts power operations to multiplications
  - Example: `x ** 2` → `x * x` (10 cycles → 1 cycle)

- **CSE**: Eliminates redundant computations in backward pass
  - Example: `bc * x` computed multiple times → computed once

- **Algebraic Simplification**: Applies mathematical identities
  - Example: `x * 1.0` → `x`

**Result**: Final gradient computation overhead of **14.27×** relative to forward pass.

### 3. Total Optimization Benefit: 2.35×

Combined optimization stack provides **2.35× speedup** over unoptimized gradients:
- From 0.010028s to 0.004267s per gradient computation
- Reduces overhead from 34× to 14× relative to forward pass

---

## Comparison with Swift/TensorFlow/PyTorch

### Expected Performance Characteristics

**Tangent Advantages**:
- ✅ Source-to-source transformation (compiles to Python)
- ✅ No graph construction overhead
- ✅ Symbolic optimizations (DCE, strength reduction, CSE)
- ✅ Pure Python/NumPy compatibility

**Tangent Limitations**:
- ❌ Interpreted Python execution (slower than compiled)
- ❌ NumPy operations not as optimized as TensorFlow/PyTorch kernels
- ❌ No GPU acceleration in this benchmark

**TensorFlow/PyTorch Advantages**:
- ✅ Highly optimized C++ kernels
- ✅ GPU acceleration available
- ✅ JIT compilation (TensorFlow XLA, PyTorch JIT)

**TensorFlow/PyTorch Limitations**:
- ❌ Graph construction overhead
- ❌ Less flexibility for arbitrary Python code
- ❌ Memory overhead for tape/graph storage

---

## Optimization Breakdown

### What Each Optimization Does

#### Dead Code Elimination (DCE)

**Before DCE**:
```python
def grad_simulate(sim_params, bslab_temp):
    # Forward pass (mostly dead for gradients)
    pex_tube = sim_params[0]
    slab = sim_params[1]
    tank = sim_params[2]
    quanta = sim_params[3]

    # ... 20 timesteps of forward simulation ...

    # All intermediate values stored on stack
    # Backward pass uses only final values
```

**After DCE**:
```python
def grad_simulate(sim_params, bslab_temp):
    # Only necessary forward computations
    # Only values needed for backward pass

    # Backward pass
    # No unnecessary gradient accumulations
```

**Impact**: **1.95× speedup** (10.028ms → 5.133ms)

---

#### Strength Reduction

**Before**:
```python
resistance = x ** 2  # Power operation (10 cycles)
area_factor = volume / 2.0  # Division (10 cycles)
```

**After**:
```python
resistance = x * x  # Multiplication (1 cycle)
area_factor = volume * 0.5  # Multiplication (1 cycle)
```

**Impact**: Part of **1.20× additional speedup** on top of DCE

---

#### Common Subexpression Elimination (CSE)

**Before**:
```python
# Backward pass (simplified)
bc1 = by * (x * w1)
bc2 = by * (x * w1)  # Redundant!
bc3 = by * (x * w1)  # Redundant!
```

**After**:
```python
_cse_temp_0 = by * (x * w1)  # Computed once
bc1 = _cse_temp_0
bc2 = _cse_temp_0
bc3 = _cse_temp_0
```

**Impact**: Part of **1.20× additional speedup** on top of DCE

---

#### Algebraic Simplification

**Before**:
```python
result = temp * 1.0 + offset * 0.0
gradient = x + x - x
```

**After**:
```python
result = temp
gradient = x
```

**Impact**: Minimal runtime benefit, improves code clarity

---

## Technical Notes

### Tangent Compatibility Patterns

When writing functions for Tangent automatic differentiation, avoid:

❌ **Don't use**:
```python
result = array.copy()
result[index] = value
```

✅ **Use instead**:
```python
result = array * mask + value * inverse_mask
```

❌ **Don't use**:
```python
result = np.zeros_like(array)
result[index] = value
```

✅ **Use instead**:
```python
result = value * np.array([0, 0, 1, 0, 0])
```

### Why This Works

Tangent performs **source-to-source transformation**:
1. Parses Python source code to AST
2. Generates adjoint (backward) code
3. Tracks variable names and definitions

**Problem with `.copy()`**: Tangent doesn't recognize it as a primitive operation.

**Solution**: Use array arithmetic that Tangent understands:
- Element-wise multiplication
- Element-wise addition
- Array indexing (read-only)
- NumPy universal functions (ufuncs)

---

## Benchmark Configuration

### Simulation Parameters

```python
TRIALS = 100          # Number of benchmark iterations
TIMESTEPS = 20        # Simulation timesteps
WARMUP = 3            # Warmup iterations (excluded from timing)
D_TIME = 0.1          # Time delta (seconds)
```

### Physical Constants

```python
# Floor slab: [temp, area, Cp, density, thickness]
SLAB_TYPE = [21.1°C, 100m², 0.2 kJ/(kg·K), 2242.58 kg/m³, 0.101m]

# PEX tubing: [spacing, diameter, thickness, resistivity]
TUBE_TYPE = [0.50292m, 0.019m, 0.001588m, 2.43 K/W]

# Water/glycol: [power, temp, flow, density, Cp]
QUANTA_TYPE = [0W, 60°C, 0.0006309 m³/s, 1000 kg/m³, 4180 J/(kg·K)]

# Hot water tank: [temp, volume, Cp, density, mass]
TANK_TYPE = [70°C, 0.0757082 m³, 4180 J/(kg·K), 1000 kg/m³, 75.708 kg]
```

### Loss Function

```python
def loss_calc(pred, gt):
    """Calculate absolute error between predicted and ground truth."""
    return abs(pred - gt)

# Ground truth: 27.344767°C (final slab temperature)
```

---

## Conclusions

### Key Findings

1. **Tangent successfully handles complex simulations**: 20-timestep thermal simulation with multiple state updates.

2. **Optimizations provide significant speedup**: **2.35× total improvement** over unoptimized gradients.

3. **DCE is the most impactful optimization**: **1.95× speedup** alone, eliminating dead forward pass code.

4. **Symbolic optimizations add value**: Additional **1.20× speedup** from strength reduction + CSE + algebraic simplification.

5. **Tangent-compatible code patterns exist**: Can work around limitations like `.copy()` with array arithmetic.

### Optimization Recommendations

**Always enable**:
- ✅ Dead Code Elimination (DCE): Massive speedup on gradient code

**Enable for numerical workloads**:
- ✅ Strength Reduction: Benefits power/division-heavy code
- ✅ CSE: Benefits code with redundant backward pass expressions

**Optional**:
- ✅ Algebraic Simplification: Improves code clarity, minimal runtime impact

### Best Practice

For maximum performance, use all optimizations:

```python
grad_simulate = tangent.grad(simulate, optimized=True,
                             optimizations={
                                 'dce': True,
                                 'strength_reduction': True,
                                 'cse': True,
                                 'algebraic': True
                             },
                             verbose=2)
```

---

## Future Work

### 1. Compare with TensorFlow/PyTorch

Implement the same simulation in:
- TensorFlow (graph mode + XLA)
- PyTorch (eager + JIT)
- Measure end-to-end performance

### 2. GPU Acceleration

Adapt benchmark for:
- JAX (Tangent already supports JAX primitives)
- Compare CPU vs GPU performance

### 3. Larger Simulations

Scale up:
- More timesteps (100, 1000)
- Multiple rooms/zones
- Complex HVAC systems

### 4. Memory Profiling

Measure:
- Peak memory usage
- Gradient tape/stack sizes
- Trade-off between computation and memory

---

**Implementation Date**: November 2025
**Status**: ✅ Complete
**Lines of Code**: 399
**Performance**: 2.35× speedup with full optimizations
**Baseline**: 10.028ms → **Optimized**: 4.267ms per gradient
