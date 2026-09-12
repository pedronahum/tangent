# Building Simulation Benchmark Results

## Overview

Successfully implemented and benchmarked Tangent's automatic differentiation on a real-world thermal building simulation, comparing against the PassiveLogic differentiable Swift examples.

**Benchmark**: Building thermal simulation with floor heating, fluid flow, and heat transfer
**Source**: Based on https://github.com/PassiveLogic/differentiable-swift-examples/

!!! note "Reproducing these numbers"
    Every figure on this page comes from two scripts, so they cannot drift
    apart:

    ```
    python benchmarks/building_simulation_tangent.py     # optimization stack
    python benchmarks/building_simulation_compare.py      # vs TensorFlow / PyTorch
    ```

    **Measured:** 2026-09-12, Python 3.12 / Linux aarch64, NumPy 2.5, PyTorch
    2.14 (CPU), TensorFlow 2.x, CPU only, 100 trials × 20 timesteps. Absolute
    times are host-specific; the ratios are the point, and re-running the
    scripts refreshes the tables below.

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

### Optimization stack (Tangent, NumPy)

| Configuration | Forward | Gradient | Overhead |
|--------------|---------|----------|----------|
| **Tangent (No Opt)** | 0.000187s | 0.008708s | 46.7× |
| **Tangent (DCE)** | 0.000188s | 0.003329s | 17.7× |
| **Tangent (All Opts)** | 0.000189s | 0.003316s | 17.5× |

| Comparison | Speedup |
|-----------|---------|
| **DCE vs No Optimization** | **2.62×** |
| **All Opts vs No Optimization** | **2.63×** |
| **All Opts vs DCE** | **1.00×** |

### Versus TensorFlow and PyTorch

Same simulation, gradient of the loss, best steady-state of 100 trials:

| Framework | Forward | Gradient | Overhead |
|-----------|---------|----------|----------|
| **Tangent (All Opts)** | 0.000194s | 0.003323s | 17.1× |
| PyTorch | 0.003540s | 0.004258s | 1.2× |
| TensorFlow | 0.001326s | 0.005262s | 4.0× |

| Tangent vs | Gradient | Forward pass |
|-----------|----------|--------------|
| **PyTorch** | **1.28× faster** | 18.2× faster |
| **TensorFlow** | **1.58× faster** | 6.8× faster |

---

## Analysis

### 1. DCE is the dominant optimization (2.62×)

Dead Code Elimination removes unused forward computations, redundant gradient
accumulations, and the tape push/pop pairs for values the backward sweep never
reads. It alone drops the gradient from ~47× the forward cost to ~18×.

### 2. Symbolic optimizations add essentially nothing here (1.00×)

On this workload, layering Strength Reduction + CSE + Algebraic Simplification
on top of DCE produced **no measurable speedup** (3.329ms → 3.316ms, within
noise). These passes help expression-heavy scalar code (see the CSE/algebraic
micro-benchmarks), but this simulation's cost is dominated by array operations
that DCE has already pruned - so they are off by default, and this benchmark is
the honest reason why. (Earlier revisions of this page reported a 1.20× gain
here from a single earlier run; a fresh measurement does not reproduce it.)

### 3. Total optimization benefit: 2.63×

The optimized gradient is **2.63× faster** than the unoptimized one
(8.708ms → 3.316ms), essentially all of it from DCE. Against the frameworks a
user would switch from, the optimized Tangent gradient is faster than both
eager PyTorch (1.28×) and TensorFlow (1.58×) on this CPU workload, and its
forward pass - plain NumPy, no graph or tape to build - is many times faster.

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

**Impact**: **2.62× speedup** (8.708ms → 3.329ms) - the whole optimization win on this workload.

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

**Impact**: No measurable gain on *this* array-dominated workload; helps
expression-heavy scalar code. Off by default.

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

**Impact**: No measurable gain on *this* workload (DCE already pruned the
redundancy); pays off on code with repeated subexpressions. Off by default.

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

2. **Optimizations provide a real speedup**: **2.63× total improvement** over unoptimized gradients.

3. **DCE is the impactful optimization**: **2.62× speedup** alone, eliminating dead forward-pass code and unread tape entries - essentially the entire win here.

4. **Symbolic optimizations are workload-dependent**: no measurable gain on this array-dominated simulation (they help expression-heavy scalar code), which is why they are off by default.

5. **Faster than the eager frameworks**: the optimized gradient beats eager PyTorch (1.28×) and TensorFlow (1.58×) on this CPU workload.

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

**Status**: ✅ Complete
**Performance**: 2.63× speedup from optimizations (all from DCE)
**Baseline**: 8.708ms → **Optimized**: 3.316ms per gradient
**Numbers last measured**: 2026-09-12 (regenerate with the two scripts above)
