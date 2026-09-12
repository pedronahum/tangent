# Framework Comparison: Tangent vs TensorFlow vs PyTorch

## Overview

This document presents a comprehensive comparison of automatic differentiation frameworks on a real-world building thermal simulation benchmark.

**Benchmark**: Building HVAC thermal simulation with radiant floor heating
**Source**: Based on https://github.com/PassiveLogic/differentiable-swift-examples/
**Simulation**: 20 timesteps, 100 trials, 3 warmup iterations

---

!!! info "Where the numbers live"
    This page explains *how* the three approaches differ. For measured,
    reproducible figures see the
    [Building Simulation Benchmark](BUILDING_SIMULATION_BENCHMARK.md) (this same
    thermal simulation, with the optimization stack and the vs-TF/PyTorch
    table) and the [Framework Gradient Benchmarks](FRAMEWORK_BENCHMARKS.md)
    (MLP / conv / scalar vs jax, torch, autograd, finite differences). Keeping
    all numbers in those two generated docs is what stops them drifting.

## Results Summary

On the thermal simulation (CPU, most recent run), the optimized Tangent
gradient is **faster than both eager PyTorch (~1.3×) and TensorFlow (~1.6×)**,
and its forward pass - plain NumPy, with no graph or tape to build - is several
times faster than either. PyTorch has the lowest backward/forward *overhead
ratio*, but the highest absolute time, because its forward pass is the slowest.
See the [Building Simulation Benchmark](BUILDING_SIMULATION_BENCHMARK.md) for
the exact per-framework table.

---

## Detailed Analysis

### Forward Pass Performance

**Why Tangent is fastest**:
- ✅ Source-to-source compilation to pure Python/NumPy
- ✅ No graph construction overhead
- ✅ No framework bookkeeping during forward pass
- ✅ Direct NumPy operations with minimal wrapping

**Why TensorFlow is middle**:
- ❌ `@tf.function` compilation overhead (mitigated by warmup)
- ❌ TensorFlow runtime overhead
- ✅ Efficient compiled kernels once warmed up

**Why PyTorch is slowest**:
- ❌ Eager execution with tape tracking overhead
- ❌ Tensor wrapping for autograd
- ❌ Python dispatch overhead for each operation

### Gradient Computation Performance

**Why Tangent and TensorFlow are close**:
- Both avoid redundant computations
- TensorFlow uses XLA compilation (behind `@tf.function`)
- Tangent uses tape-aware DCE (its dominant optimization on this workload)

**Why PyTorch is slower here**:
- Eager mode backward pass
- More dynamic graph construction overhead
- Less opportunity for optimization

---

## Framework-Specific Analysis

### Tangent

**Strengths**:
- ✅ **Fastest forward pass** - source-to-source to pure NumPy, no graph/tape build
- ✅ **Competitive gradient computation** - beats eager PyTorch and TensorFlow here
- ✅ **Source-to-source transformation** - generates readable Python code
- ✅ **Tape-aware DCE** provides the bulk of the optimization speedup
- ✅ **Pure Python/NumPy** - no special runtime required
- ✅ **Supports arbitrary Python code** (with some limitations)

**Weaknesses**:
- ❌ **High backward/forward overhead ratio** - the backward pass dominates
- ❌ **Limited tensor library support** (NumPy focus; JAX/Torch/TF/Keras/tinygrad extensions)
- ❌ **Source transformation limitations** (can't use `.copy()`, some dynamic patterns)
- ❌ **No GPU acceleration** in this benchmark (use `compile='jax'` to lower onto XLA)
- ❌ **Compilation time** - a one-time source-transform bill (amortized by the disk cache)

See the [Building Simulation Benchmark](BUILDING_SIMULATION_BENCHMARK.md) for the
measured optimization-stack breakdown (DCE is ~2.6× and does essentially all of
the work; the symbolic passes add nothing on this array-heavy workload).

### TensorFlow

**Strengths**:
- ✅ **Fast gradient computation** via XLA graph optimization
- ✅ **Low backward/forward overhead ratio**
- ✅ **Production-ready** with extensive ecosystem
- ✅ **GPU/TPU support** (not tested here)
- ✅ **Extensive library support**

**Weaknesses**:
- ❌ **Slower forward pass** than Tangent (graph/runtime overhead)
- ❌ **Graph mode complexity** (`@tf.function` can be tricky)
- ❌ **Less flexible** than eager execution for debugging
- ❌ **Slower than optimized Tangent** for gradients on this CPU workload

**Implementation notes**:
- Uses `@tf.function` decorator for graph compilation
- Requires `tf.Variable` for gradient computation
- `tf.range()` for loops (graph-compatible)
- All operations use TensorFlow ops

### PyTorch

**Strengths**:
- ✅ **Lowest backward/forward overhead ratio** - backward adds minimal overhead
- ✅ **Eager execution** - easy to debug
- ✅ **Pythonic API** - most intuitive for Python developers
- ✅ **Dynamic computational graphs** - maximum flexibility
- ✅ **Strong ecosystem** (torchvision, etc.)

**Weaknesses**:
- ❌ **Slowest absolute performance** for both forward and gradient on this CPU workload
- ❌ **Eager mode overhead** - trades performance for flexibility
- ❌ **No automatic optimization** - relies on manual JIT compilation
- ❌ **Slower than optimized Tangent** for gradients here (see the canonical table)

**Why PyTorch is slower**:
- Eager execution requires building computation graph during forward pass
- Each operation requires autograd bookkeeping
- Less opportunity for global optimization
- Python dispatch overhead on every operation

**Note**: PyTorch JIT (`torch.jit.script`) was not tested but could improve performance significantly.

---

## Benchmark Details

### Simulation Description

**Physical model**:
- Concrete floor slab (thermal mass)
- PEX tubing embedded in floor
- Hot water circulation from storage tank
- Heat transfer between fluid and floor
- Temperature evolution over 20 timesteps

**Key operations per timestep**:
1. Update source tank temperature
2. Update fluid temperature (twice per timestep)
3. Compute thermal resistance
4. Compute load power (heat transfer)
5. Update building model temperature

**Computational characteristics**:
- **No matrix operations** - scalar/vector arithmetic only
- **Sequential timesteps** - cannot be parallelized
- **Moderate complexity** - ~10-15 operations per timestep
- **Pure math** - no I/O, no conditionals (loop only)

### Why This Benchmark Matters

This benchmark represents **real-world scientific computing**:
- ✅ **Physics simulation** - thermal dynamics
- ✅ **Sequential time integration** - common in ODEs/PDEs
- ✅ **Parameter optimization** - building control systems
- ✅ **Gradient-based optimization** - HVAC parameter tuning

**Not representative of**:
- ❌ Deep learning (no large matrix multiplications)
- ❌ GPU workloads (CPU-only benchmark)
- ❌ Large-scale models (small state space)

---

## Implementation Differences

### Tangent Implementation

**Key patterns**:
```python
# Avoid .copy() - use array arithmetic
result = array * np.array([0, 1, 1, 1, 1]) + value * np.array([1, 0, 0, 0, 0])

# Compile gradient with optimizations
grad_f = tangent.grad(f, optimized=True,
                      optimizations={
                          'dce': True,
                          'strength_reduction': True,
                          'cse': True,
                          'algebraic': True
                      })
```

**Compilation required**: Gradient function compiled once, then reused.

### TensorFlow Implementation

**Key patterns**:
```python
# Use @tf.function for graph compilation
@tf.function
def simulate(sim_params):
    # TensorFlow operations
    for i in tf.range(TIMESTEPS):  # tf.range for graph mode
        # ...

# Gradient computation with GradientTape
with tf.GradientTape() as tape:
    tape.watch(sim_params)
    result = simulate(sim_params)
gradient = tape.gradient(result, sim_params)
```

**Graph compilation**: First call compiles, subsequent calls reuse graph.

### PyTorch Implementation

**Key patterns**:
```python
# Tensors with requires_grad=True
tensor = torch.tensor([...], requires_grad=True)

# Eager execution - no decorators needed
result = simulate(sim_params)

# Gradient computation
gradient = torch.autograd.grad(result, inputs, retain_graph=True)
```

**Eager execution**: No compilation, graph built during forward pass.

---

## Performance Tuning Observations

### Tangent Tuning

**What worked**:
- ✅ Enabling optimization (DCE): ~2.6× speedup - essentially the whole win
- ✅ Replacing `.copy()` with array arithmetic
- ✅ Using `+ 0.0` instead of `.copy()` for array duplication

**What didn't help**:
- Algebraic simplification had minimal impact (expressions already simple)

### TensorFlow Tuning

**What worked**:
- ✅ `@tf.function` decorator for graph compilation
- ✅ Using `tf.range()` instead of Python `range()`
- ✅ Warmup iterations to amortize compilation cost

**What we didn't test**:
- XLA compilation flags
- GPU execution
- Mixed precision

### PyTorch Tuning

**What we tested**:
- Eager execution (what's implemented)

**What we didn't test**:
- `torch.jit.script()` compilation (could significantly improve performance)
- GPU execution
- `torch.compile()` (PyTorch 2.0+)

---

## Conclusions

### Main Takeaways

1. **Tangent is competitive for scientific computing**:
   - Faster than eager TensorFlow and PyTorch for gradients on this CPU workload
   - Fastest forward pass by far (plain NumPy, no graph/tape build)

2. **Optimization matters for Tangent**:
   - ~2.6× speedup from optimization, essentially all from tape-aware DCE
   - Symbolic optimizations add nothing on this array-heavy workload (they help
     expression-heavy scalar code)

3. **Framework choice depends on use case**:
   - **Tangent**: Best for pure Python/NumPy code, scientific computing, when you need readable generated code
   - **TensorFlow**: Best for production ML, GPU acceleration, large-scale systems
   - **PyTorch**: Best for research, debugging, dynamic models, when ease-of-use matters most

4. **Tangent's source-to-source approach works**:
   - Generates efficient gradient code
   - Avoids runtime overhead of tape/graph construction
   - Enables symbolic optimizations

### Recommendations

**Use Tangent when**:
- ✅ Working with NumPy-based scientific code
- ✅ Need readable generated gradient code
- ✅ Want to avoid framework dependencies
- ✅ Forward pass performance is critical
- ✅ Gradient computation is similar speed to TensorFlow/PyTorch

**Use TensorFlow when**:
- ✅ Need production ML deployment
- ✅ GPU/TPU acceleration required
- ✅ Large-scale distributed training
- ✅ Extensive ecosystem integration

**Use PyTorch when**:
- ✅ Research and experimentation
- ✅ Need maximum flexibility
- ✅ Debugging is frequent
- ✅ Dynamic models with varying structure
- ✅ Can sacrifice performance for ease-of-use

### Surprising Results

1. **Tangent faster than eager TensorFlow** for gradients here:
   - Expected TensorFlow to win due to its mature XLA compiler
   - Tape-aware DCE on the generated code is highly effective
   - Source-to-source transformation avoids some runtime overhead

2. **Tangent's forward pass is several times faster** than both frameworks:
   - Pure NumPy with no framework wrapping
   - No graph construction or bookkeeping
   - Direct execution of optimized code

3. **PyTorch's low backward/forward overhead ratio**:
   - Backward pass adds minimal overhead
   - But absolute performance suffers from a slow forward pass
   - Trade-off: flexibility vs performance

### Future Work

**Tangent improvements**:
- GPU support via JAX backend
- JIT compilation for forward pass
- More aggressive optimizations
- Better handling of dynamic patterns

**Benchmark extensions**:
- Test with PyTorch JIT (`torch.jit.script`)
- Test with TensorFlow XLA flags
- GPU benchmark comparison
- Larger simulations (100+ timesteps)
- Memory profiling

---

## Appendix: Benchmark Configuration

**Hardware**: macOS (Darwin 24.6.0)
**Python**: 3.x
**Libraries**:
- Tangent: Latest (with all symbolic optimizations)
- TensorFlow: 2.x
- PyTorch: 2.x
- NumPy: Latest

**Simulation parameters**:
```python
TRIALS = 100
TIMESTEPS = 20
WARMUP = 3
D_TIME = 0.1
```

**Physical parameters**:
```python
# Floor slab
SLAB = [temp=21.1°C, area=100m², Cp=0.2 kJ/(kg·K),
        density=2242.58 kg/m³, thickness=0.101m]

# PEX tubing
TUBE = [spacing=0.503m, diameter=0.019m, thickness=0.00159m,
        resistivity=2.43 K/W]

# Water/glycol
QUANTA = [power=0W, temp=60°C, flow=0.000631 m³/s,
          density=1000 kg/m³, Cp=4180 J/(kg·K)]

# Hot water tank
TANK = [temp=70°C, volume=0.0757 m³, Cp=4180 J/(kg·K),
        density=1000 kg/m³, mass=75.7 kg]

# Starting conditions
STARTING_TEMP = 33.3°C
```

**Ground truth**: Final temperature = 27.344767°C

---

**Benchmark Date**: November 2025
**Status**: ✅ Complete
**Files**:
- `benchmarks/building_simulation_tangent.py`
- `benchmarks/building_simulation_tensorflow.py`
- `benchmarks/building_simulation_pytorch.py`
- `benchmarks/building_simulation_compare.py`

**Key Result**: Tangent is competitive with (and slightly faster than) TensorFlow for gradient computation, while being significantly faster for forward pass. Both outperform PyTorch by ~1.6× for this workload.
