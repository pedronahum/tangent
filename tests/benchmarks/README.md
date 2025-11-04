# Dead Code Elimination (DCE) Benchmarks

This directory contains benchmarks for measuring the impact of Dead Code Elimination optimizations on Tangent's gradient computations.

## Phase 0: Baseline Setup ✅ COMPLETE

### Files Created

1. **[dce_benchmarks.py](dce_benchmarks.py)** - Main benchmark suite
   - `SelectiveGradientBenchmark` - Tests gradient computation for 1 parameter out of many
   - `UnusedComputationBenchmark` - Tests elimination of unused variable computations
   - `UnusedRegularizationBenchmark` - Tests elimination of computed-but-unused regularization terms
   - `ConditionalBranchBenchmark` - Tests elimination of dead branches in conditionals

2. **[compare_dce.py](compare_dce.py)** - Comparison script for before/after metrics

3. **[baseline_results.json](baseline_results.json)** - Baseline performance data (before DCE implementation)

### Baseline Results

Current performance (without DCE optimization):

| Benchmark | Time (ms) | Memory (MB) | Description |
|-----------|-----------|-------------|-------------|
| Selective Gradient (1/10 params) | 0.008 | 0.00 | Computing gradient w.r.t. only 1 of 10 parameters |
| Unused Computation Elimination | 0.010 | 0.00 | Function with unused variable computations |
| Unused Regularization Term | 0.002 | 0.00 | Regularization computed but not used in output |
| Conditional Dead Branch | 0.003 | 0.00 | Dead else branch in conditional |

### Running Benchmarks

```bash
# From tangent repository root
python tests/benchmarks/dce_benchmarks.py
```

### Expected Improvements

Once DCE is implemented (Phases 1-3):

| Benchmark | Expected Speedup |
|-----------|------------------|
| Selective Gradient | 2-5× |
| Unused Computation | 1.5-3× |
| Unused Regularization | 1.5-2× |
| Conditional Dead Branch | 1.3-2× |

### Comparing Results

After implementing DCE optimizations, compare performance:

```bash
# Run optimized version (creates optimized_results.json)
python tests/benchmarks/dce_benchmarks.py  # After DCE implementation
mv tests/benchmarks/baseline_results.json tests/benchmarks/optimized_results.json

# Compare
python tests/benchmarks/compare_dce.py \
    tests/benchmarks/baseline_results.json \
    tests/benchmarks/optimized_results.json
```

### Reproducibility

Benchmarks are reproducible within ±10% variance:
- ✅ Warmup iterations: 10
- ✅ Measurement iterations: 100
- ✅ Consistent results across runs

### Next Steps

- [ ] **Phase 1**: Implement backward slicing for basic DCE
- [ ] **Phase 2**: Add activity analysis for more aggressive elimination
- [ ] **Phase 3**: Handle control flow (loops, conditionals) with SSA
- [ ] **Phase 4**: Integrate with coarsening optimization

---

## Notes

- Benchmarks use simplified functions compatible with Tangent's current feature set
- Generator expressions converted to explicit operations (Tangent limitation)
- Memory measurements may show 0.00 MB due to small allocation sizes
- Time measurements are in microseconds, showing very fast baseline performance
