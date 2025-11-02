# Tangent Benchmarks

Performance benchmarks and profiling tools for Tangent.

## Files

- **benchmark_cache.py** - Function caching performance benchmarks

## Running Benchmarks

```bash
python benchmarks/benchmark_cache.py
```

## Results Summary

The caching system provides significant speedups for repeated gradient computations:
- First computation: Full compilation (slower)
- Subsequent calls: Cached retrieval (10-100x faster)
- Cache persistence: Cross-session performance benefits
