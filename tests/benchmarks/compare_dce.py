"""
Compare performance before and after DCE implementation.
"""
import json
import sys


def load_results(filename):
    """Load benchmark results from JSON."""
    with open(filename, 'r') as f:
        return json.load(f)


def compare_results(baseline, optimized):
    """Compare and display results."""
    print("=" * 80)
    print("DCE PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    for bench_name in baseline.keys():
        base = baseline[bench_name]
        opt = optimized.get(bench_name, {})

        if not opt:
            print(f"{bench_name}: NO OPTIMIZED DATA")
            continue

        print(f"{bench_name}:")
        print("-" * 80)

        # Time comparison
        base_time = base['baseline_time'] * 1000
        opt_time = opt['baseline_time'] * 1000
        time_speedup = base_time / opt_time if opt_time > 0 else 0

        print(f"  Time:")
        print(f"    Baseline:  {base_time:.3f} ms")
        print(f"    Optimized: {opt_time:.3f} ms")
        print(f"    Speedup:   {time_speedup:.2f}×")

        # Memory comparison
        base_mem = base['baseline_memory']
        opt_mem = opt['baseline_memory']
        mem_reduction = (1 - opt_mem / base_mem) * 100 if base_mem > 0 else 0

        print(f"  Memory:")
        print(f"    Baseline:  {base_mem:.2f} MB")
        print(f"    Optimized: {opt_mem:.2f} MB")
        print(f"    Reduction: {mem_reduction:.1f}%")
        print()

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_dce.py baseline.json optimized.json")
        sys.exit(1)

    baseline = load_results(sys.argv[1])
    optimized = load_results(sys.argv[2])

    compare_results(baseline, optimized)
