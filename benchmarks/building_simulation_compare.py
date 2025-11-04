"""
Building Simulation Benchmark - Framework Comparison

Runs building simulation benchmark across Tangent, TensorFlow, and PyTorch
to compare automatic differentiation performance.

Usage:
    python benchmarks/building_simulation_compare.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tangent_benchmark():
    """Run Tangent benchmark with all optimizations."""
    print("\n" + "=" * 80)
    print("RUNNING TANGENT BENCHMARKS")
    print("=" * 80)

    import numpy as np
    import tangent
    import time

    # Simulation parameters
    TRIALS = 100
    TIMESTEPS = 20
    WARMUP = 3
    D_TIME = 0.1

    # Constants
    PI = 3.14159265359

    # Type definitions
    TUBE_TYPE = np.array([0.50292, 0.019, 0.001588, 2.43, 0.0])

    class TubeIndices:
        TUBE_SPACING = 0
        DIAMETER = 1
        THICKNESS = 2
        RESISTIVITY = 3

    SLAB_TYPE = np.array([21.1111111, 100.0, 0.2, 2242.58, 0.101])

    class SlabIndices:
        TEMP = 0
        AREA = 1
        CP = 2
        DENSITY = 3
        THICKNESS = 4

    QUANTA_TYPE = np.array([0.0, 60.0, 0.0006309, 1000.0, 4180.0])

    class QuantaIndices:
        POWER = 0
        TEMP = 1
        FLOW = 2
        DENSITY = 3
        CP = 4

    TANK_TYPE = np.array([70.0, 0.0757082, 4180.0, 1000.0, 75.708])

    class TankIndices:
        TEMP = 0
        VOLUME = 1
        CP = 2
        DENSITY = 3
        MASS = 4

    STARTING_TEMPERATURE = np.array([33.3, 0, 0, 0, 0])
    SIM_PARAMS = np.array([TUBE_TYPE, SLAB_TYPE, QUANTA_TYPE, TANK_TYPE, STARTING_TEMPERATURE])

    class SimParamsIndices:
        TUBE = 0
        SLAB = 1
        QUANTA = 2
        TANK = 3
        STARTING_TEMP = 4

    # Simulation functions
    def compute_resistance(floor, tube, quanta):
        geometry_coeff = 10.0
        tubing_surface_area = (floor[SlabIndices.AREA] / tube[TubeIndices.TUBE_SPACING]) * PI * tube[TubeIndices.DIAMETER]
        resistance_abs = tube[TubeIndices.RESISTIVITY] * tube[TubeIndices.THICKNESS] / tubing_surface_area
        return resistance_abs * geometry_coeff

    def compute_load_power(floor, tube, quanta):
        resistance_abs = compute_resistance(floor, tube, quanta)
        conductance = 1.0 / resistance_abs
        d_temp = floor[SlabIndices.TEMP] - quanta[QuantaIndices.TEMP]
        power = d_temp * conductance
        load_power = -power
        result_quanta = quanta * np.array([0.0, 1, 1, 1, 1]) + power * np.array([1.0, 0, 0, 0, 0])
        return result_quanta, load_power

    def update_quanta(quanta):
        working_volume = quanta[QuantaIndices.FLOW] * D_TIME
        working_mass = working_volume * quanta[QuantaIndices.DENSITY]
        working_energy = quanta[QuantaIndices.POWER] * D_TIME
        temp_rise = working_energy / quanta[QuantaIndices.CP] / working_mass
        new_temp = quanta[QuantaIndices.TEMP] + temp_rise
        result_quanta = (quanta * np.array([0.0, 0.0, 1, 1, 1]) +
                         new_temp * np.array([0.0, 1.0, 0, 0, 0]))
        return result_quanta

    def update_building_model(power, floor):
        floor_volume = floor[SlabIndices.AREA] * floor[SlabIndices.THICKNESS]
        floor_mass = floor_volume * floor[SlabIndices.DENSITY]
        floor_temp_change = (power * D_TIME) / floor[SlabIndices.CP] / floor_mass
        new_temp = floor[SlabIndices.TEMP] + floor_temp_change
        result_floor = (floor * np.array([0.0, 1, 1, 1, 1]) +
                        new_temp * np.array([1.0, 0, 0, 0, 0]))
        return result_floor

    def update_source_tank(store, quanta):
        mass_per_time = quanta[QuantaIndices.FLOW] * quanta[QuantaIndices.DENSITY]
        d_temp = store[TankIndices.TEMP] - quanta[QuantaIndices.TEMP]
        power = d_temp * mass_per_time * quanta[QuantaIndices.CP]
        updated_quanta = (quanta * np.array([0.0, 1, 1, 1, 1]) +
                          power * np.array([1.0, 0, 0, 0, 0]))
        tank_mass = store[TankIndices.VOLUME] * store[TankIndices.DENSITY]
        temp_rise = (power * D_TIME) / store[TankIndices.CP] / tank_mass
        new_tank_temp = store[TankIndices.TEMP] + temp_rise
        updated_store = (store * np.array([0.0, 1, 1, 1, 1]) +
                         new_tank_temp * np.array([1.0, 0, 0, 0, 0]))
        return updated_store, updated_quanta

    def simulate(sim_params):
        pex_tube = sim_params[SimParamsIndices.TUBE] + 0.0
        slab = sim_params[SimParamsIndices.SLAB] + 0.0
        tank = sim_params[SimParamsIndices.TANK] + 0.0
        quanta = sim_params[SimParamsIndices.QUANTA] + 0.0
        starting_temp = sim_params[SimParamsIndices.STARTING_TEMP][0]
        slab = (slab * np.array([0.0, 1, 1, 1, 1]) +
                starting_temp * np.array([1.0, 0, 0, 0, 0]))
        for i in range(TIMESTEPS):
            tank, quanta = update_source_tank(tank, quanta)
            quanta = update_quanta(quanta)
            quanta, power_to_building = compute_load_power(slab, pex_tube, quanta)
            quanta = update_quanta(quanta)
            slab = update_building_model(power_to_building, slab)
        return slab[SlabIndices.TEMP]

    def full_pipe(sim_params):
        pred = simulate(sim_params)
        loss = abs(pred - 27.344767)
        return loss

    # Measure forward only
    def measure_forward_only(sim_params, trials=TRIALS, warmup=WARMUP):
        total_time = 0.0
        for i in range(trials + warmup):
            start = time.perf_counter()
            result = full_pipe(sim_params)
            end = time.perf_counter()
            if i >= warmup:
                total_time += (end - start)
        return total_time / trials

    # Measure with gradient
    def measure_with_gradient(sim_params_func, grad_func, trials=TRIALS, warmup=WARMUP):
        total_time = 0.0
        for i in range(trials + warmup):
            sim_params = sim_params_func()
            start = time.perf_counter()
            gradient = grad_func(sim_params)
            end = time.perf_counter()
            if i >= warmup:
                total_time += (end - start)
        return total_time / trials

    results = []

    # Tangent with all optimizations
    print("\nCompiling Tangent gradient with all optimizations...")
    grad_simulate = tangent.grad(simulate, optimized=True,
                                  optimizations={
                                      'dce': True,
                                      'strength_reduction': True,
                                      'cse': True,
                                      'algebraic': True
                                  },
                                  verbose=0)

    def get_sim_params():
        return SIM_PARAMS.copy()

    forward_time = measure_forward_only(SIM_PARAMS)
    gradient_time = measure_with_gradient(get_sim_params, grad_simulate)

    print(f"Forward only: {forward_time:.6f} seconds")
    print(f"Forward + Backward: {gradient_time:.6f} seconds")

    results.append({
        'name': 'Tangent (All Optimizations)',
        'forward': forward_time,
        'gradient': gradient_time,
        'overhead': gradient_time / forward_time
    })

    return results


def run_tensorflow_benchmark():
    """Run TensorFlow benchmark."""
    try:
        import tensorflow as tf
        print("\n" + "=" * 80)
        print("RUNNING TENSORFLOW BENCHMARK")
        print("=" * 80)

        # Suppress TF warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        from benchmarks.building_simulation_tensorflow import run_benchmark
        result = run_benchmark()
        return [result]
    except ImportError:
        print("\n⚠️  TensorFlow not installed, skipping TensorFlow benchmark")
        return []
    except Exception as e:
        print(f"\n⚠️  Error running TensorFlow benchmark: {e}")
        return []


def run_pytorch_benchmark():
    """Run PyTorch benchmark."""
    try:
        import torch
        print("\n" + "=" * 80)
        print("RUNNING PYTORCH BENCHMARK")
        print("=" * 80)

        from benchmarks.building_simulation_pytorch import run_benchmark
        result = run_benchmark()
        return [result]
    except ImportError:
        print("\n⚠️  PyTorch not installed, skipping PyTorch benchmark")
        return []
    except Exception as e:
        print(f"\n⚠️  Error running PyTorch benchmark: {e}")
        return []


def print_comparison_table(results):
    """Print comparison table of all frameworks."""
    print("\n" + "=" * 80)
    print("FRAMEWORK COMPARISON")
    print("=" * 80)
    print(f"{'Framework':<35} {'Forward (s)':<15} {'Gradient (s)':<15} {'Overhead':<10}")
    print("-" * 80)

    # Sort by gradient time (fastest first)
    results_sorted = sorted(results, key=lambda r: r['gradient'])

    for r in results_sorted:
        print(f"{r['name']:<35} {r['forward']:<15.6f} {r['gradient']:<15.6f} {r['overhead']:<10.2f}×")

    print("-" * 80)

    if len(results) > 1:
        print("\nSpeedup Analysis:")
        baseline = results_sorted[-1]  # Slowest as baseline
        for r in results_sorted[:-1]:
            speedup = baseline['gradient'] / r['gradient']
            print(f"  {r['name']} vs {baseline['name']}: {speedup:.2f}×")

    print("=" * 80)

    return results_sorted


def export_results(results, filepath='benchmark_results.txt'):
    """Export benchmark results to a file."""
    import datetime

    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TANGENT FRAMEWORK COMPARISON BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Benchmark: Building Thermal Simulation\n")
        f.write(f"Trials: 100\n")
        f.write(f"Timesteps: 20\n")
        f.write(f"Warmup: 3\n")
        f.write("=" * 80 + "\n\n")

        # Sort by gradient time (fastest first)
        results_sorted = sorted(results, key=lambda r: r['gradient'])

        # Performance table
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Framework':<35} {'Forward (s)':<15} {'Gradient (s)':<15} {'Overhead':<10}\n")
        f.write("-" * 80 + "\n")

        for r in results_sorted:
            f.write(f"{r['name']:<35} {r['forward']:<15.6f} {r['gradient']:<15.6f} {r['overhead']:<10.2f}×\n")

        f.write("-" * 80 + "\n\n")

        # Speedup analysis
        if len(results) > 1:
            f.write("SPEEDUP ANALYSIS (Gradient Computation)\n")
            f.write("-" * 80 + "\n")

            # Find Tangent result
            tangent_result = next((r for r in results if 'Tangent' in r['name']), None)
            tensorflow_result = next((r for r in results if 'TensorFlow' in r['name']), None)
            pytorch_result = next((r for r in results if 'PyTorch' in r['name']), None)

            if tangent_result:
                f.write(f"\nTangent (baseline):\n")
                f.write(f"  Forward time:  {tangent_result['forward']:.6f} seconds\n")
                f.write(f"  Gradient time: {tangent_result['gradient']:.6f} seconds\n")
                f.write(f"  Overhead:      {tangent_result['overhead']:.2f}×\n")

                if tensorflow_result:
                    speedup_grad = tensorflow_result['gradient'] / tangent_result['gradient']
                    speedup_fwd = tensorflow_result['forward'] / tangent_result['forward']
                    f.write(f"\nTangent vs TensorFlow:\n")
                    f.write(f"  Tangent is {speedup_grad:.2f}× {'faster' if speedup_grad > 1 else 'slower'} for gradients\n")
                    f.write(f"  Tangent is {speedup_fwd:.2f}× {'faster' if speedup_fwd > 1 else 'slower'} for forward pass\n")
                    if speedup_grad < 1:
                        improvement = (1 - speedup_grad) * 100
                        f.write(f"  (TensorFlow is {improvement:.1f}% slower)\n")
                    else:
                        improvement = (speedup_grad - 1) * 100
                        f.write(f"  (TensorFlow is {improvement:.1f}% faster)\n")

                if pytorch_result:
                    speedup_grad = pytorch_result['gradient'] / tangent_result['gradient']
                    speedup_fwd = pytorch_result['forward'] / tangent_result['forward']
                    f.write(f"\nTangent vs PyTorch:\n")
                    f.write(f"  Tangent is {speedup_grad:.2f}× {'faster' if speedup_grad > 1 else 'slower'} for gradients\n")
                    f.write(f"  Tangent is {speedup_fwd:.2f}× {'faster' if speedup_fwd > 1 else 'slower'} for forward pass\n")
                    improvement = (speedup_grad - 1) * 100
                    f.write(f"  (Tangent is {improvement:.1f}% faster overall)\n")

            f.write("\n" + "-" * 80 + "\n\n")

            # Relative comparison
            f.write("RELATIVE SPEEDUPS\n")
            f.write("-" * 80 + "\n")
            baseline = results_sorted[-1]  # Slowest
            f.write(f"Baseline (slowest): {baseline['name']} - {baseline['gradient']:.6f}s\n\n")

            for r in results_sorted[:-1]:
                speedup = baseline['gradient'] / r['gradient']
                f.write(f"{r['name']:<35} {speedup:.2f}× faster than {baseline['name']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION DETAILS\n")
        f.write("=" * 80 + "\n")
        f.write("Tangent optimizations enabled:\n")
        f.write("  - Dead Code Elimination (DCE)\n")
        f.write("  - Strength Reduction (x**2 → x*x)\n")
        f.write("  - Common Subexpression Elimination (CSE)\n")
        f.write("  - Algebraic Simplification\n")
        f.write("\nTensorFlow optimizations:\n")
        f.write("  - @tf.function graph compilation\n")
        f.write("  - XLA optimization (automatic)\n")
        f.write("\nPyTorch optimizations:\n")
        f.write("  - Eager execution (baseline)\n")
        f.write("  - No JIT compilation applied\n")
        f.write("=" * 80 + "\n")

    print(f"\n✅ Results exported to: {filepath}")


def main():
    """Run all benchmarks and compare."""
    print("\n" + "=" * 80)
    print("BUILDING SIMULATION BENCHMARK - FRAMEWORK COMPARISON")
    print("=" * 80)
    print("Comparing Tangent, TensorFlow, and PyTorch on building thermal simulation")
    print("=" * 80)

    all_results = []

    # Run Tangent
    tangent_results = run_tangent_benchmark()
    all_results.extend(tangent_results)

    # Run TensorFlow
    tf_results = run_tensorflow_benchmark()
    all_results.extend(tf_results)

    # Run PyTorch
    pytorch_results = run_pytorch_benchmark()
    all_results.extend(pytorch_results)

    # Print comparison
    if len(all_results) > 0:
        results_sorted = print_comparison_table(all_results)

        # Export results to file
        export_results(all_results, filepath='benchmarks/benchmark_results.txt')
    else:
        print("\n⚠️  No benchmarks completed successfully")


if __name__ == '__main__':
    main()
