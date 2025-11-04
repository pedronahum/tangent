"""
Building Simulation Benchmark - Tangent Implementation

This implements the same building thermal simulation as:
https://github.com/PassiveLogic/differentiable-swift-examples/

Compares Tangent (with various optimization levels) against TensorFlow and PyTorch.
"""

import numpy as np
import tangent
import time
from typing import Tuple


# Simulation parameters
TRIALS = 100
TIMESTEPS = 20
WARMUP = 3
D_TIME = 0.1
PRINT_GRAD = False

# Constants
PI = 3.14159265359

# Type definitions (represented as numpy arrays)
# TubeType: [tubeSpacing, diameter, thickness, resistivity, padding]
TUBE_TYPE = np.array([0.50292, 0.019, 0.001588, 2.43, 0.0])

class TubeIndices:
    TUBE_SPACING = 0
    DIAMETER = 1
    THICKNESS = 2
    RESISTIVITY = 3

# SlabType: [temp, area, Cp, density, thickness]
SLAB_TYPE = np.array([21.1111111, 100.0, 0.2, 2242.58, 0.101])

class SlabIndices:
    TEMP = 0
    AREA = 1
    CP = 2
    DENSITY = 3
    THICKNESS = 4

# QuantaType: [power, temp, flow, density, Cp]
QUANTA_TYPE = np.array([0.0, 60.0, 0.0006309, 1000.0, 4180.0])

class QuantaIndices:
    POWER = 0
    TEMP = 1
    FLOW = 2
    DENSITY = 3
    CP = 4

# TankType: [temp, volume, Cp, density, mass]
TANK_TYPE = np.array([70.0, 0.0757082, 4180.0, 1000.0, 75.708])

class TankIndices:
    TEMP = 0
    VOLUME = 1
    CP = 2
    DENSITY = 3
    MASS = 4

# Starting temperature (padded to 5 elements)
STARTING_TEMPERATURE = np.array([33.3, 0, 0, 0, 0])

# SimParams: [tube, slab, quanta, tank, startingTemp]
SIM_PARAMS = np.array([TUBE_TYPE, SLAB_TYPE, QUANTA_TYPE, TANK_TYPE, STARTING_TEMPERATURE])

class SimParamsIndices:
    TUBE = 0
    SLAB = 1
    QUANTA = 2
    TANK = 3
    STARTING_TEMP = 4


# ============================================================================
# Simulation Functions
# ============================================================================

def compute_resistance(floor, tube, quanta):
    """Compute thermal resistance of the floor tubing."""
    geometry_coeff = 10.0

    tubing_surface_area = (floor[SlabIndices.AREA] / tube[TubeIndices.TUBE_SPACING]) * PI * tube[TubeIndices.DIAMETER]
    resistance_abs = tube[TubeIndices.RESISTIVITY] * tube[TubeIndices.THICKNESS] / tubing_surface_area
    resistance_corrected = resistance_abs * geometry_coeff

    return resistance_corrected


def compute_load_power(floor, tube, quanta):
    """Compute power transferred to/from the floor."""
    resistance_abs = compute_resistance(floor, tube, quanta)

    conductance = 1.0 / resistance_abs
    d_temp = floor[SlabIndices.TEMP] - quanta[QuantaIndices.TEMP]
    power = d_temp * conductance

    load_power = -power

    # Update quanta with power (avoid .copy() for Tangent compatibility)
    result_quanta = quanta * np.array([0.0, 1, 1, 1, 1]) + power * np.array([1.0, 0, 0, 0, 0])

    return result_quanta, load_power


def update_quanta(quanta):
    """Update fluid temperature based on power and flow."""
    working_volume = quanta[QuantaIndices.FLOW] * D_TIME
    working_mass = working_volume * quanta[QuantaIndices.DENSITY]
    working_energy = quanta[QuantaIndices.POWER] * D_TIME
    temp_rise = working_energy / quanta[QuantaIndices.CP] / working_mass

    # Tangent-compatible: [power, temp, flow, density, Cp]
    new_temp = quanta[QuantaIndices.TEMP] + temp_rise
    result_quanta = (quanta * np.array([0.0, 0.0, 1, 1, 1]) +
                     new_temp * np.array([0.0, 1.0, 0, 0, 0]))

    return result_quanta


def update_building_model(power, floor):
    """Update building thermal mass temperature."""
    floor_volume = floor[SlabIndices.AREA] * floor[SlabIndices.THICKNESS]
    floor_mass = floor_volume * floor[SlabIndices.DENSITY]
    floor_temp_change = (power * D_TIME) / floor[SlabIndices.CP] / floor_mass

    # Tangent-compatible: [temp, area, Cp, density, thickness]
    new_temp = floor[SlabIndices.TEMP] + floor_temp_change
    result_floor = (floor * np.array([0.0, 1, 1, 1, 1]) +
                    new_temp * np.array([1.0, 0, 0, 0, 0]))

    return result_floor


def update_source_tank(store, quanta):
    """Update source tank and fluid flowing from it."""
    mass_per_time = quanta[QuantaIndices.FLOW] * quanta[QuantaIndices.DENSITY]
    d_temp = store[TankIndices.TEMP] - quanta[QuantaIndices.TEMP]
    power = d_temp * mass_per_time * quanta[QuantaIndices.CP]

    # Tangent-compatible: [power, temp, flow, density, Cp]
    updated_quanta = (quanta * np.array([0.0, 1, 1, 1, 1]) +
                      power * np.array([1.0, 0, 0, 0, 0]))

    tank_mass = store[TankIndices.VOLUME] * store[TankIndices.DENSITY]
    temp_rise = (power * D_TIME) / store[TankIndices.CP] / tank_mass

    # Tangent-compatible: [temp, volume, Cp, density, mass]
    new_tank_temp = store[TankIndices.TEMP] + temp_rise
    updated_store = (store * np.array([0.0, 1, 1, 1, 1]) +
                     new_tank_temp * np.array([1.0, 0, 0, 0, 0]))

    return updated_store, updated_quanta


def simulate(sim_params):
    """Run full building simulation."""
    # Tangent-compatible: extract arrays without .copy()
    pex_tube = sim_params[SimParamsIndices.TUBE] + 0.0
    slab = sim_params[SimParamsIndices.SLAB] + 0.0
    tank = sim_params[SimParamsIndices.TANK] + 0.0
    quanta = sim_params[SimParamsIndices.QUANTA] + 0.0

    starting_temp = sim_params[SimParamsIndices.STARTING_TEMP][0]
    # Tangent-compatible: [temp, area, Cp, density, thickness]
    slab = (slab * np.array([0.0, 1, 1, 1, 1]) +
            starting_temp * np.array([1.0, 0, 0, 0, 0]))

    for i in range(TIMESTEPS):
        # Update source tank
        tank, quanta = update_source_tank(tank, quanta)

        # Update quanta
        quanta = update_quanta(quanta)

        # Compute load power
        quanta, power_to_building = compute_load_power(slab, pex_tube, quanta)

        # Update quanta again
        quanta = update_quanta(quanta)

        # Update building model
        slab = update_building_model(power_to_building, slab)

    return slab[SlabIndices.TEMP]


def loss_calc(pred, gt):
    """Calculate loss (absolute error)."""
    return abs(pred - gt)


def full_pipe(sim_params):
    """Full forward pass with loss."""
    pred = simulate(sim_params)
    loss = loss_calc(pred, 27.344767)
    return loss


# ============================================================================
# Benchmarking Functions
# ============================================================================

def measure_forward_only(sim_params, trials=TRIALS, warmup=WARMUP):
    """Measure forward pass only."""
    total_time = 0.0

    for i in range(trials + warmup):
        start = time.perf_counter()
        result = full_pipe(sim_params)
        end = time.perf_counter()

        if i >= warmup:
            total_time += (end - start)

    avg_time = total_time / trials
    return avg_time, result


def measure_with_gradient(sim_params_func, grad_func, trials=TRIALS, warmup=WARMUP):
    """Measure forward + backward pass."""
    total_time = 0.0
    last_grad = None

    for i in range(trials + warmup):
        # Create fresh sim params for this trial
        sim_params = sim_params_func()

        start = time.perf_counter()
        gradient = grad_func(sim_params)
        end = time.perf_counter()

        if i >= warmup:
            total_time += (end - start)
            last_grad = gradient

    avg_time = total_time / trials
    return avg_time, last_grad


# ============================================================================
# Tangent Gradient Compilation
# ============================================================================

def benchmark_tangent_numpy():
    """Benchmark pure NumPy/Tangent without optimizations."""
    print("\n" + "=" * 80)
    print("TANGENT BENCHMARK: NumPy (No Optimizations)")
    print("=" * 80)

    # Forward only
    forward_time, _ = measure_forward_only(SIM_PARAMS)
    print(f"Forward only: {forward_time:.6f} seconds")

    # Forward + backward (compile gradient)
    print("Compiling gradient...")
    grad_simulate = tangent.grad(simulate, optimized=False, verbose=0)

    def get_sim_params():
        return SIM_PARAMS.copy()

    gradient_time, grad = measure_with_gradient(get_sim_params, grad_simulate)
    print(f"Forward + Backward: {gradient_time:.6f} seconds")

    if PRINT_GRAD:
        print(f"Gradient: {grad}")

    print(f"Speedup vs forward-only: {gradient_time / forward_time:.2f}×")
    print("=" * 80)

    return {
        'name': 'Tangent (NumPy, No Opt)',
        'forward': forward_time,
        'gradient': gradient_time,
        'speedup': gradient_time / forward_time
    }


def benchmark_tangent_standard_opt():
    """Benchmark Tangent with standard optimizations (DCE)."""
    print("\n" + "=" * 80)
    print("TANGENT BENCHMARK: Standard Optimizations (DCE)")
    print("=" * 80)

    # Forward only
    forward_time, _ = measure_forward_only(SIM_PARAMS)
    print(f"Forward only: {forward_time:.6f} seconds")

    # Forward + backward with DCE
    print("Compiling gradient with DCE...")
    grad_simulate = tangent.grad(simulate, optimized=True,
                                  optimizations={'dce': True},
                                  verbose=0)

    def get_sim_params():
        return SIM_PARAMS.copy()

    gradient_time, grad = measure_with_gradient(get_sim_params, grad_simulate)
    print(f"Forward + Backward (with DCE): {gradient_time:.6f} seconds")

    if PRINT_GRAD:
        print(f"Gradient: {grad}")

    print(f"Speedup vs forward-only: {gradient_time / forward_time:.2f}×")
    print("=" * 80)

    return {
        'name': 'Tangent (DCE)',
        'forward': forward_time,
        'gradient': gradient_time,
        'speedup': gradient_time / forward_time
    }


def benchmark_tangent_all_opt():
    """Benchmark Tangent with all symbolic optimizations."""
    print("\n" + "=" * 80)
    print("TANGENT BENCHMARK: All Symbolic Optimizations")
    print("=" * 80)

    # Forward only
    forward_time, _ = measure_forward_only(SIM_PARAMS)
    print(f"Forward only: {forward_time:.6f} seconds")

    # Forward + backward with all optimizations
    print("Compiling gradient with full optimization stack...")
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

    gradient_time, grad = measure_with_gradient(get_sim_params, grad_simulate)
    print(f"Forward + Backward (all opts): {gradient_time:.6f} seconds")

    if PRINT_GRAD:
        print(f"Gradient: {grad}")

    print(f"Speedup vs forward-only: {gradient_time / forward_time:.2f}×")
    print("=" * 80)

    return {
        'name': 'Tangent (All Opts)',
        'forward': forward_time,
        'gradient': gradient_time,
        'speedup': gradient_time / forward_time
    }


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_all_benchmarks():
    """Run all Tangent benchmarks and generate report."""
    print("\n" + "=" * 80)
    print("BUILDING SIMULATION BENCHMARK - TANGENT")
    print("=" * 80)
    print(f"Trials: {TRIALS}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Warmup iterations: {WARMUP}")
    print("=" * 80)

    results = []

    # Run benchmarks
    results.append(benchmark_tangent_numpy())
    results.append(benchmark_tangent_standard_opt())
    results.append(benchmark_tangent_all_opt())

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Forward (s)':<15} {'Gradient (s)':<15} {'Overhead':<10}")
    print("-" * 80)

    baseline_gradient = results[0]['gradient']

    for r in results:
        overhead = r['gradient'] / r['forward']
        speedup_vs_baseline = baseline_gradient / r['gradient']
        print(f"{r['name']:<35} {r['forward']:<15.6f} {r['gradient']:<15.6f} {overhead:<10.2f}×")

    print("-" * 80)
    print("\nOptimization Impact:")
    print(f"  DCE speedup vs baseline:       {baseline_gradient / results[1]['gradient']:.2f}×")
    print(f"  All opts speedup vs baseline:  {baseline_gradient / results[2]['gradient']:.2f}×")
    print(f"  All opts speedup vs DCE:       {results[1]['gradient'] / results[2]['gradient']:.2f}×")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = run_all_benchmarks()
