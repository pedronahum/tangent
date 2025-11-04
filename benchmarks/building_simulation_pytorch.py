"""
Building Simulation Benchmark - PyTorch Implementation

This implements the same building thermal simulation as:
https://github.com/PassiveLogic/differentiable-swift-examples/

Benchmarks PyTorch's automatic differentiation on the building simulation.
"""

import torch
import time

# Simulation parameters
TRIALS = 100
TIMESTEPS = 20
WARMUP = 3
D_TIME = 0.1
PRINT_GRAD = False

# Constants
PI = 3.14159265359

# Type definitions (represented as PyTorch tensors)
# TubeType: [tubeSpacing, diameter, thickness, resistivity, padding]
TUBE_TYPE = torch.tensor([0.50292, 0.019, 0.001588, 2.43, 0.0], requires_grad=True)

class TubeIndices:
    TUBE_SPACING = 0
    DIAMETER = 1
    THICKNESS = 2
    RESISTIVITY = 3

# SlabType: [temp, area, Cp, density, thickness]
SLAB_TYPE = torch.tensor([21.1111111, 100.0, 0.2, 2242.58, 0.101], requires_grad=True)

class SlabIndices:
    TEMP = 0
    AREA = 1
    CP = 2
    DENSITY = 3
    THICKNESS = 4

# QuantaType: [power, temp, flow, density, Cp]
QUANTA_TYPE = torch.tensor([0.0, 60.0, 0.0006309, 1000.0, 4180.0], requires_grad=True)

class QuantaIndices:
    POWER = 0
    TEMP = 1
    FLOW = 2
    DENSITY = 3
    CP = 4

# TankType: [temp, volume, Cp, density, mass]
TANK_TYPE = torch.tensor([70.0, 0.0757082, 4180.0, 1000.0, 75.708], requires_grad=True)

class TankIndices:
    TEMP = 0
    VOLUME = 1
    CP = 2
    DENSITY = 3
    MASS = 4

# Starting temperature (padded to 5 elements)
STARTING_TEMPERATURE = torch.tensor([33.3, 0, 0, 0, 0], requires_grad=True)

# SimParams: [tube, slab, quanta, tank, startingTemp]
SIM_PARAMS_CONSTANT = torch.stack([TUBE_TYPE, SLAB_TYPE, QUANTA_TYPE, TANK_TYPE, STARTING_TEMPERATURE])

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

    result_quanta = quanta * torch.tensor([0.0, 1, 1, 1, 1], requires_grad=True) + power * torch.tensor([1.0, 0, 0, 0, 0], requires_grad=True)

    return result_quanta, load_power


def update_quanta(quanta):
    """Update fluid temperature based on power and flow."""
    working_volume = quanta[QuantaIndices.FLOW] * D_TIME
    working_mass = working_volume * quanta[QuantaIndices.DENSITY]
    working_energy = quanta[QuantaIndices.POWER] * D_TIME
    temp_rise = working_energy / quanta[QuantaIndices.CP] / working_mass

    result_quanta = quanta + temp_rise * torch.tensor([0.0, 1, 0, 0, 0])
    result_quanta = result_quanta * torch.tensor([0.0, 1, 1, 1, 1])

    return result_quanta


def update_building_model(power, floor):
    """Update building thermal mass temperature."""
    floor_volume = floor[SlabIndices.AREA] * floor[SlabIndices.THICKNESS]
    floor_mass = floor_volume * floor[SlabIndices.DENSITY]
    floor_temp_change = (power * D_TIME) / floor[SlabIndices.CP] / floor_mass

    result_floor = floor + floor_temp_change * torch.tensor([1.0, 0, 0, 0, 0])

    return result_floor


def update_source_tank(store, quanta):
    """Update source tank and fluid flowing from it."""
    mass_per_time = quanta[QuantaIndices.FLOW] * quanta[QuantaIndices.DENSITY]
    d_temp = store[TankIndices.TEMP] - quanta[QuantaIndices.TEMP]
    power = d_temp * mass_per_time * quanta[QuantaIndices.CP]

    updated_quanta = quanta * torch.tensor([0.0, 1, 1, 1, 1]) + power * torch.tensor([1.0, 0, 0, 0, 0])

    tank_mass = store[TankIndices.VOLUME] * store[TankIndices.DENSITY]
    temp_rise = (power * D_TIME) / store[TankIndices.CP] / tank_mass

    updated_store = store + temp_rise * torch.tensor([1.0, 0, 0, 0, 0])

    return updated_store, updated_quanta


def simulate(sim_params):
    """Run full building simulation."""
    pex_tube = sim_params[SimParamsIndices.TUBE]
    slab = sim_params[SimParamsIndices.SLAB]
    tank = sim_params[SimParamsIndices.TANK]
    quanta = sim_params[SimParamsIndices.QUANTA]

    starting_temp = sim_params[SimParamsIndices.STARTING_TEMP][0]
    slab = slab * torch.tensor([0.0, 1, 1, 1, 1]) + starting_temp * torch.tensor([1.0, 0, 0, 0, 0])

    for i in range(TIMESTEPS):
        tank, quanta = update_source_tank(tank, quanta)
        quanta = update_quanta(quanta)
        quanta, power_to_building = compute_load_power(slab, pex_tube, quanta)
        quanta = update_quanta(quanta)
        slab = update_building_model(power_to_building, slab)

    return slab[SlabIndices.TEMP]


def loss_calc(pred, gt):
    """Calculate loss (absolute error)."""
    return torch.abs(pred - gt)


def full_pipe(sim_params):
    """Full forward pass with loss."""
    pred = simulate(sim_params)
    loss = loss_calc(pred, 27.344767)
    return loss


# ============================================================================
# Benchmarking Functions
# ============================================================================

def measure(function, arguments):
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = function(arguments)
    end = time.perf_counter()
    return (end - start, result)


def run_benchmark():
    """Run PyTorch benchmark."""
    print("\n" + "=" * 80)
    print("BUILDING SIMULATION BENCHMARK - PYTORCH")
    print("=" * 80)
    print(f"Trials: {TRIALS}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Warmup iterations: {WARMUP}")
    print("=" * 80)

    total_forward_time = 0.0
    total_gradient_time = 0.0

    for i in range(TRIALS + WARMUP):
        # Forward pass
        inputs = SIM_PARAMS_CONSTANT
        forward_time, forward_output = measure(full_pipe, inputs)

        # Backward pass
        def get_gradient(sim_params):
            gradient = torch.autograd.grad(forward_output, inputs, retain_graph=True)
            return gradient

        gradient_time, gradient = measure(get_gradient, SIM_PARAMS_CONSTANT)

        if PRINT_GRAD:
            print(gradient)

        if i >= WARMUP:
            total_forward_time += forward_time
            total_gradient_time += gradient_time

    avg_forward_time = total_forward_time / TRIALS
    avg_gradient_time = total_gradient_time / TRIALS

    print(f"\nForward only: {avg_forward_time:.6f} seconds")
    print(f"Forward + Backward: {avg_gradient_time:.6f} seconds")
    print(f"Speedup (overhead): {avg_gradient_time / avg_forward_time:.2f}×")
    print("=" * 80)

    return {
        'name': 'PyTorch',
        'forward': avg_forward_time,
        'gradient': avg_gradient_time,
        'overhead': avg_gradient_time / avg_forward_time
    }


if __name__ == '__main__':
    result = run_benchmark()
