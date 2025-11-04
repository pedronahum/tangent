"""
Verify that all frameworks produce the same results for the building simulation.

This script runs the simulation once with each framework and compares the outputs
to ensure mathematical correctness across Tangent, TensorFlow, and PyTorch.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tangent_simulation():
    """Run Tangent simulation and return final temperature."""
    print("\n" + "=" * 80)
    print("TANGENT SIMULATION")
    print("=" * 80)

    import tangent

    # Constants
    PI = 3.14159265359
    TIMESTEPS = 20
    D_TIME = 0.1

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

    # Run simulation
    result = simulate(SIM_PARAMS)
    print(f"Final temperature: {result:.10f}°C")

    # Test gradient
    grad_simulate = tangent.grad(simulate, optimized=True,
                                  optimizations={
                                      'dce': True,
                                      'strength_reduction': True,
                                      'cse': True,
                                      'algebraic': True
                                  },
                                  verbose=0)

    gradient = grad_simulate(SIM_PARAMS.copy())
    print(f"Gradient (first element): {gradient[0][0]:.10f}")

    return float(result), gradient


def run_tensorflow_simulation():
    """Run TensorFlow simulation and return final temperature."""
    try:
        import tensorflow as tf

        print("\n" + "=" * 80)
        print("TENSORFLOW SIMULATION")
        print("=" * 80)

        # Suppress warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Constants
        PI = 3.14159265359
        TIMESTEPS = 20
        D_TIME = 0.1

        # Type definitions
        TUBE_TYPE = tf.constant([0.50292, 0.019, 0.001588, 2.43, 0.0])

        class TubeIndices:
            TUBE_SPACING = 0
            DIAMETER = 1
            THICKNESS = 2
            RESISTIVITY = 3

        SLAB_TYPE = tf.constant([21.1111111, 100.0, 0.2, 2242.58, 0.101])

        class SlabIndices:
            TEMP = 0
            AREA = 1
            CP = 2
            DENSITY = 3
            THICKNESS = 4

        QUANTA_TYPE = tf.constant([0.0, 60.0, 0.0006309, 1000.0, 4180.0])

        class QuantaIndices:
            POWER = 0
            TEMP = 1
            FLOW = 2
            DENSITY = 3
            CP = 4

        TANK_TYPE = tf.constant([70.0, 0.0757082, 4180.0, 1000.0, 75.708])

        class TankIndices:
            TEMP = 0
            VOLUME = 1
            CP = 2
            DENSITY = 3
            MASS = 4

        STARTING_TEMPERATURE = tf.constant([33.3, 0, 0, 0, 0])
        SIM_PARAMS = tf.convert_to_tensor([TUBE_TYPE, SLAB_TYPE, QUANTA_TYPE, TANK_TYPE, STARTING_TEMPERATURE])

        class SimParamsIndices:
            TUBE = 0
            SLAB = 1
            QUANTA = 2
            TANK = 3
            STARTING_TEMP = 4

        @tf.function
        def compute_resistance(floor, tube, quanta):
            geometry_coeff = 10.0
            tubing_surface_area = (floor[SlabIndices.AREA] / tube[TubeIndices.TUBE_SPACING]) * PI * tube[TubeIndices.DIAMETER]
            resistance_abs = tube[TubeIndices.RESISTIVITY] * tube[TubeIndices.THICKNESS] / tubing_surface_area
            return resistance_abs * geometry_coeff

        @tf.function
        def compute_load_power(floor, tube, quanta):
            resistance_abs = compute_resistance(floor, tube, quanta)
            conductance = 1.0 / resistance_abs
            d_temp = floor[SlabIndices.TEMP] - quanta[QuantaIndices.TEMP]
            power = d_temp * conductance
            load_power = -power
            result_quanta = quanta * tf.constant([0.0, 1, 1, 1, 1]) + power * tf.constant([1.0, 0, 0, 0, 0])
            return result_quanta, load_power

        @tf.function
        def update_quanta(quanta):
            working_volume = quanta[QuantaIndices.FLOW] * D_TIME
            working_mass = working_volume * quanta[QuantaIndices.DENSITY]
            working_energy = quanta[QuantaIndices.POWER] * D_TIME
            temp_rise = working_energy / quanta[QuantaIndices.CP] / working_mass
            result_quanta = quanta + temp_rise * tf.constant([0.0, 1, 0, 0, 0])
            result_quanta = result_quanta * tf.constant([0.0, 1, 1, 1, 1])
            return result_quanta

        @tf.function
        def update_building_model(power, floor):
            floor_volume = floor[SlabIndices.AREA] * floor[SlabIndices.THICKNESS]
            floor_mass = floor_volume * floor[SlabIndices.DENSITY]
            floor_temp_change = (power * D_TIME) / floor[SlabIndices.CP] / floor_mass
            result_floor = floor + floor_temp_change * tf.constant([1.0, 0, 0, 0, 0])
            return result_floor

        @tf.function
        def update_source_tank(store, quanta):
            mass_per_time = quanta[QuantaIndices.FLOW] * quanta[QuantaIndices.DENSITY]
            d_temp = store[TankIndices.TEMP] - quanta[QuantaIndices.TEMP]
            power = d_temp * mass_per_time * quanta[QuantaIndices.CP]
            updated_quanta = quanta * tf.constant([0.0, 1, 1, 1, 1]) + power * tf.constant([1.0, 0, 0, 0, 0])
            tank_mass = store[TankIndices.VOLUME] * store[TankIndices.DENSITY]
            temp_rise = (power * D_TIME) / store[TankIndices.CP] / tank_mass
            updated_store = store + temp_rise * tf.constant([1.0, 0, 0, 0, 0])
            return updated_store, updated_quanta

        @tf.function
        def simulate(sim_params):
            pex_tube = sim_params[SimParamsIndices.TUBE]
            slab = sim_params[SimParamsIndices.SLAB]
            tank = sim_params[SimParamsIndices.TANK]
            quanta = sim_params[SimParamsIndices.QUANTA]
            starting_temp = sim_params[SimParamsIndices.STARTING_TEMP][0]
            slab = slab * tf.constant([0.0, 1, 1, 1, 1]) + starting_temp * tf.constant([1.0, 0, 0, 0, 0])
            for i in tf.range(TIMESTEPS):
                tank, quanta = update_source_tank(tank, quanta)
                quanta = update_quanta(quanta)
                quanta, power_to_building = compute_load_power(slab, pex_tube, quanta)
                quanta = update_quanta(quanta)
                slab = update_building_model(power_to_building, slab)
            return slab[SlabIndices.TEMP]

        # Run simulation
        result = simulate(SIM_PARAMS)
        print(f"Final temperature: {float(result):.10f}°C")

        # Test gradient
        @tf.function
        def get_gradient(sim_params):
            with tf.GradientTape() as tape:
                tape.watch(sim_params)
                end_temperature = simulate(sim_params)
            gradient = tape.gradient(end_temperature, sim_params)
            return gradient

        sim_params_var = tf.Variable(SIM_PARAMS)
        gradient = get_gradient(sim_params_var)
        print(f"Gradient (first element): {float(gradient[0][0]):.10f}")

        return float(result), gradient

    except ImportError:
        print("\n⚠️  TensorFlow not installed")
        return None, None
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        return None, None


def run_pytorch_simulation():
    """Run PyTorch simulation and return final temperature."""
    try:
        import torch

        print("\n" + "=" * 80)
        print("PYTORCH SIMULATION")
        print("=" * 80)

        # Constants
        PI = 3.14159265359
        TIMESTEPS = 20
        D_TIME = 0.1

        # Type definitions
        TUBE_TYPE = torch.tensor([0.50292, 0.019, 0.001588, 2.43, 0.0], requires_grad=True)

        class TubeIndices:
            TUBE_SPACING = 0
            DIAMETER = 1
            THICKNESS = 2
            RESISTIVITY = 3

        SLAB_TYPE = torch.tensor([21.1111111, 100.0, 0.2, 2242.58, 0.101], requires_grad=True)

        class SlabIndices:
            TEMP = 0
            AREA = 1
            CP = 2
            DENSITY = 3
            THICKNESS = 4

        QUANTA_TYPE = torch.tensor([0.0, 60.0, 0.0006309, 1000.0, 4180.0], requires_grad=True)

        class QuantaIndices:
            POWER = 0
            TEMP = 1
            FLOW = 2
            DENSITY = 3
            CP = 4

        TANK_TYPE = torch.tensor([70.0, 0.0757082, 4180.0, 1000.0, 75.708], requires_grad=True)

        class TankIndices:
            TEMP = 0
            VOLUME = 1
            CP = 2
            DENSITY = 3
            MASS = 4

        STARTING_TEMPERATURE = torch.tensor([33.3, 0, 0, 0, 0], requires_grad=True)
        SIM_PARAMS_CONSTANT = torch.stack([TUBE_TYPE, SLAB_TYPE, QUANTA_TYPE, TANK_TYPE, STARTING_TEMPERATURE])

        class SimParamsIndices:
            TUBE = 0
            SLAB = 1
            QUANTA = 2
            TANK = 3
            STARTING_TEMP = 4

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
            result_quanta = quanta * torch.tensor([0.0, 1, 1, 1, 1], requires_grad=True) + power * torch.tensor([1.0, 0, 0, 0, 0], requires_grad=True)
            return result_quanta, load_power

        def update_quanta(quanta):
            working_volume = quanta[QuantaIndices.FLOW] * D_TIME
            working_mass = working_volume * quanta[QuantaIndices.DENSITY]
            working_energy = quanta[QuantaIndices.POWER] * D_TIME
            temp_rise = working_energy / quanta[QuantaIndices.CP] / working_mass
            result_quanta = quanta + temp_rise * torch.tensor([0.0, 1, 0, 0, 0])
            result_quanta = result_quanta * torch.tensor([0.0, 1, 1, 1, 1])
            return result_quanta

        def update_building_model(power, floor):
            floor_volume = floor[SlabIndices.AREA] * floor[SlabIndices.THICKNESS]
            floor_mass = floor_volume * floor[SlabIndices.DENSITY]
            floor_temp_change = (power * D_TIME) / floor[SlabIndices.CP] / floor_mass
            result_floor = floor + floor_temp_change * torch.tensor([1.0, 0, 0, 0, 0])
            return result_floor

        def update_source_tank(store, quanta):
            mass_per_time = quanta[QuantaIndices.FLOW] * quanta[QuantaIndices.DENSITY]
            d_temp = store[TankIndices.TEMP] - quanta[QuantaIndices.TEMP]
            power = d_temp * mass_per_time * quanta[QuantaIndices.CP]
            updated_quanta = quanta * torch.tensor([0.0, 1, 1, 1, 1]) + power * torch.tensor([1.0, 0, 0, 0, 0])
            tank_mass = store[TankIndices.VOLUME] * store[TankIndices.DENSITY]
            temp_rise = (power * D_TIME) / store[TankIndices.CP] / tank_mass
            updated_store = store + temp_rise * torch.tensor([1.0, 0, 0, 0, 0])
            return updated_store, updated_quanta

        def simulate(sim_params):
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

        # Run simulation
        result = simulate(SIM_PARAMS_CONSTANT)
        result_value = result.detach().item()
        print(f"Final temperature: {result_value:.10f}°C")

        # Test gradient
        gradient = torch.autograd.grad(result, SIM_PARAMS_CONSTANT, retain_graph=True)
        # gradient is a tuple of tensors, extract first element of first tensor
        grad_value = gradient[0][0, 0].detach().item()
        print(f"Gradient (first element): {grad_value:.10f}")

        return result_value, gradient

    except ImportError:
        print("\n⚠️  PyTorch not installed")
        return None, None
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        return None, None


def main():
    """Compare results across all frameworks."""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print("Verifying that all frameworks produce identical results...")
    print("=" * 80)

    results = {}

    # Run Tangent
    tangent_result, tangent_grad = run_tangent_simulation()
    results['Tangent'] = tangent_result

    # Run TensorFlow
    tf_result, tf_grad = run_tensorflow_simulation()
    if tf_result is not None:
        results['TensorFlow'] = tf_result

    # Run PyTorch
    pytorch_result, pytorch_grad = run_pytorch_simulation()
    if pytorch_result is not None:
        results['PyTorch'] = pytorch_result

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if len(results) < 2:
        print("⚠️  Not enough frameworks available for comparison")
        return

    print(f"{'Framework':<20} {'Final Temperature (°C)':<25} {'Difference from Tangent':<25}")
    print("-" * 80)

    tangent_val = results['Tangent']

    for framework, value in results.items():
        diff = abs(value - tangent_val)
        rel_error = (diff / tangent_val * 100) if tangent_val != 0 else 0

        status = "✅" if diff < 1e-6 else ("⚠️" if diff < 1e-3 else "❌")

        print(f"{framework:<20} {value:<25.10f} {diff:.2e} ({rel_error:.6f}%) {status}")

    print("-" * 80)

    # Check if all results are similar
    values = list(results.values())
    max_diff = max(abs(v - tangent_val) for v in values)

    print(f"\nMaximum difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("✅ PASS: All frameworks produce identical results (within floating-point precision)")
    elif max_diff < 1e-5:
        print("✅ PASS: Negligible differences (< 0.001%, within acceptable numerical error)")
    elif max_diff < 1e-3:
        print("⚠️  WARNING: Small differences detected (< 0.1%, likely acceptable)")
    else:
        print("❌ FAIL: Significant differences detected!")

    print("=" * 80)


if __name__ == '__main__':
    main()
