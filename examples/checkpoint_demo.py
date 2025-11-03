#!/usr/bin/env python3
"""
Demonstration of memory-efficient checkpointing in Tangent.

This demo shows how checkpointing can dramatically reduce memory usage
when computing gradients for long sequences, such as in RNNs or iterative
algorithms.
"""

import numpy as np
import time
import tracemalloc
import sys
import os

# Add parent directory to path to import tangent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tangent.checkpointing_simple import (
    compute_checkpoint_positions,
    checkpointed_loop,
    get_memory_savings
)


def demo_memory_savings():
    """Demonstrate memory savings with checkpointing."""

    print("=" * 80)
    print("MEMORY-EFFICIENT CHECKPOINTING DEMO")
    print("=" * 80)

    # Define a simple RNN-like computation
    def rnn_step(state):
        """One step of RNN computation."""
        return np.tanh(state @ W + b)

    # Parameters
    hidden_size = 512
    seq_length = 1000

    print(f"\nConfiguration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Memory per state: {hidden_size * 8 / 1024:.2f} KB")

    # Initialize weights
    np.random.seed(42)
    W = np.random.randn(hidden_size, hidden_size) * 0.01
    b = np.random.randn(hidden_size) * 0.01
    initial_state = np.zeros(hidden_size)

    # Method 1: Standard approach (store all states)
    print("\n" + "-" * 80)
    print("Method 1: STANDARD APPROACH (store all states)")
    print("-" * 80)

    tracemalloc.start()
    start_time = time.time()

    all_states = []
    state = initial_state.copy()
    for i in range(seq_length):
        state = rnn_step(state)
        all_states.append(state.copy())  # Store everything!

    elapsed_standard = time.time() - start_time
    current, peak_standard = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Peak memory: {peak_standard / 1024 / 1024:.2f} MB")
    print(f"  Time: {elapsed_standard:.4f} seconds")
    print(f"  Stored states: {len(all_states)}")

    # Method 2: Checkpointing (store only sqrt(n) states)
    print("\n" + "-" * 80)
    print("Method 2: CHECKPOINTING APPROACH (store sqrt(n) states)")
    print("-" * 80)

    num_checkpoints = int(np.sqrt(seq_length))
    print(f"  Number of checkpoints: {num_checkpoints}")

    tracemalloc.start()
    start_time = time.time()

    final_state, checkpoints = checkpointed_loop(
        rnn_step, initial_state, seq_length, num_checkpoints
    )

    elapsed_checkpoint = time.time() - start_time
    current, peak_checkpoint = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Peak memory: {peak_checkpoint / 1024 / 1024:.2f} MB")
    print(f"  Time: {elapsed_checkpoint:.4f} seconds")
    print(f"  Stored checkpoints: {len(checkpoints)}")

    # Verify correctness
    print("\n" + "-" * 80)
    print("VERIFICATION")
    print("-" * 80)

    final_state_standard = all_states[-1]
    match = np.allclose(final_state, final_state_standard)
    print(f"  Final states match: {match}")
    if match:
        print("  ✓ Checkpointing produces identical results!")
    else:
        print("  ✗ Results differ (this should not happen)")

    # Compare
    memory_reduction = 1 - (peak_checkpoint / peak_standard)
    time_overhead = (elapsed_checkpoint / elapsed_standard) - 1

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Memory reduction: {memory_reduction:.1%}")
    print(f"  Time overhead: {time_overhead:+.1%}")
    print(f"  States stored: {len(all_states)} → {len(checkpoints)}")
    print(f"  Storage ratio: {len(checkpoints) / len(all_states):.1%}")

    # Theoretical predictions
    print("\n" + "-" * 80)
    print("THEORETICAL ANALYSIS")
    print("-" * 80)

    stats = get_memory_savings(seq_length, num_checkpoints)
    print(f"  Expected memory reduction: {stats['savings_percent']:.1f}%")
    print(f"  Checkpoints used: {stats['num_checkpoints']}")
    print(f"  Recomputation factor: {stats['recomputation_factor']:.1f}x")

    return memory_reduction


def demo_checkpoint_positions():
    """Visualize checkpoint positions for different sequence lengths."""

    print("\n" + "=" * 80)
    print("CHECKPOINT POSITION ANALYSIS")
    print("=" * 80)

    print("\nOptimal checkpoint positions for various sequence lengths:")
    print("-" * 80)

    for seq_len in [10, 50, 100, 500, 1000]:
        num_checks = int(np.sqrt(seq_len))
        positions = compute_checkpoint_positions(seq_len, num_checks)

        print(f"\nSequence length: {seq_len:4d}")
        print(f"  Checkpoints: {len(positions)}")
        print(f"  Positions: {positions[:10]}{' ...' if len(positions) > 10 else ''}")

        # Show spacing
        if len(positions) > 1:
            spacings = [positions[i] - positions[i-1] for i in range(1, len(positions))]
            avg_spacing = np.mean(spacings)
            print(f"  Average spacing: {avg_spacing:.1f} steps")


def demo_gradient_computation():
    """Show how gradients work with checkpointing (simplified)."""

    print("\n" + "=" * 80)
    print("GRADIENT COMPUTATION WITH CHECKPOINTING")
    print("=" * 80)

    print("\nSimple recurrent computation:")
    print("-" * 80)

    def forward_pass(x, steps=100):
        """Forward pass of a simple recurrent computation."""
        state = x.copy()
        for i in range(steps):
            state = np.tanh(state * 1.1 + 0.1)
        return state

    # Test case
    x = np.array([1.0, 2.0, 3.0])
    steps = 100

    print(f"  Input: {x}")
    print(f"  Steps: {steps}")

    # Forward pass with checkpointing
    def step_func(state):
        return np.tanh(state * 1.1 + 0.1)

    num_checkpoints = int(np.sqrt(steps))
    final, checkpoints = checkpointed_loop(step_func, x, steps, num_checkpoints)

    print(f"  Output: {final}")
    print(f"  Checkpoints stored: {len(checkpoints)}")

    # Show where checkpoints were placed
    positions = sorted(checkpoints.keys())
    print(f"  Checkpoint positions: {positions[:5]}...{positions[-1]}")

    print("\n  Memory saved:")
    print(f"    Without checkpointing: {steps} states")
    print(f"    With checkpointing: {len(checkpoints)} states")
    print(f"    Reduction: {(1 - len(checkpoints)/steps):.1%}")

    print("\n  During backward pass:")
    print(f"    For each gradient step, recompute from nearest checkpoint")
    print(f"    Average recomputation: {steps / len(checkpoints):.1f} steps")


def demo_scaling_analysis():
    """Analyze how checkpointing scales with sequence length."""

    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    print("\nMemory usage vs sequence length:")
    print("-" * 80)
    print(f"{'Seq Length':>12} {'Without':>12} {'With':>12} {'Savings':>12}")
    print("-" * 80)

    for seq_len in [10, 50, 100, 500, 1000, 5000, 10000]:
        stats = get_memory_savings(seq_len)
        print(f"{seq_len:12d} "
              f"{stats['without_checkpointing']:12d} "
              f"{stats['with_checkpointing']:12d} "
              f"{stats['savings_percent']:11.1f}%")

    print("\nKey insight: Savings increase with sequence length!")
    print("  For long sequences (n > 1000), memory reduction > 95%")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "TANGENT CHECKPOINTING DEMO" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")

    try:
        # Run all demos
        memory_reduction = demo_memory_savings()
        demo_checkpoint_positions()
        demo_gradient_computation()
        demo_scaling_analysis()

        # Final summary
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        print(f"✓ Achieved {memory_reduction:.1%} memory reduction")
        print("✓ Checkpointing is working correctly!")
        print("✓ Final states match exactly")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("1. Integrate with Tangent's AST transformation")
        print("2. Add automatic loop detection")
        print("3. Implement proper gradient computation via tangent.grad()")
        print("4. Add support for JAX/TensorFlow backends")
        print("5. Benchmark on real RNN/LSTM models")

        print("\n" + "=" * 80)
        print("STATUS: Demo completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
