#!/usr/bin/env python3
"""
Basic tests for checkpointing functionality.

Tests ensure that checkpointing:
1. Computes correct checkpoint positions
2. Produces identical results to non-checkpointed execution
3. Actually reduces memory usage
4. Handles edge cases correctly
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tangent.checkpointing_simple import (
    compute_checkpoint_positions,
    checkpointed_loop,
    get_memory_savings,
    _copy_state
)


class TestCheckpointPositions:
    """Test checkpoint position computation."""

    def test_basic_positions(self):
        """Test that checkpoint positions are computed correctly."""
        positions = compute_checkpoint_positions(10, 3)
        assert len(positions) <= 3
        assert all(0 <= p < 10 for p in positions)
        assert positions == sorted(positions)  # Should be in order

    def test_large_sequence(self):
        """Test checkpoint positions for large sequence."""
        positions = compute_checkpoint_positions(1000, 10)
        assert len(positions) <= 10
        assert all(0 <= p < 1000 for p in positions)
        assert positions == sorted(positions)

    def test_more_checkpoints_than_steps(self):
        """Test edge case: more checkpoints than steps."""
        positions = compute_checkpoint_positions(5, 10)
        assert len(positions) <= 5
        assert positions == list(range(5))

    def test_no_checkpoints(self):
        """Test edge case: no checkpoints requested."""
        positions = compute_checkpoint_positions(100, 0)
        assert positions == []

    def test_one_checkpoint(self):
        """Test with single checkpoint."""
        positions = compute_checkpoint_positions(100, 1)
        assert len(positions) == 1
        assert 0 < positions[0] < 100

    def test_positions_spread_evenly(self):
        """Test that checkpoints are distributed across the sequence."""
        positions = compute_checkpoint_positions(100, 5)
        assert len(positions) == 5

        # Check that positions are reasonably spread out
        # Should not all be at the beginning or end
        assert positions[0] < 50
        assert positions[-1] > 50


class TestCheckpointedLoop:
    """Test checkpointed loop execution."""

    def test_simple_loop(self):
        """Test basic checkpointed loop execution."""
        def step_func(x):
            return x * 1.1 + 0.1

        x0 = np.array([1.0, 2.0, 3.0])

        # Standard loop
        state = x0.copy()
        for _ in range(100):
            state = step_func(state)
        expected = state

        # Checkpointed loop
        final, checkpoints = checkpointed_loop(step_func, x0, 100, 5)

        np.testing.assert_allclose(final, expected)
        assert len(checkpoints) <= 5

    def test_nonlinear_function(self):
        """Test with nonlinear function (tanh)."""
        def step_func(x):
            return np.tanh(x * 0.9 + 0.1)

        x0 = np.array([0.5, 1.0, 1.5])

        # Standard
        state = x0.copy()
        for _ in range(50):
            state = step_func(state)
        expected = state

        # Checkpointed
        final, checkpoints = checkpointed_loop(step_func, x0, 50, 7)

        np.testing.assert_allclose(final, expected)
        assert len(checkpoints) == 7

    def test_matrix_computation(self):
        """Test with matrix operations."""
        W = np.random.randn(10, 10) * 0.1
        b = np.random.randn(10) * 0.1

        def rnn_step(state):
            return np.tanh(state @ W + b)

        x0 = np.zeros(10)

        # Standard
        state = x0.copy()
        for _ in range(100):
            state = rnn_step(state)
        expected = state

        # Checkpointed
        final, checkpoints = checkpointed_loop(rnn_step, x0, 100, 10)

        np.testing.assert_allclose(final, expected, rtol=1e-10)

    def test_default_num_checkpoints(self):
        """Test default checkpoint count (sqrt(n))."""
        def step_func(x):
            return x * 1.1

        x0 = np.array([1.0])

        final, checkpoints = checkpointed_loop(step_func, x0, 100)

        # Default should be sqrt(100) = 10
        assert len(checkpoints) == 10

    def test_checkpoint_positions_correct(self):
        """Test that checkpoints are saved at computed positions."""
        def step_func(x):
            return x + 1.0

        x0 = np.array([0.0])

        final, checkpoints = checkpointed_loop(step_func, x0, 20, 4)

        # Verify checkpoints are at expected positions
        positions = sorted(checkpoints.keys())
        assert len(positions) == 4

        # Verify checkpoint values are correct
        for pos in positions:
            # At position i, value should be i (since we add 1 each step)
            expected_value = float(pos)
            np.testing.assert_allclose(checkpoints[pos], [expected_value])


class TestMemoryUsage:
    """Test memory reduction."""

    def test_memory_savings_calculation(self):
        """Test memory savings calculation."""
        stats = get_memory_savings(1000, 10)

        assert stats['without_checkpointing'] == 1000
        assert stats['with_checkpointing'] == 10
        assert stats['savings_percent'] == 99.0
        assert stats['num_checkpoints'] == 10

    def test_default_sqrt_n_checkpoints(self):
        """Test default sqrt(n) checkpoint strategy."""
        stats = get_memory_savings(100)

        # Should use sqrt(100) = 10 checkpoints
        assert stats['num_checkpoints'] == 10
        assert stats['with_checkpointing'] == 10
        assert stats['savings_percent'] == 90.0

    def test_scaling_with_sequence_length(self):
        """Test that savings increase with sequence length."""
        savings_10 = get_memory_savings(10)['savings_percent']
        savings_100 = get_memory_savings(100)['savings_percent']
        savings_1000 = get_memory_savings(1000)['savings_percent']

        # Longer sequences should have better savings
        assert savings_100 > savings_10
        assert savings_1000 > savings_100

    def test_actual_memory_reduction(self):
        """Test that checkpointing actually uses less memory."""
        import tracemalloc

        def heavy_computation(x):
            # Simulate memory-intensive operation
            temp = np.outer(x, x)
            return np.sum(temp, axis=0)

        x0 = np.random.randn(100)

        # Measure standard approach
        tracemalloc.start()
        states = []
        state = x0.copy()
        for _ in range(100):
            state = heavy_computation(state)
            states.append(state.copy())
        _, peak_standard = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure checkpointed approach
        tracemalloc.start()
        final, checkpoints = checkpointed_loop(heavy_computation, x0, 100, 10)
        _, peak_checkpoint = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Checkpointed should use less memory
        assert peak_checkpoint < peak_standard

        # Results should match
        np.testing.assert_allclose(final, states[-1])


class TestStateCopying:
    """Test state copying utility."""

    def test_copy_numpy_array(self):
        """Test copying NumPy arrays."""
        original = np.array([1.0, 2.0, 3.0])
        copied = _copy_state(original)

        assert isinstance(copied, np.ndarray)
        np.testing.assert_array_equal(original, copied)

        # Should be a copy, not a reference
        copied[0] = 999.0
        assert original[0] == 1.0

    def test_copy_tuple(self):
        """Test copying tuple of arrays."""
        original = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        copied = _copy_state(original)

        assert isinstance(copied, tuple)
        assert len(copied) == 2
        np.testing.assert_array_equal(original[0], copied[0])

        # Should be deep copy
        copied[0][0] = 999.0
        assert original[0][0] == 1.0

    def test_copy_dict(self):
        """Test copying dictionary of arrays."""
        original = {'a': np.array([1.0]), 'b': np.array([2.0])}
        copied = _copy_state(original)

        assert isinstance(copied, dict)
        assert set(copied.keys()) == {'a', 'b'}
        np.testing.assert_array_equal(original['a'], copied['a'])

        # Should be deep copy
        copied['a'][0] = 999.0
        assert original['a'][0] == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_length_sequence(self):
        """Test with zero-length sequence."""
        def step_func(x):
            return x + 1.0

        x0 = np.array([0.0])
        final, checkpoints = checkpointed_loop(step_func, x0, 0, 5)

        # Should return initial state unchanged
        np.testing.assert_array_equal(final, x0)
        assert len(checkpoints) == 0

    def test_single_step(self):
        """Test with single step."""
        def step_func(x):
            return x * 2.0

        x0 = np.array([3.0])
        final, checkpoints = checkpointed_loop(step_func, x0, 1, 5)

        np.testing.assert_allclose(final, [6.0])

    def test_very_long_sequence(self):
        """Test with very long sequence."""
        def step_func(x):
            return x * 0.999  # Decay to prevent overflow

        x0 = np.array([1.0])
        final, checkpoints = checkpointed_loop(step_func, x0, 10000, 100)

        # Should complete without error
        assert len(checkpoints) == 100
        assert not np.isnan(final).any()
        assert not np.isinf(final).any()


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("CHECKPOINTING BASIC TESTS")
    print("=" * 80)

    test_classes = [
        TestCheckpointPositions,
        TestCheckpointedLoop,
        TestMemoryUsage,
        TestStateCopying,
        TestEdgeCases
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: ERROR - {e}")
                total_failed += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Total:  {total_passed + total_failed}")

    if total_failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
