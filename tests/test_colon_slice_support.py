"""Test suite for colon slice support in array subscripting.

This tests the fix for the issue where x[0, :] and similar patterns
would fail with SyntaxError due to invalid variable name generation.
"""
import pytest
import tangent
import numpy as np


class TestColonSliceBasic:
    """Basic colon slice patterns that should work."""

    def test_row_selection_2d(self):
        """Test x[0, :] - select first row."""
        def f(x):
            row = x[0, :]
            return row.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        grad = df(x)

        # Gradient should be [1, 1, 1] in first row, [0, 0, 0] in second
        expected = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_column_selection_2d(self):
        """Test x[:, 0] - select first column."""
        def f(x):
            col = x[:, 0]
            return col.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        grad = df(x)

        # Gradient should be [1, 0, 0] in first row, [1, 0, 0] in second
        expected = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_full_array_colon(self):
        """Test x[:, :] - full array."""
        def f(x):
            full = x[:, :]
            return full.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = df(x)

        # Gradient should be all ones
        expected = np.ones_like(x)
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


class TestColonSliceMixed:
    """Mixed patterns with colons and ranges."""

    def test_range_plus_colon(self):
        """Test x[0:2, :] - range in first dim, colon in second."""
        def f(x):
            sliced = x[0:2, :]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        grad = df(x)

        # Gradient should be ones for first two rows, zeros for third
        expected = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_colon_plus_range(self):
        """Test x[:, 1:3] - colon in first dim, range in second."""
        def f(x):
            sliced = x[:, 1:3]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        grad = df(x)

        # Gradient should be zeros for col 0, ones for cols 1-2, zeros for col 3
        expected = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_int_plus_colon(self):
        """Test x[1, :] - integer in first dim, colon in second."""
        def f(x):
            row = x[1, :]
            return row.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        grad = df(x)

        # Gradient should be zeros for first row, ones for second, zeros for third
        expected = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


class TestColonSlice3D:
    """Test colon slices with 3D arrays."""

    def test_3d_first_dim_colon(self):
        """Test x[:, 0, 0] - colon in first dimension."""
        def f(x):
            sliced = x[:, 0, 0]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.ones((2, 2, 2))
        grad = df(x)

        # Gradient should be 1 at [:, 0, 0], zeros elsewhere
        expected = np.zeros((2, 2, 2))
        expected[:, 0, 0] = 1.0
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_3d_middle_dim_colon(self):
        """Test x[0, :, 0] - colon in middle dimension."""
        def f(x):
            sliced = x[0, :, 0]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.ones((2, 2, 2))
        grad = df(x)

        # Gradient should be 1 at [0, :, 0], zeros elsewhere
        expected = np.zeros((2, 2, 2))
        expected[0, :, 0] = 1.0
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_3d_last_dim_colon(self):
        """Test x[0, 0, :] - colon in last dimension."""
        def f(x):
            sliced = x[0, 0, :]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.ones((2, 2, 2))
        grad = df(x)

        # Gradient should be 1 at [0, 0, :], zeros elsewhere
        expected = np.zeros((2, 2, 2))
        expected[0, 0, :] = 1.0
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


class TestColonSliceWithStep:
    """Test colon slices with step notation."""

    def test_step_colon(self):
        """Test x[::2, :] - every other row."""
        def f(x):
            sliced = x[::2, :]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        grad = df(x)

        # Gradient should be ones for rows 0 and 2, zeros for rows 1 and 3
        expected = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_colon_with_step(self):
        """Test x[:, 0:4:2] - colon in first dim, range with step in second."""
        def f(x):
            sliced = x[:, 0:4:2]
            return sliced.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        grad = df(x)

        # Gradient should be ones at columns 0 and 2, zeros at 1 and 3
        expected = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


class TestRealWorldExamples:
    """Real-world use cases from the issue."""

    def test_np_dot_with_row_slices(self):
        """Test the original failing example: np.dot(x[0, :], y[0, :])"""
        def f(x, y):
            x_row = x[0, :]
            y_row = y[0, :]
            return np.dot(x_row, y_row)

        df = tangent.grad(f, wrt=(0,))
        x = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[4.0, 5.0, 6.0]])
        grad = df(x, y)

        # Gradient w.r.t. x should be y[0, :]
        expected = np.array([[4.0, 5.0, 6.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiply pattern."""
        def f(A, x_vec):
            # Extract row and multiply with vector
            row = A[0, :]
            return np.dot(row, x_vec)

        df = tangent.grad(f, wrt=(0,))
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_vec = np.array([5.0, 6.0])
        grad = df(A, x_vec)

        # Gradient w.r.t. A should have x_vec in first row, zeros in second
        expected = np.array([[5.0, 6.0], [0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_element_slice(self):
        """Test x[0, :] when second dimension has size 1."""
        def f(x):
            row = x[0, :]
            return row.sum()

        df = tangent.grad(f)
        x = np.array([[1.0], [2.0]])
        grad = df(x)

        expected = np.array([[1.0], [0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"

    def test_nested_slicing(self):
        """Test nested slicing operations."""
        def f(x):
            # First slice rows, then slice columns
            rows = x[0:2, :]
            result = rows[:, 0]
            return result.sum()

        df = tangent.grad(f)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        grad = df(x)

        # Gradient should be 1 at [0,0] and [1,0], zeros elsewhere
        expected = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"


if __name__ == '__main__':
    # Run tests manually for debugging
    import sys

    test_classes = [
        TestColonSliceBasic,
        TestColonSliceMixed,
        TestColonSlice3D,
        TestColonSliceWithStep,
        TestRealWorldExamples,
        TestEdgeCases,
    ]

    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'=' * 70}")
        print(f"Running {test_class.__name__}")
        print('=' * 70)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                print(f"\n{method_name}...", end=' ')
                try:
                    method = getattr(instance, method_name)
                    method()
                    print("✓ PASS")
                    passed += 1
                except Exception as e:
                    print(f"✗ FAIL")
                    print(f"  Error: {type(e).__name__}: {str(e)[:200]}")
                    failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print('=' * 70)

    sys.exit(0 if failed == 0 else 1)
