"""Augmented assignment to a subscript is supported; to an attribute it is not.

`a[i] += x` is expanded during normalization to `a[i] = a[i] + x`, keeping the
subscript as both the read and the write so the element is written back and the
scatter adjoint reduces the gathered gradient to the RHS shape. This works in
both reverse and forward mode, including in-place self-mutation of the
differentiated array.

Augmented assignment to an *attribute* (`obj.attr += x`) is still rejected with
a clear error, because attribute gradients are not supported.

Plain augmented assignment on a variable (`x += y`) is fully supported and must
keep working.
"""
import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestAugmentedSubscriptSupported:
    def test_add_to_element_reverse(self):
        def f(a):
            a[0] += a[1]
            return a[0]

        # f(a) = a0 + a1 -> gradient [1, 1]
        grad = tangent.grad(f)(np.array([1.0, 2.0]))
        assert np.allclose(grad, [1.0, 1.0])

    def test_add_to_element_forward(self):
        def f(a):
            a[0] += a[1]
            return a[0]

        df = tangent.autodiff(f, mode='forward', preserve_result=False)
        # With seed ones, the directional derivative is 1 + 1 = 2.
        assert df(np.array([1.0, 2.0]), np.ones(2)) == pytest.approx(2.0)

    def test_multiply_element_reverse(self):
        def f(a):
            a[1] *= a[0]
            return a[1]

        # f(a) = a1 * a0 -> gradient [a1, a0] = [2, 1]
        grad = tangent.grad(f)(np.array([1.0, 2.0]))
        assert np.allclose(grad, [2.0, 1.0])

    def test_scalar_rhs_into_element(self):
        def f(x):
            a = np.zeros(4)
            a[1] += x
            return np.sum(a)

        assert tangent.grad(f)(2.0) == pytest.approx(1.0)

    def test_scalar_rhs_into_slice(self):
        def f(x):
            a = np.zeros(4)
            a[0:2] += x
            return np.sum(a)

        # x is broadcast into two elements -> gradient 2
        assert tangent.grad(f)(2.0) == pytest.approx(2.0)

    def test_augmented_in_loop(self):
        def f(x):
            a = np.zeros(3)
            for i in range(3):
                a[i] += x
            return np.sum(a)

        assert tangent.grad(f)(2.0) == pytest.approx(3.0)

    def test_chained_augmentations(self):
        def f(x):
            a = np.zeros(3)
            a[0] += x
            a[0] *= 2.0
            return np.sum(a)

        # a0 = 2x -> gradient 2
        assert tangent.grad(f)(2.0) == pytest.approx(2.0)


class TestAugmentedAttributeStillRejected:
    def test_attribute_augmented_assignment(self):
        def f(x):
            a = np.zeros(3)
            a.real += x
            return np.sum(a)

        with pytest.raises(TangentParseError):
            tangent.grad(f)


class TestPlainAugmentedAssignmentStillWorks:
    def test_name_augmented_assignment(self):
        def f(x):
            total = 0.0
            total += x * x
            total += x
            return total

        # d/dx (x^2 + x) = 2x + 1 = 5 at x = 2
        assert tangent.grad(f)(2.0) == pytest.approx(5.0)

    def test_augmented_from_subscript_rhs(self):
        # Subscript on the RHS with a name target is fine.
        def f(a):
            total = 0.0
            total += a[0]
            total += a[1]
            return total

        grad = tangent.grad(f)(np.array([1.0, 2.0]))
        assert np.allclose(grad, [1.0, 1.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
