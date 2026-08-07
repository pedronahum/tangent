"""Augmented assignment to a subscript/attribute must fail loudly.

`a[i] += x` (and `obj.attr += x`) cannot be differentiated: the in-place update
is lost during normalization - the target is turned into a fresh temporary, so
the element is never written back. This corrupted both the primal value (e.g.
`a[0] += a[1]; return a[0]` returned the original `a[0]`) and the gradient,
silently. Such assignments are now rejected with a clear, actionable error.

Plain augmented assignment on a variable (`x += y`) is fully supported and must
keep working.
"""
import numpy as np
import pytest

import tangent
from tangent.errors import TangentParseError


class TestAugmentedSubscriptRejected:
    def test_add_to_element_reverse(self):
        def f(a):
            a[0] += a[1]
            return a[0]

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_add_to_element_forward(self):
        def f(a):
            a[0] += a[1]
            return a[0]

        with pytest.raises(TangentParseError):
            tangent.autodiff(f, mode='forward', preserve_result=False)

    def test_multiply_element(self):
        def f(a):
            a[1] *= a[0]
            return a[1]

        with pytest.raises(TangentParseError):
            tangent.grad(f)

    def test_error_has_suggestion(self):
        def f(a):
            a[0] += a[1]
            return a[0]

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert 'Workarounds' in str(exc.value)


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
