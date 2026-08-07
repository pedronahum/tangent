"""Break and continue must be rejected with a clear error.

Reverse-mode source-to-source AD records one tape entry per completed loop
iteration. `break` exits mid-iteration before the iteration counter is advanced,
so the backward pass replays the wrong number of iterations and silently
produces incorrect gradients; `continue` has the same defect unless it is the
very first statement in the loop body. Rather than emit plausible-but-wrong
gradients, Tangent rejects both up front with an actionable suggestion.

See the git history for the concrete miscomputations these guards prevent (e.g.
a `break` after three accumulating iterations returned the two-iteration
gradient).
"""
import pytest

import tangent
from tangent.errors import TangentParseError


class TestBreakRejected:
    def test_break_in_for_loop(self):
        def f(x):
            total = 0.0
            for i in range(10):
                total = total + x
                if i >= 2:
                    break
            return total

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert 'Break statements are not supported' in str(exc.value)

    def test_break_in_while_loop(self):
        def f(x):
            total = 0.0
            i = 0
            while i < 10:
                total = total + x
                i = i + 1
                if i >= 3:
                    break
            return total

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert 'Break statements are not supported' in str(exc.value)

    def test_break_error_has_suggestion(self):
        def f(x):
            for i in range(10):
                if i > 5:
                    break
            return x

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert '💡 Suggestion' in str(exc.value)


class TestContinueRejected:
    def test_continue_first_statement(self):
        # Even the "harmless" continue-first form is rejected for consistency.
        def f(x):
            total = 0.0
            for i in range(5):
                if i == 2:
                    continue
                total = total + x
            return total

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert 'Continue statements are not supported' in str(exc.value)

    def test_continue_with_active_computation_before(self):
        def f(x):
            total = 0.0
            for i in range(4):
                total = total + x
                if i == 2:
                    continue
                total = total + x * x
            return total

        with pytest.raises(TangentParseError) as exc:
            tangent.grad(f)
        assert 'Continue statements are not supported' in str(exc.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
