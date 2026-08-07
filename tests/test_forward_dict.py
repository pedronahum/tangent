"""Forward-mode support for local dictionaries.

Reverse mode has long supported constructing a dict and reading it back; forward
mode did not, because it never generated a tangent for the dict construction, so
subscript tangents referenced an undefined derivative (NameError). Forward mode
now builds a tangent dict with the same keys and the tangents of the values:

    d  = {'a': v}   ->   dd = {'a': dv}
    y  = d['a']     ->   dy = dd['a']

These tests exercise local dicts in forward mode. Non-active dict *parameters*
in forward mode remain a separate, pre-existing limitation and are not covered
here.
"""
import pytest

import tangent


def _fwd(f):
    return tangent.autodiff(f, mode='forward', preserve_result=False)


class TestForwardLocalDict:
    def test_single_key(self):
        def f(x):
            d = {'a': x * x}
            return d['a']

        assert _fwd(f)(3.0, 1.0) == pytest.approx(6.0)

    def test_multi_key(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
        assert _fwd(f)(2.0, 1.0) == pytest.approx(5.0)

    def test_get_method(self):
        def f(x):
            d = {'a': x * x}
            return d.get('a')

        assert _fwd(f)(3.0, 1.0) == pytest.approx(6.0)

    def test_nested_dict(self):
        def f(x):
            d = {'outer': {'inner': x * x}}
            return d['outer']['inner']

        assert _fwd(f)(3.0, 1.0) == pytest.approx(6.0)

    def test_dict_in_loop(self):
        def f(x):
            d = {'w': x ** 2}
            total = 0.0
            for i in range(3):
                total = total + d['w']
            return total

        # d/dx (3 * x^2) = 6x = 12 at x = 2
        assert _fwd(f)(2.0, 1.0) == pytest.approx(12.0)

    def test_matches_reverse_mode(self):
        def f(x):
            d = {'a': x, 'b': x ** 2}
            return d['a'] + d['b']

        fwd = _fwd(f)(4.0, 1.0)
        rev = tangent.grad(f)(4.0)
        assert fwd == pytest.approx(rev)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
