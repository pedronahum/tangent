"""Test enhanced error messages with helpful suggestions."""
import pytest
import tangent
from tangent.errors import TangentParseError
from tangent.dict_construction_error import DictConstructionError


def test_dict_comprehension_error_has_suggestion():
    """Test that dict comprehension error includes helpful suggestion."""

    def f(x):
        d = {k: x ** i for i, k in enumerate(['a', 'b'])}
        return d['a']

    with pytest.raises(TangentParseError) as exc_info:
        df = tangent.grad(f)

    error_msg = str(exc_info.value)
    assert "Dictionary Comprehensions are not supported" in error_msg
    assert "💡 Suggestion" in error_msg
    assert "Pass dict as parameter" in error_msg
    assert "Use separate variables" in error_msg


def test_fstring_now_supported():
    """f-strings are now supported (non-differentiable debug/assert messages)."""

    def f(x):
        msg = f"Value is {x}"
        return x ** 2

    df = tangent.grad(f)
    assert df(3.0) == pytest.approx(6.0)  # d(x^2)/dx = 2x, msg ignored


def test_in_operator_now_supported():
    """The 'in' operator is now supported as a (non-differentiable) branch guard."""

    def f(x):
        if x in (1.0, 2.0, 3.0):
            y = x
        else:
            y = x ** 2
        return y

    df = tangent.grad(f)
    assert df(2.0) == pytest.approx(1.0)   # in-branch: d(x)/dx = 1
    assert df(5.0) == pytest.approx(10.0)  # else-branch: d(x^2)/dx = 2x


def test_multi_key_dict_construction_now_works():
    """Multi-key dict construction with string keys now differentiates correctly.

    This previously failed because a dict named ``d`` collided with the internal
    ``d[x]`` gradient-operator sentinel, producing undefined ``_`` placeholders.
    """

    def f(x):
        d = {'a': x, 'b': x ** 2}
        return d['a'] + d['b']

    df = tangent.grad(f)
    # d/dx (x + x^2) = 1 + 2x = 5 at x = 2
    assert df(2.0) == pytest.approx(5.0)


def test_set_literal_now_supported():
    """Set literals are now supported (non-differentiable membership guards)."""

    def f(x):
        if x in {1.0, 2.0, 3.0}:
            y = x * x
        else:
            y = x * 3.0
        return y

    df = tangent.grad(f)
    assert df(2.0) == pytest.approx(4.0)   # in set -> d(x^2) = 2x
    assert df(5.0) == pytest.approx(3.0)   # else -> d(3x) = 3


if __name__ == '__main__':
    print("=" * 80)
    print("TESTING ENHANCED ERROR MESSAGES")
    print("=" * 80)

    tests = [
        ("Dict comprehension error", test_dict_comprehension_error_has_suggestion),
        ("F-string now supported", test_fstring_now_supported),
        ("In operator now supported", test_in_operator_now_supported),
        ("Multi-key dict construction now works", test_multi_key_dict_construction_now_works),
        ("Set literal now supported", test_set_literal_now_supported),
    ]

    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 80)
        try:
            test_func()
            print("✓ PASS")
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
        except Exception as e:
            print(f"✗ ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
