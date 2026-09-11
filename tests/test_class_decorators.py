"""@property and @classmethod are rejected cleanly (not with an opaque error).

Method *calls* on instances are inlined and differentiate; a @property or
@classmethod is read as a plain attribute, which is not differentiable. Rather
than surface deep in the reverse transform as "attributes are not yet
supported", class_desugar rejects them at the frontend with a TangentParseError
and a workaround. @staticmethod (a plain function) is unaffected.
"""

import pytest

import tangent
from tangent.errors import TangentParseError


class Widget:
    def __init__(self, w):
        self.w = w

    @property
    def scaled(self):
        return self.w * 2.0

    @classmethod
    def make(cls, x):
        return x * 3.0

    @staticmethod
    def squared(x):
        return x * x


def prop_inline(x):
    return Widget(x).scaled


def prop_via_var(x):
    obj = Widget(x)
    return obj.scaled


def classmethod_access(x):
    return Widget.make(x)


def staticmethod_access(x):
    return Widget.squared(x)


class TestRejectedCleanly:
    @pytest.mark.parametrize('fn', [prop_inline, prop_via_var], ids=['inline', 'via_var'])
    def test_property_rejected(self, fn):
        with pytest.raises(TangentParseError, match='property'):
            tangent.grad(fn)

    def test_property_error_has_workaround(self):
        try:
            tangent.grad(prop_inline)
            assert False
        except TangentParseError as e:
            assert 'Suggestion' in str(e)
            assert 'method' in str(e)

    def test_classmethod_rejected(self):
        with pytest.raises(TangentParseError, match='classmethod'):
            tangent.grad(classmethod_access)


class TestStillWorks:
    def test_staticmethod_differentiates(self):
        # A @staticmethod is a plain function; it must still work.
        assert tangent.grad(staticmethod_access)(3.0) == pytest.approx(6.0)

    def test_plain_class_method_call_unaffected(self):
        # The clean-rejection check must not disturb ordinary differentiable
        # code that touches no property/classmethod.
        import numpy as np

        def f(x):
            return np.sum(x * x)

        np.testing.assert_allclose(tangent.grad(f)(np.array([1.0, 2.0])), np.array([2.0, 4.0]))
