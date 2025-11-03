# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for inheritance support in Tangent automatic differentiation.

This test suite validates that Tangent can differentiate through
class hierarchies using Python's inheritance mechanism.
"""
from __future__ import absolute_import

import pytest
import numpy as np
import tangent


# =============================================================================
# Test Classes - Simple Inheritance
# =============================================================================

class SimpleBase:
    """Base class with a simple method."""

    def square(self, x):
        """Return x squared."""
        return x ** 2


class SimpleDerived(SimpleBase):
    """Derived class that inherits square method."""
    pass


# =============================================================================
# Test Classes - Method Overriding
# =============================================================================

class OverrideBase:
    """Base class with method to be overridden."""

    def compute(self, x):
        """Return x squared."""
        return x ** 2


class OverrideDerived(OverrideBase):
    """Derived class that overrides compute."""

    def compute(self, x):
        """Return x cubed (overrides base)."""
        return x ** 3


# =============================================================================
# Test Classes - Attribute Inheritance
# =============================================================================

class AttributeBase:
    """Base class with attributes."""

    def __init__(self, factor):
        self.factor = factor


class AttributeDerived(AttributeBase):
    """Derived class that uses parent attribute."""

    def __init__(self, factor):
        super().__init__(factor)

    def scale(self, x):
        """Scale x by inherited factor."""
        return x * self.factor


# =============================================================================
# Test Classes - Multi-level Inheritance
# =============================================================================

class LevelA:
    """Top-level base class."""

    def base_method(self, x):
        return x ** 2


class LevelB(LevelA):
    """Middle-level class."""

    def middle_method(self, x):
        return self.base_method(x) + x


class LevelC(LevelB):
    """Bottom-level class."""

    def top_method(self, x):
        return self.middle_method(x) * 2


# =============================================================================
# Test Classes - Combined Attributes
# =============================================================================

class VehicleBase:
    """Base vehicle class."""

    def __init__(self, speed_factor):
        self.speed_factor = speed_factor


class CarDerived(VehicleBase):
    """Derived car class with additional attribute."""

    def __init__(self, speed_factor, efficiency):
        super().__init__(speed_factor)
        self.efficiency = efficiency

    def cost(self, distance):
        """Compute cost using both parent and child attributes."""
        return distance * self.speed_factor / self.efficiency


# =============================================================================
# Test Classes - Method Chaining Across Hierarchy
# =============================================================================

class ChainBase:
    """Base class with helper method."""

    def helper(self, x):
        return x ** 2


class ChainDerived(ChainBase):
    """Derived class that calls inherited method."""

    def compute(self, x):
        # Calls inherited helper method
        return self.helper(x) + x


# =============================================================================
# Phase 1: Simple Inheritance Tests
# =============================================================================

def test_simple_inheritance():
    """Test method inherited from parent class."""
    def f(x):
        obj = SimpleDerived()
        return obj.square(x)

    # Gradient of x^2 is 2x
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 6
    assert abs(df(3.0) - 6.0) < 1e-10, f"Expected 6.0, got {df(3.0)}"

    # Test at x = 5: gradient should be 10
    assert abs(df(5.0) - 10.0) < 1e-10, f"Expected 10.0, got {df(5.0)}"


def test_inherited_method_with_multiple_uses():
    """Test using inherited method multiple times."""
    def f(x):
        obj = SimpleDerived()
        return obj.square(x) + obj.square(x * 2)

    # Gradient of x^2 + (2x)^2 = x^2 + 4x^2 = 5x^2 is 10x
    df = tangent.grad(f)

    # Test at x = 2: gradient should be 20
    assert abs(df(2.0) - 20.0) < 1e-10, f"Expected 20.0, got {df(2.0)}"


# =============================================================================
# Phase 2: Method Overriding Tests
# =============================================================================

def test_method_override():
    """Test that derived class method overrides base class method."""
    def f(x):
        obj = OverrideDerived()
        return obj.compute(x)

    # Should use Derived version: x^3, gradient is 3x^2
    df = tangent.grad(f)

    # Test at x = 2: gradient should be 12 (3 * 4)
    assert abs(df(2.0) - 12.0) < 1e-10, f"Expected 12.0, got {df(2.0)}"

    # Test at x = 3: gradient should be 27 (3 * 9)
    assert abs(df(3.0) - 27.0) < 1e-10, f"Expected 27.0, got {df(3.0)}"


def test_base_class_method_still_works():
    """Test that base class method works when using base class."""
    def f(x):
        obj = OverrideBase()
        return obj.compute(x)

    # Should use Base version: x^2, gradient is 2x
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 6
    assert abs(df(3.0) - 6.0) < 1e-10, f"Expected 6.0, got {df(3.0)}"


# =============================================================================
# Phase 3: Attribute Inheritance Tests
# =============================================================================

def test_attribute_inheritance_with_super():
    """Test attribute inherited via super().__init__()."""
    def f(x):
        obj = AttributeDerived(2.5)
        return obj.scale(x)

    # Gradient of 2.5*x is 2.5
    df = tangent.grad(f)

    assert abs(df(3.0) - 2.5) < 1e-10, f"Expected 2.5, got {df(3.0)}"
    assert abs(df(10.0) - 2.5) < 1e-10, f"Expected 2.5, got {df(10.0)}"


def test_combined_parent_child_attributes():
    """Test using both parent and child attributes."""
    def f(distance):
        car = CarDerived(speed_factor=1.5, efficiency=0.5)
        return car.cost(distance)

    # cost = distance * 1.5 / 0.5 = distance * 3.0
    # Gradient is 3.0
    df = tangent.grad(f)

    assert abs(df(10.0) - 3.0) < 1e-10, f"Expected 3.0, got {df(10.0)}"


# =============================================================================
# Phase 4: Multi-level Inheritance Tests
# =============================================================================

def test_multi_level_inheritance():
    """Test inheritance through multiple levels (A -> B -> C)."""
    def f(x):
        obj = LevelC()
        return obj.top_method(x)

    # top_method calls middle_method which calls base_method
    # Result: 2 * (x^2 + x) = 2x^2 + 2x
    # Gradient: 4x + 2
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 14 (4*3 + 2)
    assert abs(df(3.0) - 14.0) < 1e-10, f"Expected 14.0, got {df(3.0)}"

    # Test at x = 5: gradient should be 22 (4*5 + 2)
    assert abs(df(5.0) - 22.0) < 1e-10, f"Expected 22.0, got {df(5.0)}"


def test_calling_grandparent_method():
    """Test calling method from grandparent class."""
    def f(x):
        obj = LevelC()
        # Even though LevelC is two levels down, it can call base_method
        return obj.base_method(x)

    # Gradient of x^2 is 2x
    df = tangent.grad(f)

    assert abs(df(3.0) - 6.0) < 1e-10, f"Expected 6.0, got {df(3.0)}"


# =============================================================================
# Phase 5: Method Chaining with Inheritance Tests
# =============================================================================

def test_method_chaining_across_hierarchy():
    """Test derived method calling inherited method."""
    def f(x):
        obj = ChainDerived()
        return obj.compute(x)

    # compute calls inherited helper: x^2 + x
    # Gradient: 2x + 1
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 7 (2*3 + 1)
    assert abs(df(3.0) - 7.0) < 1e-10, f"Expected 7.0, got {df(3.0)}"


# =============================================================================
# Phase 6: NumPy Integration with Inheritance
# =============================================================================

class NumpyBase:
    """Base class with NumPy operations."""

    def sin_op(self, x):
        return np.sin(x)


class NumpyDerived(NumpyBase):
    """Derived class using inherited NumPy method."""

    def combined(self, x):
        return self.sin_op(x) + x ** 2


def test_numpy_with_inheritance():
    """Test NumPy operations in inherited methods."""
    def f(x):
        obj = NumpyDerived()
        return obj.combined(x)

    # Gradient of sin(x) + x^2 is cos(x) + 2x
    df = tangent.grad(f)

    x_test = 1.0
    expected = np.cos(x_test) + 2 * x_test
    result = df(x_test)

    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"


# =============================================================================
# Phase 7: Edge Cases
# =============================================================================

class EmptyDerived(SimpleBase):
    """Derived class with no additional methods or attributes."""
    def __init__(self):
        pass


def test_empty_derived_class():
    """Test derived class with empty __init__."""
    def f(x):
        obj = EmptyDerived()
        return obj.square(x)

    df = tangent.grad(f)
    assert abs(df(3.0) - 6.0) < 1e-10


class MultipleMethodBase:
    """Base with multiple methods."""

    def method_a(self, x):
        return x ** 2

    def method_b(self, x):
        return x ** 3


class MultipleMethodDerived(MultipleMethodBase):
    """Derived using multiple inherited methods."""

    def combined(self, x):
        return self.method_a(x) + self.method_b(x)


def test_multiple_inherited_methods():
    """Test using multiple methods from parent."""
    def f(x):
        obj = MultipleMethodDerived()
        return obj.combined(x)

    # Gradient of x^2 + x^3 is 2x + 3x^2
    df = tangent.grad(f)

    # At x = 2: 2*2 + 3*4 = 16
    assert abs(df(2.0) - 16.0) < 1e-10


# =============================================================================
# Test Suite Summary
# =============================================================================

if __name__ == '__main__':
    """
    Test Suite for Inheritance Support in Tangent

    Phase 1: Simple Inheritance (2 tests)
    - test_simple_inheritance
    - test_inherited_method_with_multiple_uses

    Phase 2: Method Overriding (2 tests)
    - test_method_override
    - test_base_class_method_still_works

    Phase 3: Attribute Inheritance (2 tests)
    - test_attribute_inheritance_with_super
    - test_combined_parent_child_attributes

    Phase 4: Multi-level Inheritance (2 tests)
    - test_multi_level_inheritance
    - test_calling_grandparent_method

    Phase 5: Method Chaining (1 test)
    - test_method_chaining_across_hierarchy

    Phase 6: NumPy Integration (1 test)
    - test_numpy_with_inheritance

    Phase 7: Edge Cases (2 tests)
    - test_empty_derived_class
    - test_multiple_inherited_methods

    Total: 12 tests

    Run with: pytest tests/test_inheritance.py -v
    """
    pytest.main([__file__, '-v'])
