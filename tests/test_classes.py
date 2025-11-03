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
"""Tests for class support in Tangent automatic differentiation.

This test suite validates that Tangent can differentiate through
user-defined class methods using method inlining transformation.
"""
from __future__ import absolute_import

import pytest
import numpy as np
import tangent


# =============================================================================
# Phase 1: Basic Method Inlining Tests
# =============================================================================

class SimpleCalculator:
    """Simple class with basic mathematical methods."""

    def square(self, x):
        """Return x squared."""
        return x ** 2

    def cube(self, x):
        """Return x cubed."""
        return x ** 3

    def add_constant(self, x):
        """Return x plus a constant."""
        return x + 10


def test_simple_method_square():
    """Test differentiation of simple method: x^2."""
    def f(x):
        calc = SimpleCalculator()
        return calc.square(x)

    # Gradient of x^2 is 2x
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 6
    assert abs(df(3.0) - 6.0) < 1e-10, f"Expected 6.0, got {df(3.0)}"

    # Test at x = 5: gradient should be 10
    assert abs(df(5.0) - 10.0) < 1e-10, f"Expected 10.0, got {df(5.0)}"


def test_simple_method_cube():
    """Test differentiation of simple method: x^3."""
    def f(x):
        calc = SimpleCalculator()
        return calc.cube(x)

    # Gradient of x^3 is 3x^2
    df = tangent.grad(f)

    # Test at x = 2: gradient should be 12
    assert abs(df(2.0) - 12.0) < 1e-10, f"Expected 12.0, got {df(2.0)}"

    # Test at x = 3: gradient should be 27
    assert abs(df(3.0) - 27.0) < 1e-10, f"Expected 27.0, got {df(3.0)}"


def test_simple_method_constant():
    """Test differentiation of method with constant: x + 10."""
    def f(x):
        calc = SimpleCalculator()
        return calc.add_constant(x)

    # Gradient of x + 10 is 1
    df = tangent.grad(f)

    assert abs(df(3.0) - 1.0) < 1e-10, f"Expected 1.0, got {df(3.0)}"
    assert abs(df(100.0) - 1.0) < 1e-10, f"Expected 1.0, got {df(100.0)}"


def test_multiple_methods_same_class():
    """Test using multiple methods from the same class instance."""
    def f(x):
        calc = SimpleCalculator()
        return calc.square(x) + calc.cube(x)

    # Gradient of x^2 + x^3 is 2x + 3x^2
    df = tangent.grad(f)

    # Test at x = 2: gradient should be 2*2 + 3*4 = 4 + 12 = 16
    expected = 2*2 + 3*4
    assert abs(df(2.0) - expected) < 1e-10, f"Expected {expected}, got {df(2.0)}"

    # Test at x = 3: gradient should be 2*3 + 3*9 = 6 + 27 = 33
    expected = 2*3 + 3*9
    assert abs(df(3.0) - expected) < 1e-10, f"Expected {expected}, got {df(3.0)}"


# =============================================================================
# Phase 2: Instance Attributes Tests
# =============================================================================

class Scaler:
    """Class with instance attributes."""

    def __init__(self, factor):
        self.factor = factor

    def scale(self, x):
        """Multiply x by the instance's factor."""
        return x * self.factor


class Polynomial:
    """Class representing a polynomial."""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        """Evaluate polynomial: a*x^2 + b*x + c."""
        return self.a * x ** 2 + self.b * x + self.c


def test_instance_attribute_simple():
    """Test method using simple instance attribute."""
    def f(x):
        scaler = Scaler(2.5)
        return scaler.scale(x)

    # Gradient of 2.5*x is 2.5
    df = tangent.grad(f)

    assert abs(df(3.0) - 2.5) < 1e-10, f"Expected 2.5, got {df(3.0)}"
    assert abs(df(10.0) - 2.5) < 1e-10, f"Expected 2.5, got {df(10.0)}"


def test_instance_attribute_multiple():
    """Test method using multiple instance attributes."""
    def f(x):
        poly = Polynomial(2.0, 3.0, 1.0)  # 2x^2 + 3x + 1
        return poly.evaluate(x)

    # Gradient of 2x^2 + 3x + 1 is 4x + 3
    df = tangent.grad(f)

    # Test at x = 2: gradient should be 4*2 + 3 = 11
    assert abs(df(2.0) - 11.0) < 1e-10, f"Expected 11.0, got {df(2.0)}"

    # Test at x = 5: gradient should be 4*5 + 3 = 23
    assert abs(df(5.0) - 23.0) < 1e-10, f"Expected 23.0, got {df(5.0)}"


def test_instance_attribute_different_instances():
    """Test that different instances with different attributes work correctly."""
    def f(x):
        scaler1 = Scaler(2.0)
        scaler2 = Scaler(3.0)
        return scaler1.scale(x) + scaler2.scale(x)

    # Gradient of 2x + 3x = 5x is 5
    df = tangent.grad(f)

    assert abs(df(3.0) - 5.0) < 1e-10, f"Expected 5.0, got {df(3.0)}"


# =============================================================================
# Phase 3: Method Chaining Tests
# =============================================================================

class ChainedCalculator:
    """Class with methods that call other methods."""

    def square(self, x):
        return x ** 2

    def double(self, x):
        return x * 2

    def square_then_double(self, x):
        """Call square, then double the result."""
        return self.double(self.square(x))

    def combined(self, x):
        """Use multiple method calls in one expression."""
        return self.square(x) + self.double(x)


def test_method_calling_method():
    """Test method that calls another method."""
    def f(x):
        calc = ChainedCalculator()
        return calc.square_then_double(x)

    # square_then_double(x) = 2 * (x^2) = 2x^2
    # Gradient is 4x
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 12
    assert abs(df(3.0) - 12.0) < 1e-10, f"Expected 12.0, got {df(3.0)}"

    # Test at x = 5: gradient should be 20
    assert abs(df(5.0) - 20.0) < 1e-10, f"Expected 20.0, got {df(5.0)}"


def test_method_multiple_chained_calls():
    """Test method with multiple chained method calls."""
    def f(x):
        calc = ChainedCalculator()
        return calc.combined(x)

    # combined(x) = x^2 + 2x
    # Gradient is 2x + 2
    df = tangent.grad(f)

    # Test at x = 3: gradient should be 2*3 + 2 = 8
    assert abs(df(3.0) - 8.0) < 1e-10, f"Expected 8.0, got {df(3.0)}"

    # Test at x = 4: gradient should be 2*4 + 2 = 10
    assert abs(df(4.0) - 10.0) < 1e-10, f"Expected 10.0, got {df(4.0)}"


# =============================================================================
# NumPy Integration Tests
# =============================================================================

class NumpyCalculator:
    """Class with NumPy operations."""

    def sin_plus_square(self, x):
        """Compute sin(x) + x^2."""
        return np.sin(x) + x ** 2

    def array_sum(self, x):
        """Compute sum of array operations."""
        return np.sum(x ** 2)


def test_numpy_operations_in_method():
    """Test method containing NumPy operations."""
    def f(x):
        calc = NumpyCalculator()
        return calc.sin_plus_square(x)

    # Gradient of sin(x) + x^2 is cos(x) + 2x
    df = tangent.grad(f)

    x_test = 1.0
    expected = np.cos(x_test) + 2 * x_test
    result = df(x_test)

    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"


def test_numpy_array_operations():
    """Test method with NumPy array operations."""
    def f(x):
        calc = NumpyCalculator()
        return calc.array_sum(x)

    # Gradient of sum(x^2) is 2x for each element
    df = tangent.grad(f)

    x_test = np.array([1.0, 2.0, 3.0])
    expected = 2 * x_test
    result = df(x_test)

    np.testing.assert_array_almost_equal(result, expected, decimal=10)


# =============================================================================
# Edge Cases and Advanced Tests
# =============================================================================

class MultiParameter:
    """Class with multi-parameter methods."""

    def multiply_add(self, x, y):
        """Compute x * y + x^2."""
        return x * y + x ** 2


class NestedClass:
    """Class for testing nested instantiation."""

    def __init__(self, inner_calc):
        self.inner = inner_calc

    def compute(self, x):
        """Use the inner calculator."""
        return self.inner.square(x) * 2


def test_method_multiple_parameters():
    """Test method with multiple parameters."""
    def f(x, y):
        calc = MultiParameter()
        return calc.multiply_add(x, y)

    # Gradient w.r.t. x of (x*y + x^2) is y + 2x
    df = tangent.grad(f, wrt=(0,))  # Use tuple

    # Test at x=3, y=4: gradient should be 4 + 2*3 = 10
    assert abs(df(3.0, 4.0) - 10.0) < 1e-10


def test_method_wrt_second_parameter():
    """Test gradient w.r.t. second parameter."""
    def f(x, y):
        calc = MultiParameter()
        return calc.multiply_add(x, y)

    # Gradient w.r.t. y of (x*y + x^2) is x
    df = tangent.grad(f, wrt=(1,))  # Use tuple

    # Test at x=3, y=4: gradient should be 3
    assert abs(df(3.0, 4.0) - 3.0) < 1e-10


def test_class_instantiation_with_no_args():
    """Test class instantiation without __init__ arguments."""
    def f(x):
        calc = SimpleCalculator()
        return calc.square(x) + calc.cube(x)

    df = tangent.grad(f)

    # Gradient of x^2 + x^3 at x=2 is 2*2 + 3*4 = 16
    assert abs(df(2.0) - 16.0) < 1e-10


# =============================================================================
# Test Suite Summary
# =============================================================================

if __name__ == '__main__':
    """
    Test Suite for Class Support in Tangent

    Phase 1: Basic Method Inlining (5 tests)
    - test_simple_method_square
    - test_simple_method_cube
    - test_simple_method_constant
    - test_multiple_methods_same_class

    Phase 2: Instance Attributes (3 tests)
    - test_instance_attribute_simple
    - test_instance_attribute_multiple
    - test_instance_attribute_different_instances

    Phase 3: Method Chaining (2 tests)
    - test_method_calling_method
    - test_method_multiple_chained_calls

    NumPy Integration (2 tests)
    - test_numpy_operations_in_method
    - test_numpy_array_operations

    Edge Cases (3 tests)
    - test_method_multiple_parameters
    - test_method_wrt_second_parameter
    - test_class_instantiation_with_no_args

    Total: 15 tests

    Run with: pytest tests/test_classes.py -v
    """
    pytest.main([__file__, '-v'])
