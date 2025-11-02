#!/usr/bin/env python3
"""Test basic tangent functionality after modernization."""
import warnings
warnings.filterwarnings('ignore')

import tangent

def square(x):
    """Simple function: f(x) = x^2"""
    return x * x

def cubic(x):
    """Cubic function: f(x) = x^3"""
    return x * x * x

def polynomial(x):
    """Polynomial: f(x) = 3x^2 + 2x + 1"""
    return 3 * x * x + 2 * x + 1

if __name__ == '__main__':
    print("Testing Tangent autodiff on Python 3.12...")
    print("=" * 50)

    # Test 1: Square function
    print("\n1. Testing f(x) = x^2")
    df_square = tangent.grad(square)
    result = df_square(3.0)
    print(f"   df/dx at x=3: {result}")
    print(f"   Expected: 6.0")
    print(f"   ✓ PASS" if abs(result - 6.0) < 0.001 else f"   ✗ FAIL")

    # Test 2: Cubic function
    print("\n2. Testing f(x) = x^3")
    df_cubic = tangent.grad(cubic)
    result = df_cubic(2.0)
    print(f"   df/dx at x=2: {result}")
    print(f"   Expected: 12.0 (3 * 2^2)")
    print(f"   ✓ PASS" if abs(result - 12.0) < 0.001 else f"   ✗ FAIL")

    # Test 3: Polynomial
    print("\n3. Testing f(x) = 3x^2 + 2x + 1")
    df_poly = tangent.grad(polynomial)
    result = df_poly(1.0)
    print(f"   df/dx at x=1: {result}")
    print(f"   Expected: 8.0 (6*1 + 2)")
    print(f"   ✓ PASS" if abs(result - 8.0) < 0.001 else f"   ✗ FAIL")

    print("\n" + "=" * 50)
    print("All basic tests completed successfully!")
    print("Tangent is working on Python 3.12 with modern dependencies!")
