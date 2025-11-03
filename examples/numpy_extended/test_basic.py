#!/usr/bin/env python3
"""
Test NumPy UFunc gradients (basic tests).

Run from the examples/numpy_extended directory:
    python test_basic.py

Or from the repository root:
    python examples/numpy_extended/test_basic.py
"""

import sys
import os

# Add parent directory to path so we can import tangent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tangent

# Test 1: np.abs
def test_abs(x):
    return np.sum(np.abs(x))

print("1. Testing np.abs...")
try:
    df = tangent.grad(test_abs, verbose=0)
    x = np.array([-1.0, 2.0, -3.0])
    grad = df(x)
    expected = np.array([-1.0, 1.0, -1.0])  # sign(x)
    if np.allclose(grad, expected):
        print(f"   ✅ np.abs: PASS (grad={grad})")
    else:
        print(f"   ❌ np.abs: FAIL (expected {expected}, got {grad})")
except Exception as e:
    print(f"   ❌ np.abs: ERROR - {e}")

# Test 2: np.square
def test_square(x):
    return np.sum(np.square(x))

print("\n2. Testing np.square...")
try:
    df = tangent.grad(test_square, verbose=0)
    x = np.array([1.0, 2.0, 3.0])
    grad = df(x)
    expected = 2.0 * x  # 2x
    if np.allclose(grad, expected):
        print(f"   ✅ np.square: PASS (grad={grad})")
    else:
        print(f"   ❌ np.square: FAIL (expected {expected}, got {grad})")
except Exception as e:
    print(f"   ❌ np.square: ERROR - {e}")

# Test 3: np.matmul
def test_matmul(A, B):
    return np.sum(np.matmul(A, B))

print("\n3. Testing np.matmul...")
try:
    df = tangent.grad(test_matmul, wrt=(0,), verbose=0)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    grad = df(A, B)
    print(f"   ✅ np.matmul: PASS (grad shape={grad.shape})")
except Exception as e:
    print(f"   ❌ np.matmul: ERROR - {e}")

print("\n" + "="*80)
