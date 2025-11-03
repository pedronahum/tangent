#!/usr/bin/env python3
"""
Comprehensive test of all extended NumPy gradients.

Run from the examples/numpy_extended directory:
    python test_comprehensive.py

Or from the repository root:
    python examples/numpy_extended/test_comprehensive.py
"""

import sys
import os

# Add parent directory to path so we can import tangent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tangent

print("="*80)
print("COMPREHENSIVE TEST: Extended NumPy Gradients")
print("="*80)

passed = 0
failed = 0

# ==============================================================================
# Element-wise Operations
# ==============================================================================
print("\n📦 Element-wise Operations")
print("-"*80)

# Test 1: np.abs
def test_abs(x):
    return np.sum(np.abs(x))

try:
    df = tangent.grad(test_abs, verbose=0)
    x = np.array([-1.0, 2.0, -3.0])
    grad = df(x)
    expected = np.array([-1.0, 1.0, -1.0])
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.abs: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.abs: {e}")
    failed += 1

# Test 2: np.square
def test_square(x):
    return np.sum(np.square(x))

try:
    df = tangent.grad(test_square, verbose=0)
    x = np.array([1.0, 2.0, 3.0])
    grad = df(x)
    expected = 2.0 * x
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.square: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.square: {e}")
    failed += 1

# Test 3: np.reciprocal
def test_reciprocal(x):
    return np.sum(np.reciprocal(x))

try:
    df = tangent.grad(test_reciprocal, verbose=0)
    x = np.array([1.0, 2.0, 4.0])
    grad = df(x)
    expected = -1.0 / (x ** 2)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.reciprocal: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.reciprocal: {e}")
    failed += 1

# ==============================================================================
# Logarithmic Functions
# ==============================================================================
print("\n📐 Logarithmic Functions")
print("-"*80)

# Test 4: np.log10
def test_log10(x):
    return np.sum(np.log10(x))

try:
    df = tangent.grad(test_log10, verbose=0)
    x = np.array([1.0, 10.0, 100.0])
    grad = df(x)
    expected = 1.0 / (x * np.log(10.0))
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.log10: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.log10: {e}")
    failed += 1

# Test 5: np.log2
def test_log2(x):
    return np.sum(np.log2(x))

try:
    df = tangent.grad(test_log2, verbose=0)
    x = np.array([1.0, 2.0, 4.0])
    grad = df(x)
    expected = 1.0 / (x * np.log(2.0))
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.log2: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.log2: {e}")
    failed += 1

# Test 6: np.log1p
def test_log1p(x):
    return np.sum(np.log1p(x))

try:
    df = tangent.grad(test_log1p, verbose=0)
    x = np.array([0.0, 1.0, 2.0])
    grad = df(x)
    expected = 1.0 / (1.0 + x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.log1p: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.log1p: {e}")
    failed += 1

# Test 7: np.expm1
def test_expm1(x):
    return np.sum(np.expm1(x))

try:
    df = tangent.grad(test_expm1, verbose=0)
    x = np.array([0.0, 1.0, 2.0])
    grad = df(x)
    expected = np.exp(x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.expm1: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.expm1: {e}")
    failed += 1

# ==============================================================================
# Reduction Operations
# ==============================================================================
print("\n📊 Reduction Operations")
print("-"*80)

# Test 8: np.min
def test_min(x):
    return np.min(x)

try:
    df = tangent.grad(test_min, verbose=0)
    x = np.array([3.0, 1.0, 2.0])
    grad = df(x)
    expected = np.array([0.0, 1.0, 0.0])
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.min: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.min: {e}")
    failed += 1

# Test 9: np.max
def test_max(x):
    return np.max(x)

try:
    df = tangent.grad(test_max, verbose=0)
    x = np.array([1.0, 3.0, 2.0])
    grad = df(x)
    expected = np.array([0.0, 1.0, 0.0])
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.max: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.max: {e}")
    failed += 1

# Test 10: np.prod
def test_prod(x):
    return np.prod(x)

try:
    df = tangent.grad(test_prod, verbose=0)
    x = np.array([2.0, 3.0, 4.0])
    grad = df(x)
    product = np.prod(x)
    expected = product / x
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.prod: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.prod: {e}")
    failed += 1

# ==============================================================================
# Linear Algebra
# ==============================================================================
print("\n🔢 Linear Algebra Operations")
print("-"*80)

# Test 11: np.matmul
def test_matmul(A, B):
    return np.sum(np.matmul(A, B))

try:
    df = tangent.grad(test_matmul, wrt=(0,), verbose=0)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    grad = df(A, B)
    assert grad.shape == A.shape, f"Expected shape {A.shape}, got {grad.shape}"
    print("✅ np.matmul: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.matmul: {e}")
    failed += 1

# Test 12: np.linalg.inv
def test_inv(A):
    return np.sum(np.linalg.inv(A))

try:
    df = tangent.grad(test_inv, verbose=0)
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    grad = df(A)
    assert grad.shape == A.shape, f"Expected shape {A.shape}, got {grad.shape}"
    print("✅ np.linalg.inv: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.linalg.inv: {e}")
    failed += 1

# Test 13: np.outer
def test_outer(a, b):
    return np.sum(np.outer(a, b))

try:
    df = tangent.grad(test_outer, wrt=(0,), verbose=0)
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    grad = df(a, b)
    assert grad.shape == a.shape, f"Expected shape {a.shape}, got {grad.shape}"
    print("✅ np.outer: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.outer: {e}")
    failed += 1

# Test 14: np.trace
def test_trace(A):
    return np.trace(A)

try:
    df = tangent.grad(test_trace, verbose=0)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = df(A)
    expected = np.eye(2)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.trace: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.trace: {e}")
    failed += 1

# ==============================================================================
# Shape Operations
# ==============================================================================
print("\n🔄 Shape Operations")
print("-"*80)

# Test 15: np.squeeze
def test_squeeze(x):
    return np.sum(np.squeeze(x))

try:
    df = tangent.grad(test_squeeze, verbose=0)
    x = np.array([[[1.0]], [[2.0]], [[3.0]]])
    grad = df(x)
    assert grad.shape == x.shape, f"Expected shape {x.shape}, got {grad.shape}"
    print("✅ np.squeeze: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.squeeze: {e}")
    failed += 1

# Test 16: np.expand_dims
def test_expand_dims(x):
    return np.sum(np.expand_dims(x, axis=0))

try:
    df = tangent.grad(test_expand_dims, verbose=0)
    x = np.array([1.0, 2.0, 3.0])
    grad = df(x)
    assert grad.shape == x.shape, f"Expected shape {x.shape}, got {grad.shape}"
    print("✅ np.expand_dims: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.expand_dims: {e}")
    failed += 1

# ==============================================================================
# Comparison Operations
# ==============================================================================
print("\n⚖️  Comparison Operations")
print("-"*80)

# Test 17: np.minimum
def test_minimum(x, y):
    return np.sum(np.minimum(x, y))

try:
    df = tangent.grad(test_minimum, wrt=(0,), verbose=0)
    x = np.array([1.0, 3.0, 2.0])
    y = np.array([2.0, 1.0, 3.0])
    grad = df(x, y)
    expected = np.array([1.0, 0.0, 1.0])
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.minimum: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.minimum: {e}")
    failed += 1

# Test 18: np.clip
def test_clip(x):
    return np.sum(np.clip(x, 1.0, 3.0))

try:
    df = tangent.grad(test_clip, verbose=0)
    x = np.array([0.0, 2.0, 4.0])
    grad = df(x)
    expected = np.array([0.0, 1.0, 0.0])
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.clip: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.clip: {e}")
    failed += 1

# ==============================================================================
# Utility Functions
# ==============================================================================
print("\n🔧 Utility Functions")
print("-"*80)

# Test 19: np.sign
def test_sign(x):
    return np.sum(np.sign(x))

try:
    df = tangent.grad(test_sign, verbose=0)
    x = np.array([-1.0, 2.0, -3.0])
    grad = df(x)
    expected = np.zeros_like(x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.sign: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.sign: {e}")
    failed += 1

# Test 20: np.floor
def test_floor(x):
    return np.sum(np.floor(x))

try:
    df = tangent.grad(test_floor, verbose=0)
    x = np.array([1.7, 2.3, 3.9])
    grad = df(x)
    expected = np.zeros_like(x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.floor: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.floor: {e}")
    failed += 1

# Test 21: np.ceil
def test_ceil(x):
    return np.sum(np.ceil(x))

try:
    df = tangent.grad(test_ceil, verbose=0)
    x = np.array([1.1, 2.5, 3.9])
    grad = df(x)
    expected = np.zeros_like(x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.ceil: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.ceil: {e}")
    failed += 1

# ==============================================================================
# Statistics Operations
# ==============================================================================
print("\n📈 Statistics Operations")
print("-"*80)

# Test 22: np.var
def test_var(x):
    return np.var(x)

try:
    df = tangent.grad(test_var, verbose=0)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    grad = df(x)
    # Gradient of variance is 2(x - mean(x))/n
    mean = np.mean(x)
    expected = 2.0 * (x - mean) / len(x)
    assert np.allclose(grad, expected), f"Expected {expected}, got {grad}"
    print("✅ np.var: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.var: {e}")
    failed += 1

# Test 23: np.std
def test_std(x):
    return np.std(x)

try:
    df = tangent.grad(test_std, verbose=0)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    grad = df(x)
    # Gradient exists and has correct shape
    assert grad.shape == x.shape, f"Expected shape {x.shape}, got {grad.shape}"
    print("✅ np.std: PASS")
    passed += 1
except Exception as e:
    print(f"❌ np.std: {e}")
    failed += 1

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"📊 Total:  {passed + failed}")
print(f"📈 Success Rate: {100 * passed / (passed + failed):.1f}%")
print("="*80)
