#!/usr/bin/env python3
"""
Demo: Extended NumPy Gradients in Tangent

This demo showcases all 27 newly implemented NumPy operations working with
Tangent's automatic differentiation.

Run from the examples/numpy_extended directory:
    python demo.py

Or from the repository root:
    python examples/numpy_extended/demo.py
"""

import sys
import os

# Add parent directory to path so we can import tangent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tangent

print("="*80)
print("DEMO: Extended NumPy Operations in Tangent")
print("="*80)

# ==============================================================================
# Example 1: Machine Learning Loss Function
# ==============================================================================
print("\n📊 Example 1: Machine Learning Loss Function")
print("-"*80)

def mse_with_regularization(weights, X, y, lambda_reg=0.01):
    """Mean squared error with L2 regularization."""
    predictions = np.matmul(X, weights)
    errors = predictions - y
    mse = np.mean(np.square(errors))
    l2_penalty = lambda_reg * np.sum(np.square(weights))
    return mse + l2_penalty

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(10, 3)
y = np.random.randn(10)
weights = np.array([1.0, -0.5, 0.3])

# Compute gradient
grad_fn = tangent.grad(mse_with_regularization, wrt=(0,), verbose=0)
gradient = grad_fn(weights, X, y, 0.01)

print(f"Weights: {weights}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.matmul, np.square, np.mean")

# ==============================================================================
# Example 2: Signal Processing
# ==============================================================================
print("\n📡 Example 2: Signal Processing")
print("-"*80)

def signal_energy_log_scale(signal):
    """Compute log-scale energy of signal after clipping."""
    clipped = np.clip(signal, -2.0, 2.0)
    absolute_values = np.abs(clipped)
    squared = np.square(absolute_values)
    energy = np.sum(squared)
    # Use log1p for numerical stability near zero
    return np.log1p(energy)

signal = np.array([-3.0, -1.0, 0.5, 1.5, 3.5])
grad_fn = tangent.grad(signal_energy_log_scale, verbose=0)
gradient = grad_fn(signal)

print(f"Signal: {signal}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.clip, np.abs, np.square, np.log1p")

# ==============================================================================
# Example 3: Statistics and Normalization
# ==============================================================================
print("\n📈 Example 3: Statistics and Normalization")
print("-"*80)

def normalized_variance_loss(x):
    """Loss based on normalized variance (coefficient of variation)."""
    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    # Coefficient of variation: std / mean
    cv = std / mean
    return cv + 0.1 * var

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
grad_fn = tangent.grad(normalized_variance_loss, verbose=0)
gradient = grad_fn(data)

print(f"Data: {data}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.std, np.var, np.mean")

# ==============================================================================
# Example 4: Matrix Operations
# ==============================================================================
print("\n🔢 Example 4: Matrix Operations")
print("-"*80)

def matrix_function(A):
    """Complex function involving multiple matrix operations."""
    # Compute matrix inverse
    A_inv = np.linalg.inv(A)
    # Compute trace
    tr = np.trace(A_inv)
    # Compute outer product of diagonal
    diag = np.array([A[0,0], A[1,1]])
    outer_prod = np.outer(diag, diag)
    # Sum everything
    return tr + np.sum(outer_prod)

A = np.array([[2.0, 0.5], [0.5, 3.0]])
grad_fn = tangent.grad(matrix_function, verbose=0)
gradient = grad_fn(A)

print(f"Matrix A:\n{A}")
print(f"Gradient:\n{gradient}")
print(f"✅ Uses: np.linalg.inv, np.trace, np.outer")

# ==============================================================================
# Example 5: Reduction Operations
# ==============================================================================
print("\n📊 Example 5: Min/Max/Prod Operations")
print("-"*80)

def range_based_loss(x):
    """Loss based on range and product of values."""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    prod_val = np.prod(x)
    return range_val + 0.01 * prod_val

values = np.array([1.0, 2.0, 3.0, 4.0])
grad_fn = tangent.grad(range_based_loss, verbose=0)
gradient = grad_fn(values)

print(f"Values: {values}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.min, np.max, np.prod")

# ==============================================================================
# Example 6: Element-wise Comparison
# ==============================================================================
print("\n⚖️  Example 6: Element-wise Operations")
print("-"*80)

def robust_loss(predictions, targets):
    """Robust loss using minimum and clipping."""
    # Use minimum to cap individual errors
    errors = np.minimum(np.abs(predictions - targets), 2.0)
    # Clip extreme values
    clipped_errors = np.clip(errors, 0.1, 1.5)
    return np.sum(clipped_errors)

pred = np.array([1.0, 3.0, 5.0])
target = np.array([1.5, 2.0, 4.0])
grad_fn = tangent.grad(robust_loss, wrt=(0,), verbose=0)
gradient = grad_fn(pred, target)

print(f"Predictions: {pred}")
print(f"Targets: {target}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.minimum, np.abs, np.clip")

# ==============================================================================
# Example 7: Logarithmic Operations
# ==============================================================================
print("\n📐 Example 7: Different Logarithm Bases")
print("-"*80)

def multi_scale_log_loss(x):
    """Loss using multiple logarithm bases for different scales."""
    # Base 10 for large scale
    log10_term = np.sum(np.log10(x + 1))
    # Base 2 for binary-like scaling
    log2_term = np.sum(np.log2(x + 1))
    # Natural log for small values (using log1p)
    logn_term = np.sum(np.log1p(x))
    return log10_term + 0.1 * log2_term + logn_term

x = np.array([1.0, 10.0, 100.0])
grad_fn = tangent.grad(multi_scale_log_loss, verbose=0)
gradient = grad_fn(x)

print(f"Input: {x}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.log10, np.log2, np.log1p")

# ==============================================================================
# Example 8: Shape Operations
# ==============================================================================
print("\n🔄 Example 8: Shape Manipulation")
print("-"*80)

def shape_aware_loss(x):
    """Loss that manipulates shapes."""
    # Expand dimensions
    expanded = np.expand_dims(x, axis=0)
    # Squeeze back
    squeezed = np.squeeze(expanded)
    # Sum
    return np.sum(squeezed)

x = np.array([1.0, 2.0, 3.0])
grad_fn = tangent.grad(shape_aware_loss, verbose=0)
gradient = grad_fn(x)

print(f"Input: {x}")
print(f"Gradient: {gradient}")
print(f"✅ Uses: np.expand_dims, np.squeeze")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*80)
print("✅ All 8 examples completed successfully!")
print("="*80)
print("\n📚 Operations Demonstrated:")
print("   1. np.abs, np.square, np.matmul - Machine learning")
print("   2. np.clip, np.log1p - Signal processing")
print("   3. np.std, np.var - Statistics")
print("   4. np.linalg.inv, np.trace, np.outer - Linear algebra")
print("   5. np.min, np.max, np.prod - Reductions")
print("   6. np.minimum - Element-wise comparison")
print("   7. np.log10, np.log2 - Different log bases")
print("   8. np.expand_dims, np.squeeze - Shape operations")
print("\n🎉 Tangent now supports 27 additional NumPy operations!")
print("="*80)
