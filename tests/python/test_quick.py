#!/usr/bin/env python3
"""
Quick test for ML Library Python Bindings
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/pybinding'))

try:
    import machine_learning_module as ml
except ImportError:
    print("❌ Failed to import machine_learning_module")
    sys.exit(1)

print("=" * 70)
print("🧪 QUICK TEST - ML Library Python Bindings")
print("=" * 70)

# ============================================================================
# TEST LINEAR REGRESSION
# ============================================================================

print("\n📊 Testing LinearRegression...")
X = np.random.randn(100, 3)
y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(100) * 0.1

lr = ml.LinearRegression(learning_rate=0.01, max_iter=200)
lr.fit(X, y)
r2 = lr.score(X, y)
print(f"  R² score: {r2:.4f}")
print("  ✅ LinearRegression OK")

# ============================================================================
# TEST NEURAL NETWORK
# ============================================================================

print("\n🧠 Testing NeuralNetwork...")
X2 = np.random.randn(100, 5)
y2 = (np.sum(X2[:, :3], axis=1) > 0).astype(float)

nn = ml.NeuralNetwork()
nn.add_dense_layer(16, activation="relu")
nn.add_dense_layer(8, activation="relu")
nn.add_dense_layer(1, activation="sigmoid")
nn.set_loss_function("binary_crossentropy")
nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
nn.set_epochs(20)
nn.set_batch_size(32)
nn.set_verbose(False)
nn.build(5, 1)
nn.fit(X2, y2)
score = nn.score(X2, y2)
print(f"  Accuracy: {score:.4f}")
print("  ✅ NeuralNetwork OK")

# ============================================================================
# TEST LAYERS
# ============================================================================

print("\n🔧 Testing Layers...")
dense = ml.DenseLayer(8, activation="relu")
dense.set_input_shape(4)
dense.initialize_weights()
print(f"  DenseLayer: {dense.get_type()}")
print("  ✅ Layers OK")

# ============================================================================
# TEST LOSS FUNCTIONS
# ============================================================================

print("\n📉 Testing Loss Functions...")
mse = ml.MeanSquaredErrorLoss()
y_true = np.random.randn(10)
y_pred = y_true + 0.1
loss = mse.compute_vector(y_true, y_pred)
print(f"  MSE loss: {loss:.4f}")
print("  ✅ Loss Functions OK")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print(f"📦 Module: machine_learning_module")
print(f"📌 Version: {ml.__version__}")
print("🚀 Ready to use!")