# Utils Binding

## Overview

This module exposes utility functions and classes for machine learning operations including scaling, mathematical functions, and data preprocessing.

## File Structure

# Utils Binding

## Overview

This module exposes utility functions and classes for machine learning operations including scaling, mathematical functions, and data preprocessing.

## File Structure


bindings/utils/
├── README.md # This file
├── scaler_base.h/cpp # Base Scaler class
├── standard_scaler.h/cpp # StandardScaler (mean=0, std=1)
├── minmax_scaler.h/cpp # MinMaxScaler (range [0, 1])
└── math_utils.h/cpp # Mathematical utilities


## Scaler Classes

### 1. StandardScaler

Standardizes features by removing mean and scaling to unit variance.

**Formula:** `x_scaled = (x - mean) / std`

**Constructors:**
```python
StandardScaler()
StandardScaler(epsilon: float = 1e-8)

Methods:

fit(X: List[List[float]]) -> None
transform(X: List[List[float]]) -> List[List[float]]
fit_transform(X: List[List[float]]) -> List[List[float]]
inverse_transform(X_scaled: List[List[float]]) -> List[List[float]]
get_mean() -> List[float]
get_std() -> List[float]
set_params(mean: List[float], std: List[float]) -> None
get_type() -> str

2. MinMaxScaler

Scales features to a given range, typically [0, 1].

Formula: x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min

Constructors:

MinMaxScaler()
MinMaxScaler(feature_range_min: float = 0.0, feature_range_max: float = 1.0)

Methods:

fit(X: List[List[float]]) -> None
transform(X: List[List[float]]) -> List[List[float]]
fit_transform(X: List[List[float]]) -> List[List[float]]
inverse_transform(X_scaled: List[List[float]]) -> List[List[float]]
get_min() -> List[float]
get_max() -> List[float]
set_params(min: List[float], max: List[float]) -> None
get_type() -> str

MathUtils

Static utility class with mathematical functions for ML.

Activation Functions
Method	Description
sigmoid(z)	Sigmoid activation for scalar
sigmoid_vec(z)	Sigmoid activation for vector
sigmoid_derivative(z)	Sigmoid derivative for scalar
sigmoid_derivative_vec(z)	Sigmoid derivative for vector

Matrix Operations
Method	Description
add_intercept(X)	Add bias column (ones) to matrix
safe_log(v)	Log with clipping to avoid log(0)
safe_log_matrix(m)	Log matrix with clipping

Weight Initialization
Method	Description
he_initialization(input_size, output_size)	He initialization for ReLU
xavier_initialization(input_size, output_size)	Xavier initialization for sigmoid/tanh

Preprocessing
Method	Description
one_hot_encode(labels, num_classes)	One-hot encode labels
normalize_minmax(X, min_vals, max_vals)	Normalize to [0, 1] range
standardize_features(X, mean, std)	Standardize to mean=0, std=1

ML Utilities
Method	Description
train_test_split(X, y, test_size, random_state)	Split data into train/test
accuracy_score(y_true, y_pred)	Compute accuracy

Gradient Computation
Method	Description
compute_gradient_linear(X, y, theta, lambda)	Linear regression gradient
compute_gradient_logistic(X, y, theta, lambda)	Logistic regression gradient

Usage Examples

Example 1: StandardScaler

import machine_learning_module as ml

# Create scaler
scaler = ml.StandardScaler()

# Sample data
X = [
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0]
]

# Fit and transform
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Mean:", scaler.get_mean())
print("Std:", scaler.get_std())
print("Scaled data:", X_scaled)

# Inverse transform
X_original = scaler.inverse_transform(X_scaled)

Example 2: MinMaxScaler

import machine_learning_module as ml

# Create scaler with custom range
scaler = ml.MinMaxScaler(feature_range_min=-1.0, feature_range_max=1.0)

# Sample data
X = [
    [0.0, 10.0],
    [5.0, 20.0],
    [10.0, 30.0]
]

# Fit and transform
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Min:", scaler.get_min())
print("Max:", scaler.get_max())
print("Scaled data:", X_scaled)

Example 3: MathUtils - Sigmoid

import machine_learning_module as ml
import numpy as np

# Scalar sigmoid
print(ml.MathUtils.sigmoid(0.0))   # 0.5
print(ml.MathUtils.sigmoid(1.0))   # ~0.731
print(ml.MathUtils.sigmoid(-1.0))  # ~0.269

# Vector sigmoid
z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
sigmoid_z = ml.MathUtils.sigmoid_vec(z)
print("Sigmoid:", sigmoid_z)

# Sigmoid derivative
derivative = ml.MathUtils.sigmoid_derivative(0.0)
print("Sigmoid derivative at 0:", derivative)

Example 4: MathUtils - Weight Initialization

import machine_learning_module as ml

# He initialization (for ReLU)
weights_he = ml.MathUtils.he_initialization(128, 64)
print(f"He weights shape: {weights_he.shape}")
print(f"Mean: {weights_he.mean():.4f}")
print(f"Std: {weights_he.std():.4f}")

# Xavier initialization (for sigmoid/tanh)
weights_xavier = ml.MathUtils.xavier_initialization(128, 64)
print(f"Xavier weights shape: {weights_xavier.shape}")
print(f"Mean: {weights_xavier.mean():.4f}")
print(f"Std: {weights_xavier.std():.4f}")

Example 5: MathUtils - One-Hot Encoding

import machine_learning_module as ml
import numpy as np

# Labels
labels = np.array([0, 1, 2, 0, 2, 1])

# One-hot encode
encoded = ml.MathUtils.one_hot_encode(labels, 3)
print("Labels:", labels)
print("One-hot encoded:")
print(encoded)

Example 6: MathUtils - Train/Test Split

import machine_learning_module as ml
import numpy as np

# Data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Split
splits = ml.MathUtils.train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = splits[0]
X_test, y_test = splits[1]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

Example 7: MathUtils - Gradient Computation

import machine_learning_module as ml
import numpy as np

# Generate data
X = np.random.randn(50, 3)
y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(50) * 0.1

# Add intercept
X_with_intercept = ml.MathUtils.add_intercept(X)

# Initialize theta
theta = np.zeros(4)

# Compute gradient
grad = ml.MathUtils.compute_gradient_linear(X_with_intercept, y, theta, lambda=0.01)
print("Gradient:", grad)

# Update theta
learning_rate = 0.01
theta = theta - learning_rate * grad
print("Updated theta:", theta)

Example 8: Using Scaler with Neural Network

import machine_learning_module as ml
import numpy as np

# Create data
X = np.random.randn(100, 10) * 10 + 5  # Unscaled data
y = (np.sum(X[:, :3], axis=1) > 0).astype(float)

# Scale data
scaler = ml.StandardScaler()
X_scaled = scaler.fit_transform(X.tolist())

# Convert back to numpy
X_scaled = np.array(X_scaled)

# Train neural network
nn = ml.NeuralNetwork([10, 32, 16, 1], "relu", "sigmoid")
nn.build(10, 1)
nn.set_epochs(100)
nn.fit(X_scaled, y)

print("Training complete!")

Important Notes

    Scaler Format: Scalers work with lists of lists, not numpy arrays directly. Convert as needed.

    Fitting: Always call fit() before transform() or use fit_transform().

    Inverse Transform: Only works after fitting.

    Epsilon: StandardScaler uses epsilon to avoid division by zero.

    Feature Range: MinMaxScaler can map to any range, not just [0, 1].

    Memory: Large datasets may consume significant memory during scaling.

Dependencies

    bindings/core/exceptions.h - for exception translator

    utils/*.h - C++ utility headers

    Eigen3 - for matrix operations

Testing

To test the utils binding:

cd build
python3 -c "
import machine_learning_module as ml
import numpy as np

# Test StandardScaler
scaler = ml.StandardScaler()
X = [[1, 2], [3, 4], [5, 6]]
scaler.fit(X)
X_scaled = scaler.transform(X)
print('StandardScaler OK!')

# Test MathUtils
result = ml.MathUtils.sigmoid(0.0)
print(f'Sigmoid(0) = {result}')

# Test weight initialization
weights = ml.MathUtils.he_initialization(10, 5)
print(f'He weights shape: {weights.shape}')

print('All tests passed!')
"

Version Information
Utility	Version	Notes
StandardScaler	1	-
MinMaxScaler	1	-
MathUtils	1	-
References

    C++ Utils Documentation https://../cpp/utils/

    Neural Network Binding https://../models/neural_network.md

    Preprocessing Best Practices https://scikit-learn.org/stable/modules/preprocessing.html


## Riepilogo dei file del binding utils

bindings/utils/
├── README.md # ✅ Documentazione completa
├── scaler_base.h/cpp # ✅ Classe base Scaler
├── standard_scaler.h/cpp # ✅ StandardScaler
├── minmax_scaler.h/cpp # ✅ MinMaxScaler
└── math_utils.h/cpp # ✅ MathUtils


## Aggiornamento del CMakeLists.txt

```cmake
# pybinding/CMakeLists.txt
pybind11_add_module(machine_learning_module
    machine_learning_module.cpp
    # Core
    bindings/core/enums.cpp
    bindings/core/exceptions.cpp
    bindings/core/module.cpp
    # Models
    bindings/models/estimator.cpp
    bindings/models/linear_regression.cpp
    bindings/models/logistic_regression.cpp
    bindings/models/neural_network.cpp
    # Layers
    bindings/layers/layer_base.cpp
    bindings/layers/dense_layer.cpp
    bindings/layers/conv2d_layer.cpp
    bindings/layers/pooling_layer.cpp
    bindings/layers/flatten_layer.cpp
    bindings/layers/dropout_layer.cpp
    bindings/layers/batch_norm_layer.cpp
    bindings/layers/simple_rnn_layer.cpp
    bindings/layers/lstm_layer.cpp
    bindings/layers/gru_layer.cpp
    bindings/layers/layer_factory.cpp
    # Loss
    bindings/loss/loss_base.cpp
    bindings/loss/mean_squared_error_loss.cpp
    bindings/loss/mean_absolute_error_loss.cpp
    bindings/loss/binary_cross_entropy_loss.cpp
    bindings/loss/categorical_cross_entropy_loss.cpp
    bindings/loss/huber_loss.cpp
    bindings/loss/loss_factory.cpp
    # Optimizers
    bindings/optimizers/optimizer_base.cpp
    bindings/optimizers/sgd_optimizer.cpp
    bindings/optimizers/momentum_optimizer.cpp
    bindings/optimizers/adam_optimizer.cpp
    bindings/optimizers/optimizer_factory.cpp
    # Regularizers
    bindings/regularizers/regularizer_base.cpp
    bindings/regularizers/l1_regularizer.cpp
    bindings/regularizers/l2_regularizer.cpp
    bindings/regularizers/elastic_net_regularizer.cpp
    bindings/regularizers/regularizer_factory.cpp
    # Utils
    bindings/utils/scaler_base.cpp
    bindings/utils/standard_scaler.cpp
    bindings/utils/minmax_scaler.cpp
    bindings/utils/math_utils.cpp
)

target_link_libraries(machine_learning_module PRIVATE ml_library)