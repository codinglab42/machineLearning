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
```

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

### Example 1: StandardScaler

```python
import ml_core as ml

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

```
### Example 2: MinMaxScaler

```python
import ml_core as ml

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

```
### Example 3: MathUtils - Sigmoid

```python
import ml_core as ml
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

```
### Example 4: MathUtils - Weight Initialization

```python
import ml_core as ml

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

```
### Example 5: MathUtils - One-Hot Encoding

```python
import ml_core as ml
import numpy as np

# Labels
labels = np.array([0, 1, 2, 0, 2, 1])

# One-hot encode
encoded = ml.MathUtils.one_hot_encode(labels, 3)
print("Labels:", labels)
print("One-hot encoded:")
print(encoded)

```
### Example 6: MathUtils - Train/Test Split

```python
import ml_core as ml
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

```
### Example 7: MathUtils - Gradient Computation

```python
import ml_core as ml
import numpy as np

# Generate data
X = np.random.randn(50, 3)
y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(50) * 0.1

# Add intercept
X_with_intercept = ml.MathUtils.add_intercept(X)

# Initialize theta
theta = np.zeros(4)

# Compute gradient
grad = ml.MathUtils.compute_gradient_linear(X_with_intercept, y, theta, 0.01)
print("Gradient:", grad)

# Update theta
learning_rate = 0.01
theta = theta - learning_rate * grad
print("Updated theta:", theta)

```
### Example 8: Using Scaler with Neural Network

```python
import ml_core as ml
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

```
## Important Notes

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
import ml_core as ml
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

    [Model bindings](../models/README.md)
    [Preprocessing best practices](https://scikit-learn.org/stable/modules/preprocessing.html)


## Riepilogo dei file del binding utils

bindings/utils/
├── README.md # ✅ Documentazione completa
├── scaler_base.h/cpp # ✅ Classe base Scaler
├── standard_scaler.h/cpp # ✅ StandardScaler
├── minmax_scaler.h/cpp # ✅ MinMaxScaler
└── math_utils.h/cpp # ✅ MathUtils


## Build configuration

See the current [Python binding build configuration](../../CMakeLists.txt).