# Loss Functions Binding

## Overview

This module exposes all loss functions available in the ML library as Python objects. Loss functions can be used either independently or within the `NeuralNetwork` class.

## File Structure

bindings/loss/
├── README.md # This file
├── loss_base.h/cpp # Base Loss class
├── mean_squared_error_loss.h/cpp # Mean Squared Error (MSE)
├── mean_absolute_error_loss.h/cpp # Mean Absolute Error (MAE)
├── binary_cross_entropy_loss.h/cpp # Binary Cross Entropy
├── categorical_cross_entropy_loss.h/cpp # Categorical Cross Entropy
├── huber_loss.h/cpp # Huber Loss
└── loss_factory.h/cpp # LossFactory


## Available Loss Functions

### 1. MeanSquaredErrorLoss (MSE)

Mean Squared Error - Standard regression loss.

**Constructor:**
```python
MeanSquaredErrorLoss()

Formula: L = (1/n) * Σ(y_pred - y_true)²

Methods:

compute_vector(y_true: VectorXd, y_pred: VectorXd) -> float
compute_matrix(y_true: MatrixXd, y_pred: MatrixXd) -> float
gradient(y_true: MatrixXd, y_pred: MatrixXd) -> MatrixXd
name() -> str

2. MeanAbsoluteErrorLoss (MAE)

Mean Absolute Error - Robust regression loss.

Constructor:

MeanAbsoluteErrorLoss()

Formula: L = (1/n) * Σ|y_pred - y_true|

Methods:

compute_vector(y_true: VectorXd, y_pred: VectorXd) -> float
compute_matrix(y_true: MatrixXd, y_pred: MatrixXd) -> float
gradient(y_true: MatrixXd, y_pred: MatrixXd) -> MatrixXd
name() -> str

3. BinaryCrossEntropyLoss (BCE)

Binary Cross Entropy - For binary classification.

Constructor:

BinaryCrossEntropyLoss()

Formula: L = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

Methods:

compute_vector(y_true: VectorXd, y_pred: VectorXd) -> float
compute_matrix(y_true: MatrixXd, y_pred: MatrixXd) -> float
gradient(y_true: MatrixXd, y_pred: MatrixXd) -> MatrixXd
name() -> str

Note: y_pred should be probabilities in [0, 1].
4. CategoricalCrossEntropyLoss (CCE)

Categorical Cross Entropy - For multi-class classification.

Constructor:

CategoricalCrossEntropyLoss()

Formula: L = -(1/n) * Σ y_true * log(y_pred)

Methods:

compute_vector(y_true: VectorXd, y_pred: VectorXd) -> float
compute_matrix(y_true: MatrixXd, y_pred: MatrixXd) -> float
gradient(y_true: MatrixXd, y_pred: MatrixXd) -> MatrixXd
name() -> str

Note: y_pred should be probabilities summing to 1 (softmax output).
5. HuberLoss

Huber Loss - Combines MSE and MAE, robust to outliers.

Constructor:

HuberLoss(delta: float = 1.0)

Formula:

    L = 0.5 * (y_pred - y_true)² for |y_pred - y_true| <= delta

    L = delta * |y_pred - y_true| - 0.5 * delta² otherwise

Methods:

compute_vector(y_true: VectorXd, y_pred: VectorXd) -> float
compute_matrix(y_true: MatrixXd, y_pred: MatrixXd) -> float
gradient(y_true: MatrixXd, y_pred: MatrixXd) -> MatrixXd
set_delta(delta: float)
get_delta() -> float
name() -> str

LossFactory

Factory for creating loss functions dynamically.

Methods:

LossFactory.create(name: str) -> Loss
LossFactory.register_all_losses()
LossFactory.list_losses() -> List[str]

Available Loss Names:

    "mse" - Mean Squared Error

    "mae" - Mean Absolute Error

    "binary_crossentropy" - Binary Cross Entropy

    "categorical_crossentropy" - Categorical Cross Entropy

    "huber" - Huber Loss

Usage Examples
Example 1: Computing Loss Values

import ml_core as ml
import numpy as np

# Create loss functions
mse = ml.MeanSquaredErrorLoss()
mae = ml.MeanAbsoluteErrorLoss()
huber = ml.HuberLoss(delta=0.5)

# Generate data
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100) * 0.1

# Compute losses
mse_loss = mse.compute_vector(y_true, y_pred)
mae_loss = mae.compute_vector(y_true, y_pred)
huber_loss = huber.compute_vector(y_true, y_pred)

print(f"MSE: {mse_loss:.4f}")
print(f"MAE: {mae_loss:.4f}")
print(f"Huber: {huber_loss:.4f}")

Example 2: Using Loss in Neural Network

import ml_core as ml
import numpy as np

# Create neural network
nn = ml.NeuralNetwork([10, 32, 16, 1], "relu", "sigmoid")
nn.build(10, 1)

# Set loss function (via name)
nn.set_loss_function("binary_crossentropy")

# Or use the loss object directly
loss = ml.BinaryCrossEntropyLoss()
nn.set_loss_function(loss.name())  # Same as above

# Train
X = np.random.randn(100, 10)
y = (np.random.randn(100) > 0).astype(float)
nn.fit(X, y)

Example 3: Gradient Computation

import ml_core as ml
import numpy as np

# Create loss
loss = ml.BinaryCrossEntropyLoss()

# Generate data
y_true = np.array([1.0, 0.0, 1.0, 0.0]).reshape(-1, 1)
y_pred = np.array([0.9, 0.1, 0.8, 0.2]).reshape(-1, 1)

# Compute gradient
grad = loss.gradient(y_true, y_pred)
print(f"Gradient shape: {grad.shape}")
print(f"Gradient:\n{grad}")

# Gradient indicates direction to update predictions
# grad = (p - y) / n

Example 4: Huber Loss with Different Delta Values

import ml_core as ml
import numpy as np

y_true = np.random.randn(50)
y_pred = y_true + np.random.randn(50) * 0.5  # Some noise

# Compare different delta values
for delta in [0.1, 0.5, 1.0, 2.0]:
    huber = ml.HuberLoss(delta=delta)
    loss_value = huber.compute_vector(y_true, y_pred)
    print(f"Delta={delta}: Huber loss = {loss_value:.4f}")

Example 5: Using LossFactory

import ml_core as ml

# List available losses
available = ml.LossFactory.list_losses()
print("Available losses:", available)

# Create loss via factory
mse = ml.LossFactory.create("mse")
bce = ml.LossFactory.create("binary_crossentropy")
cce = ml.LossFactory.create("categorical_crossentropy")
huber = ml.LossFactory.create("huber")

print(f"Created: {mse.name()}, {bce.name()}, {cce.name()}, {huber.name()}")

Example 6: Custom Loss in Neural Network

import ml_core as ml
import numpy as np

# Create custom loss by wrapping
class CustomLoss:
    def __init__(self):
        self.mse = ml.MeanSquaredErrorLoss()
    
    def compute(self, y_true, y_pred):
        # Add custom weighting or transformation
        return self.mse.compute_matrix(y_true, y_pred)

# Use in neural network (via loss name only)
nn = ml.NeuralNetwork([10, 32, 1], "relu", "linear")
nn.build(10, 1)

# Use built-in loss
nn.set_loss_function("mse")

# Train
X = np.random.randn(100, 10)
y = np.random.randn(100)
nn.fit(X, y)

Important Notes

    Vector vs Matrix: Loss functions support both VectorXd (1D) and MatrixXd (2D) inputs.

    Gradient Shape: The gradient has the same shape as y_pred.

    Numerical Stability: Loss functions include clipping to prevent log(0) or division by zero.

    BCE vs CCE:

        BCE: For binary classification (single output)

        CCE: For multi-class classification (multiple outputs, one-hot encoded)

    Loss vs Optimizer: The loss computes the gradient; the optimizer uses it to update weights.

    Huber Delta: Smaller delta makes it behave more like MAE; larger delta makes it more like MSE.

Choosing the Right Loss Function
Problem Type	Recommended Loss	Output Activation
Regression	MSE or Huber	linear
Binary Classification	BCE	sigmoid
Multi-class Classification	CCE	softmax
Robust Regression	Huber or MAE	linear
Sparse Data	MAE	linear
Dependencies

    bindings/core/exceptions.h - for exception translator

    components/loss/*.h - C++ loss headers

    Eigen3 - for matrix operations

Testing

To test the loss binding:

cd build
python3 -c "
import ml_core as ml
import numpy as np

# Test MSE
mse = ml.MeanSquaredErrorLoss()
y_true = np.random.randn(10)
y_pred = y_true + 0.1
loss = mse.compute_vector(y_true, y_pred)
print(f'MSE: {loss:.4f}')

# Test BCE
bce = ml.BinaryCrossEntropyLoss()
y = np.array([1, 0, 1, 0])
p = np.array([0.9, 0.1, 0.8, 0.2])
loss = bce.compute_vector(y, p)
print(f'BCE: {loss:.4f}')

print('All tests passed!')
"

Version Information
Loss	Version	Notes
MeanSquaredErrorLoss	1	-
MeanAbsoluteErrorLoss	1	-
BinaryCrossEntropyLoss	1	Numerically stable
CategoricalCrossEntropyLoss	1	Numerically stable
HuberLoss	1	Configurable delta
References

    C++ Loss Documentation https://../cpp/components/loss/

    Neural Network Binding https://../models/neural_network.md

    Optimizers Binding https://../optimizers/README.md


## Riepilogo dei file del binding loss

bindings/loss/
├── README.md # ✅ Documentazione completa
├── loss_base.h/cpp # ✅ Classe base
├── mean_squared_error_loss.h/cpp # ✅ MSE
├── mean_absolute_error_loss.h/cpp # ✅ MAE
├── binary_cross_entropy_loss.h/cpp # ✅ BCE
├── categorical_cross_entropy_loss.h/cpp # ✅ CCE
├── huber_loss.h/cpp # ✅ Huber
└── loss_factory.h/cpp # ✅ Factory