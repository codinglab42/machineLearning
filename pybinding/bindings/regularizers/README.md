# Regularizers Binding

## Overview

This module exposes all regularizers available in the ML library as Python objects. Regularizers add penalty terms to the loss function to prevent overfitting.

## File Structure

bindings/regularizers/
├── README.md # This file
├── regularizer_base.h/cpp # Base Regularizer class
├── l1_regularizer.h/cpp # L1 Regularizer (Lasso)
├── l2_regularizer.h/cpp # L2 Regularizer (Ridge)
├── elastic_net_regularizer.h/cpp # Elastic Net Regularizer
└── regularizer_factory.h/cpp # RegularizerFactory


## Available Regularizers

### 1. L1Regularizer (Lasso)

L1 regularization promotes sparsity in weights.

**Constructor:**
```python
L1Regularizer(strength: float = 0.01)

Formula: R(w) = λ * Σ|w|
Gradient: ∂R/∂w = λ * sign(w)

Properties:

    Encourages sparse weights (many zeros)

    Good for feature selection

    Can be used for dimensionality reduction

2. L2Regularizer (Ridge)

L2 regularization penalizes large weights.

Constructor:

L2Regularizer(strength: float = 0.01)

Formula: R(w) = 0.5 * λ * Σw²
Gradient: ∂R/∂w = λ * w

Properties:

    Encourages small weights (weight decay)

    Prevents overfitting

    Smoother optimization landscape

3. ElasticNetRegularizer

Combines L1 and L2 regularization.

Constructor:

ElasticNetRegularizer(strength: float = 0.01, l1_ratio: float = 0.5)

Formula: R(w) = λ * (l1_ratio * Σ|w| + (1-l1_ratio) * 0.5 * Σw²)
Gradient: ∂R/∂w = λ * (l1_ratio * sign(w) + (1-l1_ratio) * w)

Properties:

    Combines advantages of L1 and L2

    l1_ratio=0 → pure L2

    l1_ratio=1 → pure L1

    Good when features are correlated

RegularizerFactory

Factory for creating regularizers dynamically.

Methods:

RegularizerFactory.create_by_type(type: RegularizerType, strength: float = 0.01,
                                  params: Dict[str, float] = {}) -> Regularizer
RegularizerFactory.create_by_name(type_str: str, strength: float = 0.01,
                                  params: Dict[str, float] = {}) -> Regularizer
RegularizerFactory.string_to_type(type_str: str) -> RegularizerType
RegularizerFactory.type_to_string(type: RegularizerType) -> str
RegularizerFactory.list_regularizers() -> List[str]

Available Regularizer Names:

    "none" - No regularization

    "l1" - L1 Regularizer

    "l2" - L2 Regularizer

    "elastic_net" - Elastic Net Regularizer

Usage Examples
Example 1: Creating and Using Regularizers

import ml_core as ml
import numpy as np

# Create different regularizers
l1 = ml.L1Regularizer(strength=0.01)
l2 = ml.L2Regularizer(strength=0.01)
elastic = ml.ElasticNetRegularizer(strength=0.01, l1_ratio=0.5)

# Sample weights
weights = np.random.randn(10, 10)

# Compute regularization loss
l1_loss = l1.compute_loss_matrix(weights)
l2_loss = l2.compute_loss_matrix(weights)
elastic_loss = elastic.compute_loss_matrix(weights)

print(f"L1 loss: {l1_loss:.4f}")
print(f"L2 loss: {l2_loss:.4f}")
print(f"Elastic Net loss: {elastic_loss:.4f}")

# Compute gradients
l1_grad = l1.compute_gradient_matrix(weights)
l2_grad = l2.compute_gradient_matrix(weights)
elastic_grad = elastic.compute_gradient_matrix(weights)

print(f"L1 grad norm: {np.linalg.norm(l1_grad):.4f}")
print(f"L2 grad norm: {np.linalg.norm(l2_grad):.4f}")
print(f"Elastic grad norm: {np.linalg.norm(elastic_grad):.4f}")

Example 2: L1 Effect on Sparsity

import ml_core as ml
import numpy as np
import matplotlib.pyplot as plt

# Create L1 regularizer
l1 = ml.L1Regularizer(strength=0.1)

# Test different weight values
weights = np.linspace(-2, 2, 100).reshape(-1, 1)
losses = [l1.compute_loss_matrix(np.array([[w]])) for w in weights.flatten()]
gradients = [l1.compute_gradient_matrix(np.array([[w]]))[0, 0] for w in weights.flatten()]

print("Effect of L1 regularization:")
print(f"Weight=0.5: loss={l1.compute_loss_matrix(np.array([[0.5]])):.4f}, grad={l1.compute_gradient_matrix(np.array([[0.5]]))[0, 0]:.4f}")
print(f"Weight=0.0: loss={l1.compute_loss_matrix(np.array([[0.0]])):.4f}, grad={l1.compute_gradient_matrix(np.array([[0.0]]))[0, 0]:.4f}")
print(f"Weight=-0.5: loss={l1.compute_loss_matrix(np.array([[-0.5]])):.4f}, grad={l1.compute_gradient_matrix(np.array([[-0.5]]))[0, 0]:.4f}")

Example 3: L2 Effect on Weight Decay

import ml_core as ml
import numpy as np

# Create L2 regularizer
l2 = ml.L2Regularizer(strength=0.1)

# Test update simulation
weights = np.array([[1.0]])
gradient = l2.compute_gradient_matrix(weights)
print(f"Weight=1.0, L2 gradient: {gradient[0, 0]:.4f}")

# Simulate weight decay with learning rate
learning_rate = 0.1
new_weights = weights - learning_rate * gradient
print(f"After update: {new_weights[0, 0]:.4f}")

Example 4: Using Regularizer in Neural Network

import ml_core as ml
import numpy as np

# Create neural network
nn = ml.NeuralNetwork([10, 32, 16, 1], "relu", "sigmoid")
nn.build(10, 1)

# Set regularizer via type and strength
nn.set_regularizer(ml.RegularizerType.L2, strength=0.001)

# Or via name string
# nn.set_regularizer("l1", strength=0.01)

# Or create regularizer directly
# l2_reg = ml.L2Regularizer(strength=0.001)
# nn.set_regularizer(l2_reg.get_type(), l2_reg.get_strength())

# Train
X = np.random.randn(100, 10)
y = (np.random.randn(100) > 0).astype(float)
nn.fit(X, y)

Example 5: Using RegularizerFactory

import ml_core as ml

# List available regularizers
available = ml.RegularizerFactory.list_regularizers()
print("Available regularizers:", available)

# Create via type enum
l2 = ml.RegularizerFactory.create_by_type(
    ml.RegularizerType.L2,
    strength=0.001
)

# Create via name string
l1 = ml.RegularizerFactory.create_by_name(
    "l1",
    strength=0.01
)

# Create Elastic Net with custom l1_ratio
elastic = ml.RegularizerFactory.create_by_name(
    "elastic_net",
    strength=0.01,
    params={"l1_ratio": 0.7}
)

print(f"Created: {l1.get_type_str()}, {l2.get_type_str()}, {elastic.get_type_str()}")

Example 6: Comparing Different Regularization Strengths

import ml_core as ml
import numpy as np

def simulate_training(regularizer, steps=100):
    """Simulate training with regularization"""
    weights = np.random.randn(5, 5) * 0.5
    loss_history = []
    lr = 0.01
    
    for _ in range(steps):
        # Data loss gradient (simulated)
        data_grad = weights * 0.1 + np.random.randn(5, 5) * 0.01
        
        # Regularization gradient
        reg_grad = regularizer.compute_gradient_matrix(weights)
        
        # Total gradient
        total_grad = data_grad + reg_grad
        
        # Update weights
        weights -= lr * total_grad
        
        # Compute total loss
        data_loss = np.linalg.norm(weights)
        reg_loss = regularizer.compute_loss_matrix(weights)
        total_loss = data_loss + reg_loss
        loss_history.append(total_loss)
    
    return loss_history

# Compare different strengths
strengths = [0.0, 0.001, 0.01, 0.1]
for strength in strengths:
    reg = ml.L2Regularizer(strength=strength)
    history = simulate_training(reg)
    print(f"Strength={strength}: final loss={history[-1]:.4f}")

Example 7: L1 vs L2 on Sparse Weights

import ml_core as ml
import numpy as np

# Create regularizers
l1 = ml.L1Regularizer(strength=0.1)
l2 = ml.L2Regularizer(strength=0.1)

# Test on sparse weight matrix
weights = np.zeros((5, 5))
weights[2, 2] = 1.0
weights[3, 3] = -1.0

l1_loss = l1.compute_loss_matrix(weights)
l2_loss = l2.compute_loss_matrix(weights)

print("Effect on sparse weights:")
print(f"L1 loss: {l1_loss:.4f} (encourages sparsity)")
print(f"L2 loss: {l2_loss:.4f} (penalizes magnitude)")

# Test on dense weights
weights = np.random.randn(5, 5) * 0.1
l1_loss_dense = l1.compute_loss_matrix(weights)
l2_loss_dense = l2.compute_loss_matrix(weights)

print(f"\nEffect on dense weights:")
print(f"L1 loss: {l1_loss_dense:.4f}")
print(f"L2 loss: {l2_loss_dense:.4f}")

Important Notes

    Strength Parameter: Higher strength = stronger regularization.

    L1 vs L2:

        L1: Encourages sparsity (many weights become zero)

        L2: Encourages small weights (weight decay)

        Elastic Net: Combines both

    Bias Regularization: Bias terms are typically not regularized (or regularized with lower strength).

    Gradient Sign: L1 gradient is discontinuous at 0 (subgradient).

    Hyperparameter Tuning: Regularization strength should be tuned via cross-validation.

    None Regularizer: Use RegularizerType.NONE or strength=0 for no regularization.

Choosing the Right Regularizer
Regularizer	Best For	Strengths	Weaknesses
L1 (Lasso)	Feature selection, sparse models	Creates sparsity, interpretability	Unstable with correlated features
L2 (Ridge)	General purpose	Stable, smooth optimization	Doesn't create sparsity
Elastic Net	Correlated features	Combines L1 and L2	More hyperparameters
Hyperparameter Guidelines
Regularizer	Strength Range	Notes
L1	1e-5 to 1e-1	Higher = sparser
L2	1e-5 to 1e-1	Higher = smaller weights
Elastic Net	1e-5 to 1e-1	l1_ratio ∈ [0, 1]
Dependencies

    bindings/core/enums.h - for RegularizerType

    components/regularizers/*.h - C++ regularizer headers

    Eigen3 - for matrix operations

Testing

To test the regularizers binding:

cd build
python3 -c "
import ml_core as ml
import numpy as np

# Test L1
l1 = ml.L1Regularizer(0.01)
w = np.random.randn(10, 10)
loss = l1.compute_loss_matrix(w)
grad = l1.compute_gradient_matrix(w)
print(f'L1 OK: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}')

# Test L2
l2 = ml.L2Regularizer(0.01)
loss = l2.compute_loss_matrix(w)
grad = l2.compute_gradient_matrix(w)
print(f'L2 OK: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}')

# Test Elastic Net
elastic = ml.ElasticNetRegularizer(0.01, 0.5)
loss = elastic.compute_loss_matrix(w)
grad = elastic.compute_gradient_matrix(w)
print(f'Elastic Net OK: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}')

print('All tests passed!')
"

Version Information
Regularizer	Version	Notes
L1Regularizer	1	-
L2Regularizer	1	-
ElasticNetRegularizer	1	With configurable l1_ratio
References

    C++ Regularizer Documentation https://../cpp/components/regularizers/

    Neural Network Binding https://../models/neural_network.md

    Loss Functions Binding https://../loss/README.md

    Optimizers Binding https://../optimizers/README.md


## Riepilogo dei file del binding regularizers

pybinding/bindings/regularizers/
├── README.md # ✅ Documentazione completa
├── regularizer_base.h/cpp # ✅ Classe base
├── l1_regularizer.h/cpp # ✅ L1
├── l2_regularizer.h/cpp # ✅ L2
├── elastic_net_regularizer.h/cpp # ✅ Elastic Net
└── regularizer_factory.h/cpp # ✅ Factory

