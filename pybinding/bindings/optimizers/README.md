# Optimizers Binding

## Overview

This module exposes all optimizers available in the ML library as Python objects. Optimizers are used to update model weights during training based on computed gradients.

## File Structure

bindings/optimizers/
├── README.md # This file
├── optimizer_base.h/cpp # Base Optimizer class
├── sgd_optimizer.h/cpp # SGD Optimizer
├── momentum_optimizer.h/cpp # Momentum Optimizer
├── adam_optimizer.h/cpp # Adam Optimizer
└── optimizer_factory.h/cpp # OptimizerFactory


## Available Optimizers

### 1. SGDOptimizer

Stochastic Gradient Descent - Basic optimizer.

**Constructor:**
```python
SGDOptimizer(learning_rate: float = 0.01, decay: float = 0.0)

Parameters:

    learning_rate: Step size for updates

    decay: Learning rate decay factor per iteration

Update Rule: w = w - lr * grad
2. MomentumOptimizer

SGD with Momentum - Accelerates convergence.

Constructor:

MomentumOptimizer(learning_rate: float = 0.01, momentum: float = 0.9,
                  decay: float = 0.0, nesterov: bool = False)

Parameters:

    learning_rate: Step size for updates

    momentum: Momentum coefficient (typically 0.9)

    decay: Learning rate decay factor

    nesterov: Use Nesterov accelerated gradient

Update Rule: v = momentum * v - lr * grad
w = w + v
3. AdamOptimizer

Adaptive Moment Estimation - Most popular optimizer.

Constructor:

AdamOptimizer(learning_rate: float = 0.001, beta1: float = 0.9,
              beta2: float = 0.999, epsilon: float = 1e-8,
              decay: float = 0.0)

Parameters:

    learning_rate: Step size for updates (typically 0.001)

    beta1: Exponential decay rate for first moment (0.9)

    beta2: Exponential decay rate for second moment (0.999)

    epsilon: Small constant for numerical stability

    decay: Learning rate decay factor

Update Rule: Adaptive per-parameter learning rates.
OptimizerFactory

Factory for creating optimizers dynamically.

Methods:

OptimizerFactory.create_by_type(type: OptimizerType, learning_rate: float = 0.01,
                                params: Dict[str, float] = {}) -> Optimizer
OptimizerFactory.create_by_name(type_str: str, learning_rate: float = 0.01,
                                params: Dict[str, float] = {}) -> Optimizer
OptimizerFactory.string_to_type(type_str: str) -> OptimizerType
OptimizerFactory.type_to_string(type: OptimizerType) -> str
OptimizerFactory.list_optimizers() -> List[str]

Available Optimizer Names:

    "sgd" - SGD Optimizer

    "momentum" - Momentum Optimizer

    "adam" - Adam Optimizer

Usage Examples
Example 1: Creating and Using SGD

import ml_core as ml
import numpy as np

# Create optimizer
optimizer = ml.SGDOptimizer(learning_rate=0.01, decay=0.001)

# Simulate training
weights = np.random.randn(10, 10)
gradient = np.random.randn(10, 10)

# Update weights
optimizer.update_weights(weights, gradient)
print(f"Weights after update: shape {weights.shape}")

# Get learning rate (with decay)
current_lr = optimizer.get_current_learning_rate()
print(f"Current learning rate: {current_lr:.4f}")

Example 2: Using Momentum Optimizer

import ml_core as ml
import numpy as np

# Create momentum optimizer
optimizer = ml.MomentumOptimizer(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

# Update weights and bias
weights = np.random.randn(10, 10)
bias = np.random.randn(10)
grad_w = np.random.randn(10, 10)
grad_b = np.random.randn(10)

optimizer.update_weights(weights, grad_w)
optimizer.update_bias(bias, grad_b)

Example 3: Using Adam Optimizer

import ml_core as ml
import numpy as np

# Create Adam optimizer
optimizer = ml.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

# Update multiple times
weights = np.random.randn(20, 20)
for step in range(100):
    grad = np.random.randn(20, 20)
    optimizer.update_weights(weights, grad)
    
    if step % 10 == 0:
        lr = optimizer.get_current_learning_rate()
        print(f"Step {step}, LR: {lr:.6f}")

print(f"Final weights norm: {np.linalg.norm(weights):.4f}")

Example 4: Using OptimizerFactory

import ml_core as ml

# Create via type enum
adam = ml.OptimizerFactory.create_by_type(
    ml.OptimizerType.ADAM,
    learning_rate=0.001
)

# Create via name string
sgd = ml.OptimizerFactory.create_by_name(
    "sgd",
    learning_rate=0.01
)

# With additional parameters
momentum = ml.OptimizerFactory.create_by_name(
    "momentum",
    learning_rate=0.01,
    params={"momentum": 0.95, "nesterov": 1.0}
)

print(f"Created: {sgd.get_type_str()}, {adam.get_type_str()}")

Example 5: Optimizer in Neural Network

import ml_core as ml
import numpy as np

# Create neural network
nn = ml.NeuralNetwork([10, 32, 16, 1], "relu", "sigmoid")
nn.build(10, 1)

# Set optimizer via enum
nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)

# Or set via name string
# nn.set_optimizer(ml.OptimizerType.SGD, learning_rate=0.01)

# Train
X = np.random.randn(100, 10)
y = (np.random.randn(100) > 0).astype(float)
nn.fit(X, y)

Example 6: Custom Learning Rate Schedule

import ml_core as ml
import numpy as np

class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr, decay_rate):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def step(self, epoch):
        new_lr = self.initial_lr * (1.0 / (1.0 + self.decay_rate * epoch))
        self.optimizer.set_learning_rate(new_lr)

# Usage
optimizer = ml.AdamOptimizer(learning_rate=0.01)
scheduler = LearningRateScheduler(optimizer, 0.01, 0.001)

weights = np.random.randn(10, 10)
for epoch in range(100):
    scheduler.step(epoch)
    grad = np.random.randn(10, 10)
    optimizer.update_weights(weights, grad)
    
    if epoch % 10 == 0:
        current_lr = optimizer.get_learning_rate()
        print(f"Epoch {epoch}, LR: {current_lr:.6f}")

Example 7: Comparing Optimizers

import ml_core as ml
import numpy as np

def train_with_optimizer(optimizer, steps=100):
    """Simple training loop for comparison"""
    weights = np.random.randn(5, 5) * 0.1
    loss_history = []
    
    for step in range(steps):
        # Simulate gradient (minimize norm of weights)
        grad = weights * 0.1
        optimizer.update_weights(weights, grad)
        loss = np.linalg.norm(weights)
        loss_history.append(loss)
    
    return loss_history

# Compare different optimizers
optimizers = [
    ml.SGDOptimizer(learning_rate=0.01),
    ml.MomentumOptimizer(learning_rate=0.01, momentum=0.9),
    ml.AdamOptimizer(learning_rate=0.01)
]

names = ["SGD", "Momentum", "Adam"]

for name, opt in zip(names, optimizers):
    history = train_with_optimizer(opt)
    print(f"{name}: final loss = {history[-1]:.4f}, "
          f"improvement = {history[0] - history[-1]:.4f}")

Important Notes

    Iterations: Each update() call increments the iteration counter, used for learning rate decay.

    Learning Rate Decay: get_current_learning_rate() returns lr / (1 + decay * iterations).

    State Preservation: Momentum and Adam maintain internal state (velocity, moments). Use clone() to copy.

    Reset: Use reset() to clear optimizer state (useful for new training runs).

    Serialization: Optimizers support serialize() and deserialize() for saving/loading training state.

    Batch vs Individual: The same optimizer can be used for both weights (Matrix) and bias (Vector).

Choosing the Right Optimizer
Optimizer	Use Case	Pros	Cons
SGD	Simple models, sparse data	Simple, memory efficient	Slow convergence
Momentum	Deep networks	Faster than SGD, smooth updates	Additional memory for velocity
Adam	Most problems (default)	Fast convergence, adaptive LR	More memory, hyperparameters to tune
Hyperparameter Guidelines
Optimizer	Parameters	Typical Values
SGD	learning_rate	0.01, 0.001
Momentum	learning_rate, momentum	0.01, 0.9
Adam	learning_rate, beta1, beta2	0.001, 0.9, 0.999
Dependencies

    bindings/core/enums.h - for OptimizerType

    components/optimizers/*.h - C++ optimizer headers

    Eigen3 - for matrix operations

Testing

To test the optimizers binding:

cd build
python3 -c "
import ml_core as ml
import numpy as np

# Test SGD
sgd = ml.SGDOptimizer(0.01)
w = np.random.randn(5, 5)
g = np.random.randn(5, 5)
sgd.update_weights(w, g)
print('SGD OK!')

# Test Adam
adam = ml.AdamOptimizer(0.001)
adam.update_weights(w, g)
print('Adam OK!')

print('All tests passed!')
"

Version Information
Optimizer	Version	Notes
SGDOptimizer	1	With learning rate decay
MomentumOptimizer	1	With Nesterov option
AdamOptimizer	1	-
References

    C++ Optimizer Documentation https://../cpp/components/optimizers/

    Neural Network Binding https://../models/neural_network.md

    Loss Functions Binding https://../loss/README.md


## Riepilogo dei file del binding optimizers

bindings/optimizers/
├── README.md # ✅ Documentazione completa
├── optimizer_base.h/cpp # ✅ Classe base
├── sgd_optimizer.h/cpp # ✅ SGD
├── momentum_optimizer.h/cpp # ✅ Momentum
├── adam_optimizer.h/cpp # ✅ Adam
└── optimizer_factory.h/cpp # ✅ Factory
