# Models Binding

## Overview

This module exposes all machine learning models available in the ML library as Python objects. Models provide high-level APIs for training, prediction, and evaluation.

## File Structure

bindings/models/
├── README.md # This file
├── estimator.h/cpp # Base Estimator class
├── linear_regression.h/cpp # LinearRegression model
├── logistic_regression.h/cpp # LogisticRegression model
└── neural_network.h/cpp # NeuralNetwork model


## Available Models

### 1. Estimator (Base Class)

Base class for all models providing common functionality.

**Methods:**
```python
fit(X: MatrixXd, y: VectorXd) -> None
predict(X: MatrixXd) -> VectorXd
score(X: MatrixXd, y: VectorXd) -> float
save(filename: str) -> None
load(filename: str) -> None
to_string() -> str
get_model_type() -> str
set_learning_rate(lr: float) -> None
get_learning_rate() -> float
set_optimizer(type: OptimizerType, learning_rate: float = 0.01) -> None

2. LinearRegression

Linear regression model with multiple solvers.

Constructors:
LinearRegression()
LinearRegression(learning_rate: float = 0.01, max_iter: int = 1000,
                 lambda: float = 0.0,
                 solver: LinearSolver = LinearSolver.GRADIENT_DESCENT)

Parameters:

    learning_rate: Learning rate for gradient descent

    max_iter: Maximum number of iterations

    lambda: L2 regularization strength (Ridge)

    solver: Solver type (GRADIENT_DESCENT, NORMAL_EQUATION, SVD)

Main Methods:

fit(X: MatrixXd, y: VectorXd) -> None
predict(X: MatrixXd) -> VectorXd
predict_single(x: VectorXd) -> float
score(X: MatrixXd, y: VectorXd) -> float  # R² score
mse(X: MatrixXd, y: VectorXd) -> float
mae(X: MatrixXd, y: VectorXd) -> float
r2_score(X: MatrixXd, y: VectorXd) -> float
save(filename: str) -> None
load(filename: str) -> None
to_string() -> str

Static Methods:

cross_val_score(X: MatrixXd, y: VectorXd, cv: int = 5,
                solver: LinearSolver = LinearSolver.GRADIENT_DESCENT) -> VectorXd

Properties:

@coefficients  # Model coefficients (theta)
@intercept     # Model intercept
@cost_history  # History of cost values during training

3. LogisticRegression

Logistic regression for binary classification.

Constructors:

LogisticRegression()
LogisticRegression(learning_rate: float = 0.1, max_iter: int = 1000,
                   lambda: float = 0.0, tolerance: float = 1e-4,
                   verbose: bool = False)

Parameters:

    learning_rate: Learning rate for gradient descent

    max_iter: Maximum number of iterations

    lambda: L2 regularization strength

    tolerance: Convergence tolerance

    verbose: Whether to print progress

Main Methods:

fit(X: MatrixXd, y: VectorXd) -> None
predict(X: MatrixXd) -> VectorXd  # Probability predictions
predict_class(X: MatrixXd, threshold: float = 0.5) -> VectorXi
score(X: MatrixXd, y: VectorXd) -> float  # Accuracy
precision_recall_f1(X: MatrixXd, y: VectorXd, threshold: float = 0.5) -> Vector3d
confusion_matrix(X: MatrixXd, y: VectorXd, threshold: float = 0.5) -> MatrixXd
save(filename: str) -> None
load(filename: str) -> None
to_string() -> str

Properties:

@coefficients     # Model coefficients
@intercept        # Model intercept
@cost_history     # History of cost values
@accuracy_history # History of accuracy values

4. NeuralNetwork

Flexible neural network with multiple layer types.

Constructors:

NeuralNetwork()
NeuralNetwork(layer_sizes: List[int], activation: str = "relu",
              output_activation: str = "sigmoid",
              optimizer_type: OptimizerType = OptimizerType.ADAM,
              learning_rate: float = 0.001)

Layer Building Methods:

add_dense_layer(units: int, activation: str = "relu", use_bias: bool = True)
add_conv2d_layer(filters: int, kernel_size: int, strides: int = 1,
                 padding: str = "valid", activation: str = "relu")
add_pooling_layer(pool_size: int = 2, strides: int = 2, pool_type: str = "max")
add_flatten_layer()
add_dropout_layer(rate: float)
add_batch_norm_layer()
add_recurrent_layer(type: RecurrentType, units: int,
                    return_sequences: bool = False, activation: str = "tanh")

Training Methods:

build(n_features: int, n_classes: int) -> None
fit(X: MatrixXd, y: VectorXd) -> None
fit(X: MatrixXd, y: MatrixXd) -> None
fit_advanced(X: MatrixXd, y: VectorXd, epochs: int, batch_size: int, verbose: bool = False)
fit_advanced_matrix(X: MatrixXd, y: MatrixXd, epochs: int, batch_size: int, verbose: bool = False)

Prediction Methods:

predict(X: MatrixXd) -> VectorXd
predict_proba(X: MatrixXd) -> MatrixXd
score(X: MatrixXd, y: VectorXd) -> float

Network Information:

summary() -> None
get_summary() -> str
get_training_history() -> Tuple[List[float], List[float], List[float]]
get_num_layers() -> int
get_num_parameters() -> int
get_input_size() -> int
get_output_size() -> int
is_fitted() -> bool

Configuration Methods:

set_batch_size(batch_size: int) -> None
set_epochs(epochs: int) -> None
set_validation_split(split: float) -> None
set_verbose(verbose: bool) -> None
set_loss_function(loss: str) -> None  # "mse", "mae", "binary_crossentropy", etc.
set_regularizer(type: RegularizerType, strength: float = 0.01,
                params: Dict[str, float] = {}) -> None
set_optimizer(type: OptimizerType, learning_rate: float = 0.01) -> None

Layer Access Methods:

get_layer(idx: int) -> Layer
get_weights_for_layer(idx: int) -> MatrixXd
set_weights_for_layer(idx: int, weights: MatrixXd) -> None
get_layer_type(idx: int) -> str
get_layer_config(idx: int) -> str

Serialization:

save(filename: str) -> None
load(filename: str) -> None
to_string() -> str
reset() -> None
reset_history() -> None

Properties:

@num_layers        # Number of layers
@num_parameters    # Total parameters
@loss_history      # Training loss history
@val_loss_history  # Validation loss history
@accuracy_history  # Accuracy history
@input_size        # Input dimension
@output_size       # Output dimension
@is_fitted         # Whether the model is trained

Usage Examples
Example 1: Linear Regression

import machine_learning_module as ml
import numpy as np

# Generate synthetic data
X = np.random.randn(100, 3)
y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(100) * 0.1

# Create and train model
model = ml.LinearRegression(learning_rate=0.01, max_iter=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² score: {model.score(X, y)}")
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

# Cross-validation
scores = ml.LinearRegression.cross_val_score(X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {np.mean(scores)}")

# Save and load
model.save("linear_model.bin")
loaded_model = ml.LinearRegression()
loaded_model.load("linear_model.bin")

Example 2: Logistic Regression

import machine_learning_module as ml
import numpy as np

# Generate binary classification data
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(float)

# Create and train model
model = ml.LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X, y)

# Make predictions
probabilities = model.predict(X)
predictions = model.predict_class(X)
accuracy = model.score(X, y)

print(f"Accuracy: {accuracy}")
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")

# Get metrics
precision, recall, f1 = model.precision_recall_f1(X, y)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Confusion matrix
conf_matrix = model.confusion_matrix(X, y)
print(f"Confusion Matrix:\n{conf_matrix}")

Example 3: Neural Network for Classification

import machine_learning_module as ml
import numpy as np

# Generate synthetic data
X = np.random.randn(500, 10)
y = (np.sum(X[:, :3], axis=1) > 0).astype(float)

# Create neural network
nn = ml.NeuralNetwork()
nn.add_dense_layer(64, activation="relu")
nn.add_dropout_layer(0.2)
nn.add_dense_layer(32, activation="relu")
nn.add_dense_layer(1, activation="sigmoid")

# Configure training
nn.set_loss_function("binary_crossentropy")
nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
nn.set_epochs(100)
nn.set_batch_size(32)
nn.set_verbose(True)

# Build and train
nn.build(n_features=10, n_classes=1)
nn.fit(X, y)

# Make predictions
predictions = nn.predict(X)
score = nn.score(X, y)
print(f"Accuracy: {score:.3f}")

# Get training history
loss_history, val_loss_history, acc_history = nn.get_training_history()

# Save and load
nn.save("model.bin")
loaded_nn = ml.NeuralNetwork()
loaded_nn.load("model.bin")

Example 4: Neural Network for Regression

import machine_learning_module as ml
import numpy as np

# Generate regression data
X = np.random.randn(500, 5)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(500) * 0.1

# Create neural network for regression
nn = ml.NeuralNetwork()
nn.add_dense_layer(128, activation="relu")
nn.add_batch_norm_layer()
nn.add_dense_layer(64, activation="relu")
nn.add_dense_layer(1, activation="linear")  # linear for regression

# Configure
nn.set_loss_function("mse")
nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
nn.set_epochs(150)
nn.set_batch_size(32)

# Build and train
nn.build(n_features=5, n_classes=1)
nn.fit(X, y)

# Make predictions
predictions = nn.predict(X)
r2_score = nn.score(X, y)  # R² for regression
print(f"R² score: {r2_score:.3f}")

# Save
nn.save("regression_model.bin")

Example 5: Neural Network with Layer Access

import machine_learning_module as ml
import numpy as np

# Create network
nn = ml.NeuralNetwork([10, 32, 16, 1], "relu", "sigmoid")
nn.build(10, 1)

# Access individual layers
num_layers = nn.num_layers
print(f"Number of layers: {num_layers}")

for i in range(num_layers):
    layer_type = nn.get_layer_type(i)
    config = nn.get_layer_config(i)
    weights = nn.get_weights_for_layer(i)
    print(f"Layer {i}: {layer_type}, config: {config}, weights shape: {weights.shape}")

# Modify weights of first layer
new_weights = np.random.randn(32, 11) * 0.1  # input_size=10, units=32, +1 for bias
nn.set_weights_for_layer(0, new_weights)

# Get a specific layer
first_layer = nn.get_layer(0)
print(f"First layer type: {first_layer.get_type()}")
print(f"First layer config: {first_layer.get_config()}")

Example 6: Using NeuralNetwork with Matrix y (Multi-class)

import machine_learning_module as ml
import numpy as np

# Multi-class classification (3 classes)
X = np.random.randn(300, 8)
y = np.zeros((300, 3))
for i in range(300):
    cls = int(np.sum(X[i, :3]) % 3)
    y[i, cls] = 1.0

# Create and train
nn = ml.NeuralNetwork()
nn.add_dense_layer(64, activation="relu")
nn.add_dense_layer(32, activation="relu")
nn.add_dense_layer(3, activation="softmax")  # 3 classes

nn.set_loss_function("categorical_crossentropy")
nn.set_optimizer(ml.OptimizerType.ADAM, learning_rate=0.001)
nn.set_epochs(100)
nn.set_batch_size(32)

nn.build(n_features=8, n_classes=3)
nn.fit_matrix(X, y)  # Matrix version for multi-class

# Make predictions
probabilities = nn.predict_proba(X)
predictions = np.argmax(probabilities, axis=1)
print(f"Predictions shape: {predictions.shape}")

Important Notes

    Build Before Training: Always call build(n_features, n_classes) before training a NeuralNetwork.

    Input Normalization: For best results, normalize your input features before training.

    Loss Functions: Choose the appropriate loss function:

        "mse" / "mae" for regression

        "binary_crossentropy" for binary classification

        "categorical_crossentropy" for multi-class classification

    Output Activation: Match activation to task:

        "linear" for regression

        "sigmoid" for binary classification

        "softmax" for multi-class classification

    Training Verbosity: set_verbose(True) to monitor training progress.

    Layer Access: Use layer access methods for fine-grained control and debugging.

    Serialization: Models are saved in binary format. Ensure compatibility between versions.

    Memory Management: Large models may require significant memory. Use clear_cache() when needed.

Dependencies

    bindings/core/enums.h - for OptimizerType, RegularizerType

    bindings/core/exceptions.h - for exception translator

    bindings/layers/ - for layer bindings

    components/models/*.h - C++ model headers

    Eigen3 - for matrix operations

Testing

To test the models binding:

cd build
python3 -c "
import machine_learning_module as ml
import numpy as np

# Test Linear Regression
X = np.random.randn(100, 3)
y = np.random.randn(100)
model = ml.LinearRegression()
model.fit(X, y)
print('LinearRegression OK!')

# Test Neural Network
nn = ml.NeuralNetwork([5, 10, 1], 'relu', 'sigmoid')
nn.build(5, 1)
print('NeuralNetwork OK!')

print('All tests passed!')
"

Version Information
Model	Version	Notes
Estimator	1	Base class
LinearRegression	1	Supports 3 solvers
LogisticRegression	1	With L2 regularization
NeuralNetwork	2	Full layer support
Troubleshooting
Issue	Solution
NotFittedException	Call fit() before predict() or score()
Dimension mismatch	Check input shape matches n_features
NaN/Inf in loss	Reduce learning rate, normalize input
Slow training	Increase batch size, reduce network size
Overfitting	Add dropout, L2 regularization, reduce epochs
References

    C++ Models Documentation https://../cpp/models/

    Layer Binding https://../layers/README.md

    Loss Functions Binding https://../loss/README.md

    Optimizers Binding https://../optimizers/README.md

    Regularizers Binding https://../regularizers/README.md


