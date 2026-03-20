
# ML Library Documentation
### Detailed Documentation

## Table of Contents
1. [Installation](#installation)
2. [Models](#models)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [Neural Network](#neural-network)
3. [Optimizers](#optimizers)
4. [Regularizers](#regularizers)
5. [Layers](#layers)
6. [Python Bindings](#python-bindings)
7. [Examples](#examples)
8. [API Reference](#api-reference)

## Installation

### C++ (static library)
```bash
./build.sh
# Library will be in build/lib/
```
Python (via pyenv)
```bash
pyenv activate ai-devel
pip install -r tests/python/requirements.txt
```
### Models
**Linear Regression**
```cpp
// C++ example
LinearRegression model(learning_rate=0.01, max_iter=1000, lambda=0.0, solver=GRADIENT_DESCENT);
model.fit(X, y);
Eigen::VectorXd y_pred = model.predict(X_test);

// Additional methods
double r2 = model.score(X_test, y_test);
double mse = model.mse(X_test, y_test);
double mae = model.mae(X_test, y_test);
Eigen::VectorXd coef = model.coefficients();
double intercept = model.intercept();
Parameters:

learning_rate: Step size for gradient descent (default: 0.01)

max_iter: Maximum number of iterations (default: 1000)

lambda: L2 regularization strength (default: 0.0)

solver: Solver type (GRADIENT_DESCENT, NORMAL_EQUATION, SVD)
```
**Logistic Regression**
```cpp
LogisticRegression model(learning_rate=0.1, max_iter=1000, lambda=0.001);
model.fit(X, y);
Eigen::VectorXd y_proba = model.predict(X_test);
Eigen::VectorXi y_pred = model.predict_class(X_test);
double accuracy = model.score(X_test, y_test);
Eigen::MatrixXd cm = model.confusion_matrix(X_test, y_test);
Parameters:

learning_rate: Step size (default: 0.1)

max_iter: Maximum iterations (default: 1000)

lambda: L2 regularization strength (default: 0.001)

tolerance: Convergence tolerance (default: 1e-4)
```
**Neural Network**
```cpp
// Create network
NeuralNetwork network({input_size, hidden1, hidden2, output_size}, 
                      "relu", "softmax", OptimizerType::ADAM, 0.01);
network.set_loss_function("categorical_crossentropy");
network.set_epochs(100);
network.set_batch_size(32);
network.set_validation_split(0.2);

// Training
network.fit(X_train, y_train);

// Prediction
Eigen::VectorXd y_pred = network.predict(X_test);
Eigen::MatrixXd y_proba = network.predict_proba(X_test);

// Training history
auto [loss, val_loss, acc] = network.get_training_history();
Parameters:

layer_sizes: Vector of layer sizes [input, hidden1, hidden2, ..., output]

activation: Hidden layer activation function ("relu", "tanh", "sigmoid")

output_activation: Output layer activation ("softmax", "sigmoid", "linear")

optimizer_type: Optimizer type (SGD, MOMENTUM, ADAM)

learning_rate: Learning rate (default: 0.01)
```
### Optimizers
| Type | Description | Parameters |  
| :--- | :--- | :--- |  
| **SGD** | Stochastic Gradient Descent | learning_rate, decay |  
| **Momentum** | SGD with momentum | learning_rate, momentum, decay, nesterov |  
| **Adam** |	Adaptive Moment Estimation | learning_rate, beta1, beta2, epsilon, decay |  

Usage in Python:

```python
# SGD
optimizer = ml.OptimizerFactory.create("sgd", 0.01)

# Momentum
params = {"momentum": 0.9}
optimizer = ml.OptimizerFactory.create("momentum", 0.01, params)

# Adam
params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
optimizer = ml.OptimizerFactory.create("adam", 0.001, params)
```
### Regularizers
Type	Description	Formula
L1	Lasso	λ * Σ|w|
L2	Ridge	λ/2 * Σw²
Elastic Net	L1 + L2 combination	λ * (α*L1 + (1-α)*L2)
Usage in Python:

```python
# L1
reg = ml.RegularizerFactory.create("l1", 0.01)

# L2
reg = ml.RegularizerFactory.create("l2", 0.01)

# Elastic Net
params = {"l1_ratio": 0.5}
reg = ml.RegularizerFactory.create("elastic_net", 0.01, params)
```
### Layers
**Dense Layer**
Fully connected layer with configurable activation.

```cpp
auto dense = std::make_unique<layers::DenseLayer>(units, "relu", true);
dense->set_input_shape(input_size);
```
**Convolutional Layer**
2D convolution for image processing.

```cpp
auto conv = std::make_unique<layers::Conv2DLayer>(filters, kernel_size, strides, "valid", "relu");
conv->set_input_shape(input_size);
```
**Pooling Layer**
Max or average pooling for downsampling.

```cpp
auto pool = std::make_unique<layers::PoolingLayer>(pool_size, stride, layers::PoolingLayer::MAX, channels);
```
**Recurrent Layers**
RNN, LSTM, and GRU for sequence data.

```cpp
auto rnn = std::make_unique<layers::SimpleRNNLayer>(units, input_size);
auto lstm = std::make_unique<layers::LSTMLayer>(units, input_size);
auto gru = std::make_unique<layers::GRULayer>(units, input_size);
```
### Python Bindings
All C++ classes are exposed to Python with a clean API:

```python
import machine_learning_module as ml
import numpy as np

# Linear Regression
model = ml.LinearRegression(0.01, 1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Logistic Regression
model = ml.LogisticRegression(0.1, 1000, 0.001)
y_pred = model.predict_class(X_test)
cm = model.confusion_matrix(X_test, y_test)

# Neural Network
network = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid")
network.set_loss_function("binary_crossentropy")
network.set_epochs(500)
network.fit(X, y)
```
## Examples
### Example 1: Iris Classification with Logistic Regression
```python
import machine_learning_module as ml
import numpy as np

# Load dataset (example)
X = np.loadtxt("iris_X.txt")
y = np.loadtxt("iris_y.txt")

# Split data
split = 0.8
split_idx = int(len(X) * split)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
model = ml.LogisticRegression(0.1, 1000, 0.001)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = model.confusion_matrix(X_test, y_test)
print("Confusion Matrix:")
print(cm)
```
### Example 2: XOR with Neural Network
```python
import machine_learning_module as ml
import numpy as np

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y = np.array([0, 1, 1, 0], dtype=np.float64)

# Create network
network = ml.NeuralNetwork([2, 4, 1], "relu", "sigmoid")
network.set_loss_function("binary_crossentropy")
network.set_epochs(1000)
network.set_batch_size(4)

# Train
network.fit(X, y)

# Test
y_pred = network.predict(X)
print("Predictions:", (y_pred > 0.5).astype(int).flatten())
print("True values:", y.astype(int))
```
### Example 3: Boston Housing with Linear Regression
```python
import machine_learning_module as ml
import numpy as np

# Load data
X = np.loadtxt("boston_X.txt")
y = np.loadtxt("boston_y.txt")

# Standardize features
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / std

# Train
model = ml.LinearRegression(0.01, 1000)
model.fit(X_scaled, y)

# Evaluate
r2 = model.score(X_scaled, y)
print(f"R² Score: {r2:.4f}")
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept}")
```
### API Reference
For complete API documentation, see the Doxygen-generated documentation.

### Performance Tips
Use Normal Equation for small datasets (< 10k samples) with linear regression

Use Gradient Descent for large datasets with appropriate learning rate

Standardize features for neural networks and logistic regression

Use Adam optimizer for most neural network tasks

Enable OpenMP for better performance on multi-core systems

### Troubleshooting
**Common Issues**
*Issue*: basic_string::_M_create error during serialization  
*Solution*: Ensure the model is properly trained before saving. Use the latest version.

*Issue*: Dimension mismatch in forward/backward  
*Solution*: Check that input shapes match the layer configuration.

*Issue*: Python module not found  
*Solution*: Ensure the build directory is in Python path. Use conftest.py as provided.
