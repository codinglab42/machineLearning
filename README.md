# ML Library - Machine Learning in C++

[![Build Status](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml/badge.svg)](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

A modern, high-performance machine learning library written in C++ with Python bindings.

---

## ✨ Features

### Models
- **Linear Regression** (GD, Normal Equation, SVD)
- **Logistic Regression** (with L2 regularization)
- **Neural Networks** (fully connected, CNN, RNN)

### Neural Network Layers
| Layer Type | Description |
|------------|-------------|
| **Dense** | Fully connected layer with configurable activation |
| **Conv2D** | 2D convolutional layer |
| **Pooling** | Max/Average pooling |
| **Flatten** | Flattens input for dense layers |
| **Dropout** | Dropout regularization |
| **BatchNorm** | Batch normalization |
| **SimpleRNN** | Vanilla RNN layer |
| **LSTM** | Long Short-Term Memory |
| **GRU** | Gated Recurrent Unit |

### Optimizers
- SGD
- Momentum (with Nesterov)
- Adam (β₁=0.9, β₂=0.999, ε=1e-8)

### Regularizers
- L1 (Lasso)
- L2 (Ridge)
- Elastic Net

### Loss Functions
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Binary Cross Entropy
- Categorical Cross Entropy
- Huber Loss

### Activation Functions
- ReLU
- Leaky ReLU
- Sigmoid
- Tanh
- Softmax
- Linear

---

## 📊 Performance

| Model | Dataset | Time | Accuracy/R² |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 100k samples, 10 features | ~335 ms | 0.9988 |
| **Logistic Regression** | 10k samples, 10 features | ~289 ms | 0.955 |
| **Neural Network (Dense)** | 10k samples, 10 features | ~1.06 s | 0.98 |
| **LSTM** | Sequence 100, 64 units | ~2.5 s | - |

---

## 🚀 Installation

### Prerequisites

**Ubuntu/Debian**
```bash
sudo apt-get install build-essential cmake python3-dev libeigen3-dev
```

**pyenv for Python (optional)**
```bash
   curl [https://pyenv.run](https://pyenv.run) | bash
```
   
### Build
```
   git clone [https://github.com/codinglab42/machineLearning.git](https://github.com/codinglab42/machineLearning.git)
   cd machineLearning
   ./build.sh
```

### Python Installation (with pyenv)
```
   pyenv install 3.11.14
   pyenv virtualenv 3.11.14 ai-devel
   pyenv activate ai-devel
   pip install pytest numpy
   ./build.sh
```

### Using the Python Package
```bash
   # Install directly from source
   pip install -e .

   # Or build wheel
   python setup.py bdist_wheel
   pip install dist/ml_library-3.0.0-*.whl
```

## 📖 Examples
### Linear Regression in C++
```c++

#include "models/linear_regression.h"
#include <Eigen/Dense>

// Create and train model
models::LinearRegression model(0.01, 1000);
model.fit(X, y);

// Predict
Eigen::VectorXd y_pred = model.predict(X_test);
double r2 = model.score(X_test, y_test);
```

### Neural Network in C++
```c++

#include "models/neural_network.h"

models::NeuralNetwork nn;

// Add layers
nn.add_dense_layer(128, "relu");
nn.add_batch_norm_layer();
nn.add_dropout_layer(0.3);
nn.add_dense_layer(64, "relu");
nn.add_dense_layer(10, "softmax");

// Configure training
nn.set_optimizer(models::OptimizerType::ADAM, 0.001);
nn.set_loss_function("categorical_crossentropy");
nn.set_verbose(true);

// Train
nn.fit(X_train, y_train, epochs=50, batch_size=32);

// Predict
auto predictions = nn.predict(X_test);
```

### Neural Network in Python (LSTM Example)
```python

import machine_learning_module as ml
import numpy as np

# Create LSTM network for sequence prediction
nn = ml.NeuralNetwork()
nn.add_recurrent_layer(ml.RecurrentType.LSTM, 64, activation="tanh")
nn.add_dense_layer(1, "linear")

nn.set_optimizer(ml.OptimizerType.ADAM, 0.001)
nn.set_loss_function("mse")

# Train on sequence data
X = np.random.randn(1000, 10, 1)  # (samples, timesteps, features)
y = np.random.randn(1000, 1)

nn.fit(X, y, epochs=100, batch_size=32)

# Predict
predictions = nn.predict(X)
```

### Using Scaler
```python

import machine_learning_module as ml

# Create scaler
scaler = ml.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Or with MinMax scaling
scaler = ml.MinMaxScaler(feature_range_min=0, feature_range_max=1)
X_scaled = scaler.fit_transform(X)

# Transform back
X_original = scaler.inverse_transform(X_scaled)
```

### Save/Load Model
```python

# Save model
nn.save("my_model.bin")

# Load model
loaded_nn = ml.NeuralNetwork()
loaded_nn.load("my_model.bin")
```

## 📁 Project Structure
```text
machineLearning
├── include               # Header files
│   ├── components/       # Core components (layers, activation, loss...)
│   │   ├── layers/       # Layer implementations
│   │   ├── cache/        # Forward pass cache
│   │   ├── activation/   # Activation functions
│   │   ├── loss/         # Loss functions
│   │   ├── optimizers/   # Optimizers
│   │   └── regularizers/ # Regularizers
│   ├── models/           # Model implementations
│   ├── utils/            # Utility classes
│   └── exceptions/       # Exception handling
├── src                   # Source files
├── pybinding             # Python Binding
├── tests
│   ├── cpp               # Unit and integration tests
│   └── python            # Python binding tests
├── examples/             # Example programs
├── build.sh              # Build script
├── setup.py              # Python package setup
└── CMakeLists.txt        # CMake configuration
```

##🧪 Running Tests
###C++ Tests
```bash
cd build
ctest --output-on-failure
```

###Python Tests
```bash
pytest tests/python/
```

## 🤝 Contributing
1. Fork the project
2. Create a branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

##Coding Standards

C++17 standard
Use Eigen::MatrixXd for matrices
Exceptions for error handling (see exceptions/)
Document public APIs with Doxygen comments

## 📝 License
Distributed under the MIT License. See LICENSE for more information.

## 📧 Contact
**Maurizio Penna** - mauriziopenna@gmail.com  
**Project Link**: https://github.com/codinglab42/machineLearning

##🙏 Acknowledgements

Eigen3 - Linear algebra library
pybind11 - Python bindings
Andrew Ng - Machine Learning course inspiration

## Riepilogo delle modifiche principali

| Sezione | Modifica |
|---------|----------|
| **C++ standard** | C++14 → C++17 |
| **Python version** | 3.11+ → 3.7+ (più flessibile) |
| **Features** | Aggiunti LSTM, GRU, BatchNorm, regolarizzatori |
| **Esempi** | Aggiunti esempi LSTM e scaler |
| **Project structure** | Aggiornata con la nuova struttura |
| **setup.py** | Aggiunta menzione per installazione Python |
| **Acknowledgements** | Aggiunto Andrew Ng (fonte d'ispirazione) |