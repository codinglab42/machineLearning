# ML Library - Machine Learning in C++

[![Build Status](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml/badge.svg)](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++14](https://img.shields.io/badge/C++-14-blue.svg)](https://en.cppreference.com/w/cpp/14)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

A modern, high-performance machine learning library written in C++ with Python bindings.


✨ Features

Models:
Linear Regression (GD, Normal Equation, SVD)
Logistic Regression (with L2 regularization)
Neural Networks (fully connected layers, dropout, batch norm)
Optimizers: SGD, Momentum, Adam
Regularizers: L1, L2, Elastic Net
Layers: Dense, Conv2D, Pooling, Flatten, Dropout, BatchNorm, RNN, LSTM, GRU
Python bindings via pybind11
High performance powered by Eigen3


📊 Performance

| Model               | Dataset                   | Time   | Accuracy/R² |
|---------------------|---------------------------|--------|-------------|
| Linear Regression   | 100k samples, 10 feature  | 335 ms | 0.9988      |
| Logistic Regression | 10k samples, 10 feature   | 289 ms | 0.955       |
| Neural Network      | 10k samples, 10 feature   | 1.06 s | 1.00        |


## 🚀 Installation

### Prerequisites

# Ubuntu/Debian
sudo apt-get install build-essential cmake python3-dev

# pyenv per Python (opzionale)
curl https://pyenv.run | bash


Build

git clone https://github.com/tuo-username/machineLearning.git
cd machineLearning
./build.sh


Python (con pyenv)

pyenv install 3.11.14
pyenv virtualenv 3.11.14 ai-devel
pyenv activate ai-devel
pip install pytest numpy
./build.sh


📖 Examples

Linear Regression in C++

#include "models/linear_regression.h"

// Create and train model
LinearRegression model(0.01, 1000);
model.fit(X, y);

// Predict
Eigen::VectorXd y_pred = model.predict(X_test);
double r2 = model.score(X_test, y_test);

Logistic Regression in Python

import machine_learning_module as ml
import numpy as np

# Create model
model = ml.LogisticRegression(0.1, 1000, 0.001)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict_class(X_test)
accuracy = model.score(X_test, y_test)

Neural Network in Python

import machine_learning_module as ml

# Create network
network = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid")
network.set_loss_function("binary_crossentropy")
network.set_epochs(500)
network.fit(X, y)

# Predict
y_pred = network.predict(X)

🧪 Test

cd build
ctest --output-on-failure


📈 Benchmark

cd build
./bin/benchmark_linear
./bin/benchmark_logistic
./bin/benchmark_neural


📁 Structure

machineLearning/
├── include/          # Header files
├── src/              # Source files
├── tests/
│   ├── cpp/          # Unit e integration tests
│   └── python/       # Python binding tests
├── build.sh          # Script di build
└── CMakeLists.txt    # CMake configuration


🤝 Contributing

1. Fork the project
2. Create a branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

📝 License
Distributed under the MIT License. See LICENSE for more information.

📧 Contact
Maurizio - mauriziopenna@gmail.com
Project Link: https://github.com/codinglab42/machineLearning

