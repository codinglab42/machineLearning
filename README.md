# ML Library - Machine Learning in C++

[![Build Status](https://github.com/tuo-username/machineLearning/actions/workflows/build.yml/badge.svg)](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++14](https://img.shields.io/badge/C++-14-blue.svg)](https://en.cppreference.com/w/cpp/14)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

Una libreria di machine learning moderna e performante scritta in C++ con binding Python.

## ✨ Caratteristiche

- **Modelli**:
  - Linear Regression (GD, Normal Equation, SVD)
  - Logistic Regression (con regolarizzazione L2)
  - Neural Network (layer fully connected, dropout, batch norm)

- **Ottimizzatori**: SGD, Momentum, Adam
- **Regolarizzatori**: L1, L2, Elastic Net
- **Layer**: Dense, Conv2D, Pooling, Flatten, Dropout, BatchNorm, RNN, LSTM, GRU
- **Python bindings** via pybind11
- **Alta performance** con Eigen3

## 📊 Performance

| Modello             | Dataset                   | Tempo  | Accuracy/R² |
|---------------------|---------------------------|--------|-------------|
| Linear Regression   | 100k campioni, 10 feature | 335 ms | 0.9988      |
| Logistic Regression | 10k campioni, 10 feature  | 289 ms | 0.955       |
| Neural Network      | 10k campioni, 10 feature  | 1.06 s | 1.00        |

## 🚀 Installazione

### Prerequisiti
```bash
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


📖 Esempi

Linear Regression in C++

#include "models/linear_regression.h"

// Crea e allena modello
LinearRegression model(0.01, 1000);
model.fit(X, y);

// Predici
Eigen::VectorXd y_pred = model.predict(X_test);
double r2 = model.score(X_test, y_test);

Logistic Regression in Python

import machine_learning_module as ml
import numpy as np

# Crea modello
model = ml.LogisticRegression(0.1, 1000, 0.001)
model.fit(X_train, y_train)

# Predici
y_pred = model.predict_class(X_test)
accuracy = model.score(X_test, y_test)

Neural Network in Python

import machine_learning_module as ml

# Crea rete
network = ml.NeuralNetwork([2, 8, 1], "relu", "sigmoid")
network.set_loss_function("binary_crossentropy")
network.set_epochs(500)
network.fit(X, y)

# Predici
y_pred = network.predict(X)


🧪 Test

cd build
ctest --output-on-failure


📈 Benchmark

cd build
./bin/benchmark_linear
./bin/benchmark_logistic
./bin/benchmark_neural


📁 Struttura

machineLearning/
├── include/          # Header files
├── src/              # Source files
├── tests/
│   ├── cpp/          # Unit e integration tests
│   └── python/       # Python binding tests
├── build.sh          # Script di build
└── CMakeLists.txt    # CMake configuration


🤝 Contribuire

1. Fork    il progetto
2. Crea    un branch (git checkout -b feature/AmazingFeature)
3. Commit  (git commit -m 'Add AmazingFeature')
4. Push    (git push origin feature/AmazingFeature)
5. Apri    una Pull Request


📝 License
Distribuito sotto licenza MIT. Vedi LICENSE per maggiori informazioni.

📧 Contatti
Maurizio - mauriziopenna@gmail.com

Link progetto: https://github.com/codinglab42/machineLearning

