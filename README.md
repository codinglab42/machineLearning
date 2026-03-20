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