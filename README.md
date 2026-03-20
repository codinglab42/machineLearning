# ML Library - Machine Learning in C++

[![Build Status](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml/badge.svg)](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++14](https://img.shields.io/badge/C++-14-blue.svg)](https://en.cppreference.com/w/cpp/14)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

A modern, high-performance machine learning library written in C++ with Python bindings.

---

## ✨ Features

* **Models**:
    * **Linear Regression** (GD, Normal Equation, SVD)
    * **Logistic Regression** (with L2 regularization)
    * **Neural Networks** (fully connected layers, dropout, batch norm)
* **Optimizers**: SGD, Momentum, Adam
* **Regularizers**: L1, L2, Elastic Net
* **Layers**: Dense, Conv2D, Pooling, Flatten, Dropout, BatchNorm, RNN, LSTM, GRU
* **Python bindings** via `pybind11`
* **High performance** powered by **Eigen3**

---

## 📊 Performance

| Model | Dataset | Time | Accuracy/R² |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 100k samples, 10 features | 335 ms | 0.9988 |
| **Logistic Regression** | 10k samples, 10 features | 289 ms | 0.955 |
| **Neural Network** | 10k samples, 10 features | 1.06 s | 1.00 |

---

## 🚀 Installation

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake python3-dev

# pyenv for Python (optional)
curl [https://pyenv.run](https://pyenv.run) | bash