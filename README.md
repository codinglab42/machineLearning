# ML Library - Machine Learning in C++

[![Build Status](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml/badge.svg)](https://github.com/codinglab42/machineLearning/actions/workflows/build.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python 3.11 only](https://img.shields.io/badge/Python-3.11%20only-blue.svg)](https://www.python.org/)

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
- RMSprop (ρ=0.9)
- Adagrad

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
| **Linear Regression** | 100k samples, 10 features | 1,252 ms | R² 0.9988 |
| **Logistic Regression** | 100k samples, 10 features | 8,588 ms | Accuracy 94.72% |
| **Neural Network (Dense)** | 10k samples, 10 features, 100 epochs | 1,305 ms | Accuracy 87.5% |

These are median training times from three runs on synthetic data, not
real-world performance claims. See the [benchmark summary](docs/performance_results/2026-11-25-updated/summary.md)
for the environment, test split, baselines, and limitations.

---

## 🚀 Installation

### Prerequisites

**Ubuntu/Debian**
```bash
sudo apt-get install build-essential cmake python3-dev libeigen3-dev
```

**pyenv for Python (optional)**
```bash
   curl https://pyenv.run | bash
```
   
### Build
```
   git clone https://github.com/codinglab42/machineLearning.git
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
   python -m pip install .

   # Or build a wheel, then install it
   python -m pip wheel . --no-deps --wheel-dir dist
   python -m pip install dist/ml_core-4.0.0-*.whl
```

### Upgrading from 3.0.0

Version 4.0.0 changes the C++ ABI and parts of the source API. Rebuild C++
clients and adapt code where needed. Some Python error types have changed,
for example invalid parameters now raise `ValueError` rather than `RuntimeError`.
Version 4 can read older neural-network saves, subject to the limitations
described below, but saves in the new format require the updated library.

---

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

models::NeuralNetwork nn({input_size, 128}, "relu", "relu");

// Add layers
nn.add_batch_norm_layer();
nn.add_dropout_layer(0.3);
nn.add_dense_layer(64, "relu");
nn.add_dense_layer(1, "linear");

// Configure training
nn.set_optimizer(models::OptimizerType::ADAM, 0.001);
nn.set_loss_function("mse");
nn.set_verbose(true);
nn.set_epochs(50);
nn.set_batch_size(32);

// Train
nn.fit(X_train, y_train);

// Predict
auto predictions = nn.predict(X_test);
```

For neural networks, optimizer state is maintained separately for each trainable
layer. `set_optimizer` resets that state; `set_learning_rate` updates the rate
without resetting it. `build`, `reset`, and changing the layer list also reset
optimizer state. A subsequent `fit` continues training an already fitted
network for both vector and matrix targets. New network saves preserve the
optimizer type and per-layer state in a versioned format; they require an
updated library to load. New files use network format version 4 and preserve
recurrent `return_sequences` settings, network and Dropout random-generator
states, and recurrent hidden/cell states. This allows training to continue
reproducibly after loading in the same build environment. Older network files
can still be loaded: unversioned files resume with fresh SGD, while versions 1
through 3 retain the state supported by their respective formats.
For an older standalone GRU payload written without a layer-version header,
call `GRULayer::deserialize_versioned(stream, 1)`; network and versioned layer
files select the correct GRU reader automatically.

Neural-network regularization applies to each trainable layer's weight matrix,
including any parameters packed into it; separate bias vectors are excluded.
Its gradient is added before weight-gradient clipping. New network saves preserve
the regularizer configuration. Older files contain no such configuration and
load without regularization.
BatchNorm stores both its scale and shift in its weight matrix; both are updated
once per training step. For a single-sample batch, its batch variance is zero.

### Recurrent sequences

SimpleRNN, LSTM and GRU accept batch-first sequences of shape
`(batch_size, time_steps, features)`, reset their hidden (and LSTM cell) state at
the start of each sequence call, and train using backpropagation through time.
With `return_sequences=False`, the output is `(batch_size, units)`; with
`True`, it is `(batch_size, time_steps, units)`. Sequence outputs can feed
another recurrent layer or be the network output. Non-recurrent layers such as
Dense require a preceding recurrent layer with `return_sequences=False`.
Targets must match the network output shape, except a single-output 2D target
may also be supplied as a 1D vector.

```python
import ml_core as ml
import numpy as np

nn = ml.NeuralNetwork()
nn.add_recurrent_layer(ml.RecurrentType.LSTM, 32, return_sequences=True)
nn.add_recurrent_layer(ml.RecurrentType.GRU, 16)
nn.add_dense_layer(1, "linear")

X = np.random.randn(100, 10, 2)
y = np.random.randn(100, 1)
nn.fit_advanced(X, y, epochs=20, batch_size=16)
predictions = nn.predict(X)  # shape (100, 1)
```

For 3D inputs, `predict` and `predict_proba` both return raw network outputs
with the shapes above. NumPy inputs are converted to contiguous `float64`
when needed and copied into the C++ sequence buffer. C++ callers use
`layers::SequenceTensor`, whose `data` is
contiguous batch-first, with `forward_sequence`/`backward_sequence` on
recurrent layers or `fit_sequences`/`predict_sequences` on networks.

```cpp
models::NeuralNetwork nn;
nn.add_recurrent_layer(models::RecurrentType::GRU, 8);
nn.add_dense_layer(1, "linear");
layers::SequenceTensor X{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3, 1, true};
layers::SequenceTensor y{{0.4, 0.7}, 2, 1, 1, false};
nn.fit_sequences(X, y, 20, 2, false);
auto predictions = nn.predict_sequences(X);
```

The existing 2D interface remains a single-step, stateful interface: hidden
state persists between calls until the batch size changes or `reset_state()`
is called on the layer. In network training, samples are shuffled, so 2D
state may carry between unrelated samples; use 3D input for independent
sequences.

### Using Scaler
```python

import ml_core as ml

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

---

## 🧪 Running Tests
### C++ Tests
```bash
cd build
ctest --output-on-failure
```

### Python Tests
```bash
pytest tests/python/
```

### Performance benchmarks

The benchmarks are opt-in and do not require Google Test or Python bindings.
Build and run them sequentially:

```bash
cmake -S . -B build/benchmarks -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_EXAMPLES=OFF -DUSE_OPENMP=OFF
cmake --build build/benchmarks --target benchmark_linear benchmark_logistic benchmark_neural benchmark_sequences
python scripts/generate_all_benchmark.py --build-dir build/benchmarks \
  --output-dir docs/performance_results/my-run --repetitions 3
```

The reporting script requires `matplotlib` and `seaborn`; it refuses to overwrite
an existing output directory. It saves each raw run, CSV/JSON measurements,
median/min/max timings, environment metadata and PNG charts. Repetitions use
the same deterministic datasets and seeds, and measure wall-clock time; compare
runs only on a comparable, otherwise idle environment. Timings include only
training (`fit`), except for the explicitly labelled in-memory checkpoint
serialization/deserialization. For the linear, logistic and Dense benchmarks,
dataset sizes are total sizes; 80% is used for training and 20% for testing.
The sequence benchmark uses 256 training and 64 separate test sequences.
Neural loss is the final training loss. Neural
classification reports the test-set majority-class baseline; sequence regression
reports test MSE and the zero-prediction baseline.

The suite covers linear/logistic regression, Dense optimizers and regularizers,
SimpleRNN/LSTM/GRU sequence training, and an in-memory GRU checkpoint with
continuation validation. It does not measure Python bindings or disk I/O.
For optional regression checking, provide a previous `results.json` from an
identical workload and comparable machine with `--baseline path/to/results.json`.
The default maximum median slowdown is 20%, configurable with
`--max-regression-pct`; an exceeded threshold gives a nonzero exit status.
Without a baseline no timing threshold is applied. On shared hardware the
results are indicative, not a statistically controlled certification.

To run the executables once through CTest instead of producing a report:
`ctest --test-dir build/benchmarks -L performance --output-on-failure`.

The separate `benchmark_inference` target trains fresh synthetic models once
and then reports five repeated batched C++ inference measurements as CSV:

```bash
cmake --build build/benchmarks --target benchmark_inference
build/benchmarks/bin/benchmark_inference > inference.csv
```

Its per-item time is the total batched prediction time divided by the number
of items, not the latency of a one-item request. Matrix models use batches of
1,024; recurrent models use batches of 64 eight-step windows. Results include
the public prediction call and consumption of one output value, but not data
generation or fitting. These synthetic training datasets are separate from
the fit benchmark datasets and the NASA IMS application benchmark.
---

## 🤝 Contributing
1. Fork the project
2. Create a branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Coding Standards

C++17 standard
Use Eigen::MatrixXd for matrices
Exceptions for error handling (see exceptions/)
Document public APIs with Doxygen comments

## 📝 License
The community version is distributed under the GNU Affero General Public License version 3 or, at your option, any later version (AGPL-3.0-or-later). The official version 3 license text is available at https://www.gnu.org/licenses/agpl-3.0.html.

For a separate commercial licensing agreement, contact the copyright holder. Third-party components remain subject to their respective licenses.

## 📧 Contact
**Maurizio Penna** - mauriziopenna@gmail.com  
**Project Link**: https://github.com/codinglab42/machineLearning

## 🙏 Acknowledgements

Eigen3 - Linear algebra library
pybind11 - Python bindings
Andrew Ng - Machine Learning course inspiration

## Riepilogo delle modifiche principali

| Sezione | Modifica |
|---------|----------|
| **C++ standard** | C++14 → C++17 |
| **Python version** | 3.11 only (`>=3.11,<3.12`, package metadata) |
| **Features** | Aggiunti LSTM, GRU, BatchNorm, regolarizzatori |
| **Esempi** | Aggiunti esempi LSTM e scaler |
| **Project structure** | Aggiornata con la nuova struttura |
| **setup.py** | Aggiunta menzione per installazione Python |
| **Acknowledgements** | Aggiunto Andrew Ng (fonte d'ispirazione) |
