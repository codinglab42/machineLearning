# Layers Binding

## Overview

This module exposes all available layers from the ML library as Python objects. Layers can be used either as independent components for building custom neural networks, or through the `NeuralNetwork` class which manages them automatically.

## File Structure

bindings/layers/
├── README.md # Questo file
├── layer_base.h/cpp # Classe base Layer
├── dense_layer.h/cpp # DenseLayer (fully connected)
├── conv2d_layer.h/cpp # Conv2DLayer (convoluzionale 2D)
├── pooling_layer.h/cpp # PoolingLayer (max/average pooling)
├── flatten_layer.h/cpp # FlattenLayer (appiattimento)
├── dropout_layer.h/cpp # DropoutLayer (regolarizzazione)
├── batch_norm_layer.h/cpp # BatchNormLayer (normalizzazione batch)
├── simple_rnn_layer.h/cpp # SimpleRNNLayer (RNN base)
├── lstm_layer.h/cpp # LSTMLayer (Long Short-Term Memory)
├── gru_layer.h/cpp # GRULayer (Gated Recurrent Unit)
└── layer_factory.h/cpp # LayerFactory (creazione dinamica)


## Available Layers

### 1. DenseLayer (Fully Connected)
A fully connected layer with various activation functions.

**Constructors:**
```python
DenseLayer(units: int, activation: str = "relu", use_bias: bool = True)
DenseLayer()  # Default constructor

Parameters:

    units: Number of neurons in the layer

    activation: Activation function ("relu", "sigmoid", "tanh", "softmax", "linear")

    use_bias: Whether to use bias

Main Methods:

forward(input: MatrixXd, training: bool = False) -> MatrixXd
backward(gradient: MatrixXd) -> MatrixXd
get_weights() -> MatrixXd
set_weights(weights: MatrixXd)
get_biases() -> VectorXd
set_biases(biases: VectorXd)
initialize_weights()
get_parameter_count() -> int


2. Conv2DLayer (Convolutional 2D)

Convolutional layer for images and spatial data.

Constructors:

Conv2DLayer(filters: int, kernel_size: int, strides: int = 1,
            padding: str = "valid", activation: str = "relu")
Conv2DLayer()  # Default constructor

Parameters:

    filters: Number of filters (output channels)

    kernel_size: Kernel size (e.g., 3 for 3x3)

    strides: Stride of the convolution

    padding: "valid" or "same"

    activation: Activation function

3. PoolingLayer

Pooling layer for dimensionality reduction.

Constructors:

PoolingLayer(pool_size: int = 2, stride: int = 2,
             type: PoolType = PoolType.MAX, channels: int = 1)
PoolingLayer()  # Default constructor

Pool Types:

    PoolType.MAX: Max pooling

    PoolType.AVG: Average pooling

4. FlattenLayer

Flattens multi-dimensional input into a vector.

Constructors:
FlattenLayer()

5. DropoutLayer

Dropout layer for preventing overfitting.

Constructors:
DropoutLayer(rate: float = 0.5)
DropoutLayer()  # Default constructor

Parameters:

    rate: Dropout rate (probability of deactivating a neuron)

6. BatchNormLayer

Batch Normalization for stabilizing training.

Constructors:

BatchNormLayer(epsilon: float = 1e-5, momentum: float = 0.9)
BatchNormLayer()  # Default constructor

7. SimpleRNNLayer

Basic Recurrent Neural Network layer.

Constructors:

SimpleRNNLayer(units: int, input_size: int,
               activation: str = "tanh", use_bias: bool = True)
SimpleRNNLayer()  # Default constructor

8. LSTMLayer

Long Short-Term Memory - Advanced RNN.

Constructors:

LSTMLayer(units: int, input_size: int,
          activation: str = "tanh",
          recurrent_activation: str = "sigmoid",
          use_bias: bool = True)
LSTMLayer()  # Default constructor

9. GRULayer

Gated Recurrent Unit - Simplified RNN.

Constructors:

GRULayer(units: int, input_size: int,
         activation: str = "tanh",
         recurrent_activation: str = "sigmoid",
         use_bias: bool = True)
GRULayer()  # Default constructor

LayerFactory

Factory for dynamically creating layers.

Methods:

LayerFactory.create(type: LayerType) -> Layer
LayerFactory.create_by_name(name: str) -> Layer
LayerFactory.get_name(type: LayerType) -> str
LayerFactory.register_all_layers()

Available Layer Types:

    LayerType.DENSE

    LayerType.CONV2D

    LayerType.MAX_POOLING

    LayerType.AVERAGE_POOLING

    LayerType.FLATTEN

    LayerType.DROPOUT

    LayerType.BATCH_NORM

    LayerType.SIMPLE_RNN

    LayerType.LSTM

    LayerType.GRU

Usage Examples
Example 1: Creating and Using a DenseLayer

import ml_core as ml
import numpy as np

# Create a dense layer
layer = ml.DenseLayer(64, activation="relu", use_bias=True)
layer.set_input_shape(128)  # 128 input features
layer.initialize_weights()

# Forward pass
input_data = np.random.randn(32, 128)  # batch 32, features 128
output = layer.forward(input_data, training=True)
print(f"Output shape: {output.shape}")  # (32, 64)

# Backward pass
gradient = np.random.randn(32, 64)
dX = layer.backward(gradient)
print(f"Gradient shape: {dX.shape}")  # (32, 128)

Example 2: Building a Network with Separate Layers

import ml_core as ml
import numpy as np

# Create layers
layer1 = ml.DenseLayer(64, activation="relu")
layer1.set_input_shape(128)

layer2 = ml.DenseLayer(32, activation="relu")
layer2.set_input_shape(64)

layer3 = ml.DenseLayer(10, activation="softmax")
layer3.set_input_shape(32)

# Initialize weights
layer1.initialize_weights()
layer2.initialize_weights()
layer3.initialize_weights()

# Manual forward pass
X = np.random.randn(16, 128)
h1 = layer1.forward(X, training=True)
h2 = layer2.forward(h1, training=True)
output = layer3.forward(h2, training=True)
print(f"Output shape: {output.shape}")  # (16, 10)

Example 3: Using LayerFactory

import ml_core as ml

# Create layer using factory
layer = ml.LayerFactory.create(ml.LayerType.DENSE)
# or
layer = ml.LayerFactory.create_by_name("DenseLayer")

print(f"Layer type: {layer.get_type()}")
print(f"Layer config: {layer.get_config()}")

Example 4: Recurrent Layers

import ml_core as ml
import numpy as np

# LSTM Layer
lstm = ml.LSTMLayer(units=64, input_size=128)
lstm.set_input_shape(128)
lstm.initialize_weights()

# Forward pass for sequences
X_seq = np.random.randn(10, 32, 128)  # 10 timesteps, batch 32, features 128
hidden_states = []

# Reset state before a new sequence
lstm.reset_state()

for t in range(10):
    output = lstm.forward(X_seq[t], training=True)
    hidden_states.append(output)
    # Get cell state as well
    cell_state = lstm.get_cell_state()

Example 5: Weight Management

import ml_core as ml
import numpy as np

layer = ml.DenseLayer(64, activation="relu")
layer.set_input_shape(128)
layer.initialize_weights()

# Get and modify weights
weights = layer.get_weights()
print(f"Weights shape: {weights.shape}")  # (128, 65) with bias

# Modify weights (e.g., multiply by 0.5)
weights *= 0.5
layer.set_weights(weights)

# Get and modify biases
biases = layer.get_biases()
print(f"Biases shape: {biases.shape}")  # (64,)

# Set biases to zero
layer.set_biases(np.zeros(64))

Important Notes

    Input Shape: Before using a layer, set_input_shape() must be called to initialize dimensions.

    Training Mode: The training=True parameter is important for layers like Dropout and BatchNorm that behave differently during training and inference.

    Serialization: Each layer supports serialize() and deserialize() for disk storage.

    Cache: Layers maintain a cache of forward pass values for backward pass. Use clear_cache() to free memory.

    Weights and Biases: Layers with weights expose getter/setter methods for weights and gradients.

Dependencies

    bindings/core/enums.h - for LayerType, PoolType

    bindings/core/exceptions.h - for exception translator

    components/layers/*.h - C++ layer headers

    Eigen3 - for matrix operations

Testing

To test the layer binding:

cd build
python3 -c "
import ml_core as ml
layer = ml.DenseLayer(64, 'relu')
print('Layer created:', layer.get_type())
print('Config:', layer.get_config())
print('Test OK!')
"

Version Information
Layer	Version	Notes
DenseLayer	2	Unified bias support
Conv2DLayer	1	-
PoolingLayer	2	Supports max and avg
FlattenLayer	1	-
DropoutLayer	1	-
BatchNormLayer	1	-
SimpleRNNLayer	2	-
LSTMLayer	2	-
GRULayer	1	-

References

    C++ Layer Documentation https://../cpp/components/layers/

    Neural Network Binding https://../models/neural_network.md

    Loss Functions Binding https://../loss/README.md

    Optimizers Binding https://../optimizers/README.md
