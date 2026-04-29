// components/layers/layer_registry.cpp
#include <memory>
#include "components/layers/layer_factory.h"
#include "components/layers/dense_layer.h"
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"

namespace layers {

void LayerFactory::register_all_layers() {
    // 1. DENSE LAYER
    register_layer(LayerType::DENSE, 
        []() { return std::make_unique<DenseLayer>(1, "relu", true); },
        "DenseLayer");
    
    // 2. CONV2D LAYER
    register_layer(LayerType::CONV2D,
        []() { return std::make_unique<Conv2DLayer>(1, 3, 1, "valid", "relu"); },
        "Conv2DLayer");
    
    // 3. MAX POOLING LAYER
    register_layer(LayerType::MAX_POOLING,
        []() { return std::make_unique<PoolingLayer>(2, 2, PoolingLayer::MAX, 1); },
        "MaxPoolingLayer");
    
    // 4. AVERAGE POOLING LAYER
    register_layer(LayerType::AVERAGE_POOLING,
        []() { return std::make_unique<PoolingLayer>(2, 2, PoolingLayer::AVG, 1); },
        "AveragePoolingLayer");
    
    // 5. FLATTEN LAYER
    register_layer(LayerType::FLATTEN,
        []() { return std::make_unique<FlattenLayer>(); },
        "FlattenLayer");
    
    // 6. DROPOUT LAYER
    register_layer(LayerType::DROPOUT,
        []() { return std::make_unique<DropoutLayer>(0.5); },
        "DropoutLayer");
    
    // 7. BATCH NORM LAYER
    register_layer(LayerType::BATCH_NORM,
        []() { return std::make_unique<BatchNormLayer>(); },
        "BatchNormLayer");
    
    // 8. SIMPLE RNN LAYER
    register_layer(LayerType::SIMPLE_RNN,
        []() { return std::make_unique<SimpleRNNLayer>(1, 1, "tanh", true); },
        "SimpleRNNLayer");
    
    // 9. LSTM LAYER
    register_layer(LayerType::LSTM,
        []() { return std::make_unique<LSTMLayer>(1, 1, "tanh", "sigmoid", true); },
        "LSTMLayer");
    
    // 10. GRU LAYER
    register_layer(LayerType::GRU,
        []() { return std::make_unique<GRULayer>(1, 1, "tanh", "sigmoid", true); },
        "GRULayer");
}

} // namespace layers