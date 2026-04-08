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

// Implementazione dei metodi statici privati
std::unordered_map<LayerType, LayerFactory::Creator>& LayerFactory::get_creators() {
    static std::unordered_map<LayerType, Creator> creators;
    return creators;
}

std::unordered_map<LayerType, std::string>& LayerFactory::get_names() {
    static std::unordered_map<LayerType, std::string> names;
    return names;
}

std::unordered_map<std::string, LayerType>& LayerFactory::get_type_map() {
    static std::unordered_map<std::string, LayerType> type_map;
    return type_map;
}

bool& LayerFactory::is_registered() {
    static bool registered = false;
    return registered;
}

void LayerFactory::register_layer(LayerType type, Creator creator, const std::string& name) {
    get_creators()[type] = creator;
    get_names()[type] = name;
    get_type_map()[name] = type;
}

std::unique_ptr<Layer> LayerFactory::create(LayerType type) {
    // Assicura che i layer siano registrati
    if (!is_registered()) {
        register_all_layers();
        is_registered() = true;
    }
    
    auto it = get_creators().find(type);
    if (it != get_creators().end()) {
        return it->second();
    }
    ML_THROW_PARAMETER_ERROR("layer type", 
                             "unknown layer type: " + std::to_string(static_cast<int>(type)), 
                             "LayerFactory");
    return nullptr;
}

std::unique_ptr<Layer> LayerFactory::create(const std::string& name) {
    if (!is_registered()) {
        register_all_layers();
        is_registered() = true;
    }
    
    auto it = get_type_map().find(name);
    if (it != get_type_map().end()) {
        return create(it->second);
    }
    ML_THROW_PARAMETER_ERROR("layer name", 
                             "unknown layer name: " + name, 
                             "LayerFactory");
    return nullptr;
}

void LayerFactory::serialize_type(std::ostream& out, LayerType type) {
    utils::write_scalar(out, static_cast<uint8_t>(type));
}

LayerType LayerFactory::deserialize_type(std::istream& in) {
    uint8_t type_val;
    utils::read_scalar(in, type_val);
    if (type_val < static_cast<uint8_t>(LayerType::DENSE) || 
        type_val > static_cast<uint8_t>(LayerType::GRU)) {
        ML_THROW_PARAMETER_ERROR("layer type", 
                                 "invalid layer type value: " + std::to_string(type_val), 
                                 "LayerFactory");
    }
    return static_cast<LayerType>(type_val);
}

std::string LayerFactory::get_name(LayerType type) {
    auto it = get_names().find(type);
    if (it != get_names().end()) {
        return it->second;
    }
    return "Unknown";
}

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