// include/components/layers/layer.h
#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <iostream>
#include "exceptions/exception_macros.h"
#include "components/cache/layer_cache.h"

namespace layers {

enum class LayerType : uint8_t {
    DENSE = 1,
    CONV2D = 2,
    MAX_POOLING = 3,
    AVERAGE_POOLING = 4,
    FLATTEN = 5,
    DROPOUT = 6,
    BATCH_NORM = 7,
    SIMPLE_RNN = 8,
    LSTM = 9,
    GRU = 10
};

class Layer {
public:
    virtual ~Layer() = default;
    
    // Forward/Backward pass
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) = 0;
    
    // Gestione pesi (ORA SONO SOLO QUI!)
    virtual bool has_weights() const = 0;
    virtual void initialize_weights() = 0;
    virtual Eigen::MatrixXd get_weights() const = 0;
    virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
    virtual void set_weights_gradient(const Eigen::MatrixXd& gradient) = 0;
    virtual const Eigen::MatrixXd& get_weights_gradient() const = 0;
    
    virtual bool get_use_bias() const = 0;
    virtual Eigen::VectorXd get_biases() const = 0;
    virtual void set_biases(const Eigen::VectorXd& biases) = 0;
    virtual void set_bias_gradient(const Eigen::VectorXd& gradient) = 0;
    virtual const Eigen::VectorXd& get_bias_gradient() const = 0;
    
    virtual int get_parameter_count() const = 0;
    
    // Cache management (solo dati temporanei)
    virtual void clear_cache() = 0;
    virtual std::shared_ptr<LayerCache> get_cache() const = 0;
    virtual void set_cache(std::shared_ptr<LayerCache> cache) = 0;
    
    // Info layer
    virtual LayerType get_layer_type() const = 0;
    virtual std::string get_type() const = 0;
    virtual std::string get_config() const = 0;
    virtual uint32_t get_version() const { return 1; }
    
    // Dimensioni
    virtual int get_input_size() const = 0;
    virtual int get_output_size() const = 0;
    virtual void set_input_shape(int input_size) = 0;
    
    // Serializzazione
    virtual void serialize(std::ostream& out) const = 0;
    virtual void deserialize(std::istream& in) = 0;

protected:

    int input_size_ = 0;
    int output_size_ = 0;
};

} // namespace layers

#endif