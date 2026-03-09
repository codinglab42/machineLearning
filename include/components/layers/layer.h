#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <iostream>
#include "exceptions/exception_macros.h"

namespace layers {

    class LayerCache;

    class Layer {
    public:
        virtual ~Layer() = default;
        
        // Forward pass
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) {
            return forward(input);
        }
        
        // Backward pass
        virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) = 0;
        
        // Serializzazione
        virtual void serialize(std::ostream& out) const = 0;
        virtual void deserialize(std::istream& in) = 0;
        
        // Info layer
        virtual std::string get_type() const = 0;
        virtual std::string get_config() const = 0;
        
        // Gestione parametri
        virtual bool has_weights() const = 0;
        virtual Eigen::MatrixXd get_weights() const = 0;
        virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
        virtual int get_parameter_count() const = 0;
        
        // Dimensioni
        virtual int get_input_size() const = 0;
        virtual int get_output_size() const = 0;
        
        // Cache management
        virtual void clear_cache() = 0;
        virtual std::shared_ptr<LayerCache> get_cache() const = 0;
        virtual void set_cache(std::shared_ptr<LayerCache> cache) = 0;
        
        // Bias management
        virtual Eigen::VectorXd get_biases() const = 0;
        virtual void set_biases(const Eigen::VectorXd& biases) = 0;
        
        // Input shape
        virtual void set_input_shape(int input_size) = 0;
    };

} // namespace layers

#endif

