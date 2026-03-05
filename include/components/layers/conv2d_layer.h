#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include "layer.h"
#include "components/cache/conv_cache.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

namespace layers {

    class Conv2DLayer : public Layer {
    public:
        Conv2DLayer(int filters, int kernel_size, int strides = 1,
                   const std::string& padding = "valid",
                   const std::string& activation = "relu");
        ~Conv2DLayer() override = default;

        // Layer interface - implementa entrambe le versioni di forward
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        // Serializzazione
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        // Info layer
        std::string get_type() const override { return "Conv2DLayer"; }
        std::string get_config() const override;
        
        // Gestione parametri
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        int get_parameter_count() const override;
        
        // Dimensioni
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override;
        
        // Gestione cache - usa LayerCache
        void clear_cache() override { 
            if (cache_) cache_->clear();
        }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<ConvCache>(cache);
        }
        
        // Bias management
        Eigen::VectorXd get_biases() const override { return bias_; }
        void set_biases(const Eigen::VectorXd& biases) override { bias_ = biases; }
        
        // Input shape
        void set_input_shape(int input_size) override;

    private:
        // Attivazioni
        Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& z) const;
        Eigen::MatrixXd apply_activation_derivative(const Eigen::MatrixXd& z) const;
        
        // Utilità convoluzione
        Eigen::MatrixXd im2col(const Eigen::MatrixXd& input, int batch_idx, int start_idx) const;
        Eigen::MatrixXd col2im(const Eigen::MatrixXd& col, int batch_idx, int start_idx) const;
        
        // Calcolo dimensioni output
        void compute_output_dimensions();
        
        // Cache specifica (const version per metodi const)
        std::shared_ptr<ConvCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<ConvCache>(cache_);
        }
        
        // Cache specifica (non-const version)
        std::shared_ptr<ConvCache> get_specific_cache() {
            return std::dynamic_pointer_cast<ConvCache>(cache_);
        }
        
        // Parametri del layer
        int filters_;
        int kernel_size_;
        int strides_;
        std::string padding_;
        std::string activation_;
        
        // Pesi
        Eigen::MatrixXd kernels_;
        Eigen::VectorXd bias_;
        
        // Dimensioni
        int input_size_;
        int input_height_;
        int input_width_;
        int input_channels_;
        int output_height_;
        int output_width_;
        int kernel_elements_;
        
        // Cache
        std::shared_ptr<ConvCache> cache_;
    };

} // namespace layers

#endif
