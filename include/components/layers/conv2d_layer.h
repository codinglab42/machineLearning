#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include "layer.h"
#include "components/cache/conv_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

    class Conv2DLayer : public Layer {
    public:
        Conv2DLayer(int filters, int kernel_size, int strides = 1,
                   const std::string& padding = "valid",
                   const std::string& activation = "relu");
        ~Conv2DLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Conv2DLayer"; }
        LayerType get_layer_type() const override { return LayerType::CONV2D; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        int get_parameter_count() const override;
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override;
        
        void clear_cache() override { if (cache_) cache_->clear(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<ConvCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return bias_; }
        void set_biases(const Eigen::VectorXd& biases) override { bias_ = biases; }
        void set_input_shape(int input_size) override;

    
        const Eigen::MatrixXd& get_weights_gradient() const { return weights_gradient_; }
        const Eigen::VectorXd& get_bias_gradient() const { return bias_gradient_; }
        void set_weights_gradient(const Eigen::MatrixXd& gradient) override { weights_gradient_ = gradient; }
        void set_bias_gradient(const Eigen::VectorXd& gradient) override { bias_gradient_ = gradient; }

        bool get_use_bias() const { return use_bias_; }
        
    private:
        Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& z) const;
        Eigen::MatrixXd apply_activation_derivative(const Eigen::MatrixXd& z) const;
        
        Eigen::MatrixXd im2col(const Eigen::MatrixXd& input, int batch_idx, int start_idx) const;
        Eigen::MatrixXd col2im(const Eigen::MatrixXd& col, int batch_idx, int start_idx) const;
        
        void compute_output_dimensions();
        
        std::shared_ptr<ConvCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<ConvCache>(cache_);
        }
        
        int filters_;
        int kernel_size_;
        int strides_;
        std::string padding_;
        std::string activation_;
        
        Eigen::MatrixXd kernels_;
        Eigen::VectorXd bias_;

        bool use_bias_;
        Eigen::MatrixXd weights_gradient_;
        Eigen::VectorXd bias_gradient_;
        
        int input_size_;
        int input_height_;
        int input_width_;
        int input_channels_;
        int output_height_;
        int output_width_;
        int kernel_elements_;
        
        std::shared_ptr<ConvCache> cache_;
    };

} // namespace layers

#endif

