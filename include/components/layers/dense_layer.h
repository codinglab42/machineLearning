#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "components/cache/dense_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

    class DenseLayer : public Layer {
    public:
        DenseLayer(int units, const std::string& activation = "relu", bool use_bias = true);
        ~DenseLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "DenseLayer"; }
        LayerType get_layer_type() const override { return LayerType::DENSE; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        int get_parameter_count() const override;
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return units_; }
        
        void clear_cache() override { if (cache_) cache_->clear(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<DenseCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return bias_; }
        void set_biases(const Eigen::VectorXd& biases) override { bias_ = biases; }
        void set_input_shape(int input_size) override;

    private:
        Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& z) const;
        Eigen::MatrixXd apply_activation_derivative(const Eigen::MatrixXd& z) const;
        
        std::shared_ptr<DenseCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<DenseCache>(cache_);
        }
        
        int units_;
        std::string activation_;
        bool use_bias_;
        
        Eigen::MatrixXd weights_;
        Eigen::VectorXd bias_;
        
        int input_size_;
        std::shared_ptr<DenseCache> cache_;
    };

} // namespace layers

#endif
