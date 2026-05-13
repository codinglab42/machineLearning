#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "components/cache/dropout_cache.h"
#include <Eigen/Dense>
#include <random>

namespace layers {

    class DropoutLayer : public Layer {
    public:
        DropoutLayer(double rate = 0.5);
        ~DropoutLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "DropoutLayer"; }
        LayerType get_layer_type() const override { return LayerType::DROPOUT; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        double get_rate() const { return rate_; }
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return Eigen::MatrixXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        int get_parameter_count() const override { return 0; }
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }
        
        void clear_cache() override { if (cache_) cache_->clear(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<DropoutCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return Eigen::VectorXd(); }
        void set_biases(const Eigen::VectorXd& biases) override {}
        void set_input_shape(int input_size) override;

    
        const Eigen::MatrixXd& get_weights_gradient() const { return weights_gradient_; }
        const Eigen::VectorXd& get_bias_gradient() const { return bias_gradient_; }
        void set_weights_gradient(const Eigen::MatrixXd& gradient) override { (void)gradient; }
        void set_bias_gradient(const Eigen::VectorXd& gradient) override { (void)gradient; }

        bool get_use_bias() const { return use_bias_; }
        
    private:
        std::shared_ptr<DropoutCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<DropoutCache>(cache_);
        }
        
        double rate_;
        double scale_;
        int input_size_;
        mutable std::mt19937 rng_;
        std::shared_ptr<DropoutCache> cache_;

        bool use_bias_;
        Eigen::MatrixXd weights_gradient_;
        Eigen::VectorXd bias_gradient_;
    };

} // namespace layers

#endif
