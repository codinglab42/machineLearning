#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "layer.h"
#include "components/cache/batchnorm_cache.h"
#include <Eigen/Dense>

namespace layers {

    class BatchNormLayer : public Layer {
    public:
        BatchNormLayer(double epsilon = 1e-5, double momentum = 0.9);
        ~BatchNormLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "BatchNormLayer"; }
        LayerType get_layer_type() const override { return LayerType::BATCH_NORM; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        int get_parameter_count() const override;
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }
        
        void clear_cache() override { if (cache_) cache_.reset(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<BatchNormCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return beta_; }
        void set_biases(const Eigen::VectorXd& biases) override { beta_ = biases; }
        void set_input_shape(int input_size) override;

        
        const Eigen::MatrixXd& get_weights_gradient() const { return weights_gradient_; }
        const Eigen::VectorXd& get_bias_gradient() const { return bias_gradient_; }
        void set_weights_gradient(const Eigen::MatrixXd& gradient) override { weights_gradient_ = gradient; }
        void set_bias_gradient(const Eigen::VectorXd& gradient) override { bias_gradient_ = gradient; }

        bool get_use_bias() const { return use_bias_; }
        
    private:
        std::shared_ptr<BatchNormCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<BatchNormCache>(cache_);
        }
        
        double epsilon_;
        double momentum_;
        
        Eigen::VectorXd gamma_;
        Eigen::VectorXd beta_;
        
        Eigen::VectorXd running_mean_;
        Eigen::VectorXd running_var_;
        
        int input_size_;
        std::shared_ptr<BatchNormCache> cache_;

        bool use_bias_;
        Eigen::MatrixXd weights_gradient_;
        Eigen::VectorXd bias_gradient_;
        
    };

} // namespace layers

#endif
