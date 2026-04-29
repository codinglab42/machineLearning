#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "layer.h"
#include "components/cache/flatten_cache.h"
#include <Eigen/Dense>

namespace layers {

    class FlattenLayer : public Layer {
    public:
        FlattenLayer();
        ~FlattenLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "FlattenLayer"; }
        LayerType get_layer_type() const override { return LayerType::FLATTEN; }
        std::string get_config() const override;
        uint32_t get_version() const override { return 1; }
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return Eigen::MatrixXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        int get_parameter_count() const override { return 0; }
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }
        
        void clear_cache() override { if (cache_) cache_.reset(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<FlattenCache>(cache);
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
        std::shared_ptr<FlattenCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<FlattenCache>(cache_);
        }
        
        int input_size_;
        std::shared_ptr<FlattenCache> cache_;

        bool use_bias_;
        Eigen::MatrixXd weights_gradient_;
        Eigen::VectorXd bias_gradient_;
    };

} // namespace layers

#endif
