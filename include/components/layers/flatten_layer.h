#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "layer.h"
#include "components/cache/flatten_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class FlattenLayer : public Layer {
    public:
        FlattenLayer();
        ~FlattenLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training = false) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        std::string get_type() const override { return "FlattenLayer"; }
        std::string get_config() const override;
        
        bool has_weights() const override { return false; }
        Eigen::MatrixXd get_weights() const override { return Eigen::MatrixXd(); }
        void set_weights(const Eigen::MatrixXd& weights) override {}
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }
        
        void clear_cache() override { 
            if (cache_) cache_->clear();
        }
        std::shared_ptr<Cache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<Cache> cache) override { 
            cache_ = std::dynamic_pointer_cast<FlattenCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return Eigen::VectorXd(); }
        void set_biases(const Eigen::VectorXd& biases) override {}
        void set_input_shape(int input_size) override { input_size_ = input_size; }

    private:
        std::shared_ptr<FlattenCache> get_specific_cache() {
            return std::dynamic_pointer_cast<FlattenCache>(cache_);
        }
        
        int input_size_;
        std::shared_ptr<FlattenCache> cache_;
    };

} // namespace layers

#endif
