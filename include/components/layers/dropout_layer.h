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

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training = false) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        std::string get_type() const override { return "DropoutLayer"; }
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
            cache_ = std::dynamic_pointer_cast<DropoutCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return Eigen::VectorXd(); }
        void set_biases(const Eigen::VectorXd& biases) override {}
        void set_input_shape(int input_size) override { input_size_ = input_size; }

    private:
        std::shared_ptr<DropoutCache> get_specific_cache() {
            return std::dynamic_pointer_cast<DropoutCache>(cache_);
        }
        
        double rate_;
        double scale_;
        int input_size_;
        std::mt19937 rng_;
        std::shared_ptr<DropoutCache> cache_;
    };

} // namespace layers

#endif
