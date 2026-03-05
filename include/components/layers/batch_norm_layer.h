#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include "layer.h"
#include "components/cache/batchnorm_cache.h"
#include <Eigen/Dense>
#include <cmath>

namespace layers {

    class BatchNormLayer : public Layer {
    public:
        BatchNormLayer(double epsilon = 1e-5, double momentum = 0.9);
        ~BatchNormLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training = false) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        std::string get_type() const override { return "BatchNormLayer"; }
        std::string get_config() const override;
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return input_size_; }
        
        void clear_cache() override { 
            if (cache_) cache_->clear();
        }
        std::shared_ptr<Cache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<Cache> cache) override { 
            cache_ = std::dynamic_pointer_cast<BatchNormCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override { return beta_; }
        void set_biases(const Eigen::VectorXd& biases) override { beta_ = biases; }
        void set_input_shape(int input_size) override;

    private:
        std::shared_ptr<BatchNormCache> get_specific_cache() {
            return std::dynamic_pointer_cast<BatchNormCache>(cache_);
        }
        
        double epsilon_;
        double momentum_;
        
        // Parametri addestrabili
        Eigen::VectorXd gamma_;  // scale
        Eigen::VectorXd beta_;   // shift
        
        // Statistiche running (non nella cache perché persistono tra batch)
        Eigen::VectorXd running_mean_;
        Eigen::VectorXd running_var_;
        
        int input_size_;
        std::shared_ptr<BatchNormCache> cache_;
    };

} // namespace layers

#endif

