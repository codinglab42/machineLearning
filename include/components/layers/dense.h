#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "components/activation/activation.h"
#include <memory>

namespace layers {

    class Dense : public Layer {
    public:
         Dense(int input_size, int output_size,
          const std::string& activation = "identity",
          double l1_lambda = 0.0,
          double l2_lambda = 0.0,
          const std::string& weight_initializer = "he",
          const std::string& bias_initializer = "zeros");
    
        ~Dense() override = default;

        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "Dense"; }
        std::string get_config() const override;
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return output_size_; }
        int get_parameter_count() const override;
        // accesso ai gradienti
        Eigen::MatrixXd get_weight_gradients() const { return grad_weights_; }
        Eigen::VectorXd get_bias_gradients() const { return grad_biases_; }
    
        
        void clear_cache() override;
        const LayerCache& get_cache() const override { return cache_; }
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override { return weights_; }
        Eigen::VectorXd get_biases() const override { return biases_; }
        void set_weights(const Eigen::MatrixXd& weights) override;
        void set_biases(const Eigen::VectorXd& biases) override;
        
        
        // Specific methods
        void set_activation(const std::string& activation);
        void set_regularization(double l1, double l2);

        // Utility per test
        void clear_gradients() {
            grad_weights_.setZero();
            grad_biases_.setZero();
        }
        
    private:
        int input_size_;
        int output_size_;
        Eigen::MatrixXd weights_;
        Eigen::VectorXd biases_;
        std::unique_ptr<activation::Activation> activation_;

        // Memorize Gradients
        Eigen::MatrixXd grad_weights_;
        Eigen::VectorXd grad_biases_;
        
        // Regularization
        double l1_lambda_;
        double l2_lambda_;
        
        // Cache
        LayerCache cache_;
        
        // Initialization
        void initialize_weights(const std::string& initializer);
        void initialize_biases(const std::string& initializer);
        
        // Regularization gradients
        Eigen::MatrixXd l1_regularization_gradient() const;
        Eigen::MatrixXd l2_regularization_gradient() const;
    };

} // namespace layers

#endif