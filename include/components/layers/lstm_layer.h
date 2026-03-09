#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/lstm_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

    class LSTMLayer : public RecurrentLayer {
    public:
        LSTMLayer(int units, int input_size, 
                 const std::string& activation = "tanh",
                 const std::string& recurrent_activation = "sigmoid",
                 bool use_bias = true);
        ~LSTMLayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "LSTMLayer"; }
        std::string get_config() const override;
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        int get_parameter_count() const override;
        
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return units_; }
        
        void clear_cache() override { if (cache_) cache_->clear(); }
        std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
        void set_cache(std::shared_ptr<LayerCache> cache) override { 
            cache_ = std::dynamic_pointer_cast<LSTMCache>(cache);
        }
        
        Eigen::VectorXd get_biases() const override;
        void set_biases(const Eigen::VectorXd& biases) override;
        void set_input_shape(int input_size) override;
        
        void reset_state() override;
        Eigen::MatrixXd get_hidden_state() const override;

    private:
        Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) const;
        Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x) const;
        Eigen::MatrixXd tanh(const Eigen::MatrixXd& x) const;
        Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd& x) const;
        
        std::shared_ptr<LSTMCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<LSTMCache>(cache_);
        }
        
        int units_;
        int input_size_;
        std::string activation_;
        std::string recurrent_activation_;
        bool use_bias_;
        
        // Pesi per i 4 gate: input, forget, cell, output
        Eigen::MatrixXd kernel_i;      // Pesi input gate [input_size, units]
        Eigen::MatrixXd kernel_f;      // Pesi forget gate [input_size, units]
        Eigen::MatrixXd kernel_c;      // Pesi cell candidate [input_size, units]
        Eigen::MatrixXd kernel_o;      // Pesi output gate [input_size, units]
        
        Eigen::MatrixXd recurrent_i;   // Pesi ricorrenti input gate [units, units]
        Eigen::MatrixXd recurrent_f;   // Pesi ricorrenti forget gate [units, units]
        Eigen::MatrixXd recurrent_c;   // Pesi ricorrenti cell candidate [units, units]
        Eigen::MatrixXd recurrent_o;   // Pesi ricorrenti output gate [units, units]
        
        Eigen::VectorXd bias_i;        // Bias input gate [units]
        Eigen::VectorXd bias_f;        // Bias forget gate [units]
        Eigen::VectorXd bias_c;        // Bias cell candidate [units]
        Eigen::VectorXd bias_o;        // Bias output gate [units]
        
        Eigen::MatrixXd hidden_state_;  // Stato nascosto corrente [batch, units]
        Eigen::MatrixXd cell_state_;    // Stato cella corrente [batch, units]
        
        std::shared_ptr<LSTMCache> cache_;
    };

} // namespace layers

#endif

