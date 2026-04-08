#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/gru_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

    class GRULayer : public RecurrentLayer {
    public:
        GRULayer(int units, int input_size, 
                const std::string& activation = "tanh",
                const std::string& recurrent_activation = "sigmoid",
                bool use_bias = true);
        ~GRULayer() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, double learning_rate) override;
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::string get_type() const override { return "GRULayer"; }
        LayerType get_layer_type() const override { return LayerType::GRU; }
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
            cache_ = std::dynamic_pointer_cast<GRUCache>(cache);
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
        
        std::shared_ptr<GRUCache> get_specific_cache() const {
            return std::dynamic_pointer_cast<GRUCache>(cache_);
        }
        
        int units_;
        int input_size_;
        std::string activation_;
        std::string recurrent_activation_;
        bool use_bias_;
        
        // Pesi per i 3 gate: reset, update, candidate
        Eigen::MatrixXd kernel_r;      // Pesi reset gate [input_size, units]
        Eigen::MatrixXd kernel_z;      // Pesi update gate [input_size, units]
        Eigen::MatrixXd kernel_h;      // Pesi candidate [input_size, units]
        
        Eigen::MatrixXd recurrent_r;   // Pesi ricorrenti reset gate [units, units]
        Eigen::MatrixXd recurrent_z;   // Pesi ricorrenti update gate [units, units]
        Eigen::MatrixXd recurrent_h;   // Pesi ricorrenti candidate [units, units]
        
        Eigen::VectorXd bias_r;        // Bias reset gate [units]
        Eigen::VectorXd bias_z;        // Bias update gate [units]
        Eigen::VectorXd bias_h;        // Bias candidate [units]
        
        Eigen::MatrixXd hidden_state_;  // Stato nascosto corrente [batch, units]
        
        std::shared_ptr<GRUCache> cache_;
    };

} // namespace layers

#endif

