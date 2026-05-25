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
    // Costruttori
    LSTMLayer() = default;
    LSTMLayer(int units, int input_size, 
              const std::string& activation = "tanh",
              const std::string& recurrent_activation = "sigmoid",
              bool use_bias = true);
    ~LSTMLayer() override = default;

    // Forward/Backward
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;

    // Gestione pesi
    bool has_weights() const override { return true; }
    void initialize_weights() override;
    Eigen::MatrixXd get_weights() const override;
    void set_weights(const Eigen::MatrixXd& weights) override;
    const Eigen::MatrixXd& get_weights_gradient() const override { return weights_gradient_; }
    void set_weights_gradient(const Eigen::MatrixXd& gradient) override { weights_gradient_ = gradient; }
    
    // Gestione bias
    bool get_use_bias() const override { return use_bias_; }
    Eigen::VectorXd get_biases() const override;
    void set_biases(const Eigen::VectorXd& biases) override;
    const Eigen::VectorXd& get_bias_gradient() const override { return bias_gradient_; }
    void set_bias_gradient(const Eigen::VectorXd& gradient) override { bias_gradient_ = gradient; }
    
    // Parametri
    int get_parameter_count() const override;
    
    // Cache
    void clear_cache() override { if (cache_) cache_->clear(); }
    std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
    void set_cache(std::shared_ptr<LayerCache> cache) override { 
        cache_ = std::dynamic_pointer_cast<LSTMCache>(cache);
    }
    
    // Info
    std::string get_type() const override { return "LSTMLayer"; }
    LayerType get_layer_type() const override { return LayerType::LSTM; }
    std::string get_config() const override;
    uint32_t get_version() const override { return 2; }
    
    // Dimensioni
    int get_input_size() const override { return input_size_; }
    int get_output_size() const override { return units_; }
    void set_input_shape(int input_size) override;
    
    // Metodi RNN
    void reset_state() override;
    Eigen::MatrixXd get_hidden_state() const override;
    Eigen::MatrixXd get_cell_state() const;

    // Serializzazione
    void serialize(std::ostream& out) const override;
    void deserialize(std::istream& in) override;

private:
    // Metodi privati
    Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& z, const std::string& activation) const;
    Eigen::MatrixXd apply_activation_derivative(const Eigen::MatrixXd& z, const std::string& activation) const;
    
    std::shared_ptr<LSTMCache> get_specific_cache() const {
        return std::dynamic_pointer_cast<LSTMCache>(cache_);
    }
    
    // Cache
    std::shared_ptr<LSTMCache> cache_;
    
    // PESI - 4 gate: input (i), forget (f), cell (c), output (o)
    Eigen::MatrixXd kernel_i_;      // input gate kernel [input_size, units]
    Eigen::MatrixXd kernel_f_;      // forget gate kernel [input_size, units]
    Eigen::MatrixXd kernel_c_;      // cell gate kernel [input_size, units]
    Eigen::MatrixXd kernel_o_;      // output gate kernel [input_size, units]
    
    Eigen::MatrixXd recurrent_i_;   // input gate recurrent [units, units]
    Eigen::MatrixXd recurrent_f_;   // forget gate recurrent [units, units]
    Eigen::MatrixXd recurrent_c_;   // cell gate recurrent [units, units]
    Eigen::MatrixXd recurrent_o_;   // output gate recurrent [units, units]
    
    Eigen::VectorXd bias_i_;        // input gate bias [units]
    Eigen::VectorXd bias_f_;        // forget gate bias [units]
    Eigen::VectorXd bias_c_;        // cell gate bias [units]
    Eigen::VectorXd bias_o_;        // output gate bias [units]
    
    // GRADIENTI
    Eigen::MatrixXd weights_gradient_;
    Eigen::VectorXd bias_gradient_;
    
    // Configurazione
    int units_ = 0;
    int input_size_ = 0;
    std::string activation_ = "tanh";
    std::string recurrent_activation_ = "sigmoid";
    bool use_bias_ = true;
    bool return_sequences_ = false;
    
    // Stato
    Eigen::MatrixXd hidden_state_;
    Eigen::MatrixXd cell_state_;
};

} // namespace layers

#endif