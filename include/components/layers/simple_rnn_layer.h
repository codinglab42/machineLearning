// include/components/layers/simple_rnn_layer.h
#ifndef SIMPLE_RNN_LAYER_H
#define SIMPLE_RNN_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/simple_rnn_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

class SimpleRNNLayer : public RecurrentLayer {
public:
    SimpleRNNLayer(int units, int input_size, 
                  const std::string& activation = "tanh",
                  bool use_bias = true);
    ~SimpleRNNLayer() override = default;

    // ========================================================================
    // FORWARD/BACKWARD PASS
    // ========================================================================
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;  // ← 1 parametro!

    // ========================================================================
    // GESTIONE PESI
    // ========================================================================
    bool has_weights() const override { return true; }
    
    // Getter pesi
    Eigen::MatrixXd get_weights() const override;
    const Eigen::MatrixXd& get_weights_gradient() const override { return weights_gradient_; }
    
    // Setter pesi
    void set_weights(const Eigen::MatrixXd& weights) override;
    void set_weights_gradient(const Eigen::MatrixXd& gradient) override { weights_gradient_ = gradient; }
    
    // ========================================================================
    // GESTIONE BIAS
    // ========================================================================
    bool get_use_bias() const override { return use_bias_; }
    
    // Getter bias
    Eigen::VectorXd get_biases() const override;
    const Eigen::VectorXd& get_bias_gradient() const override { return bias_gradient_; }
    
    // Setter bias
    void set_biases(const Eigen::VectorXd& biases) override;
    void set_bias_gradient(const Eigen::VectorXd& gradient) override { bias_gradient_ = gradient; }
    
    // ========================================================================
    // PARAMETRI
    // ========================================================================
    int get_parameter_count() const override;
    
    // ========================================================================
    // CACHE MANAGEMENT
    // ========================================================================
    void clear_cache() override { if (cache_) cache_->clear(); }
    std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
    void set_cache(std::shared_ptr<LayerCache> cache) override { 
        cache_ = std::dynamic_pointer_cast<SimpleRNNCache>(cache);
    }
    
    // ========================================================================
    // INFO LAYER
    // ========================================================================
    std::string get_type() const override { return "SimpleRNNLayer"; }
    LayerType get_layer_type() const override { return LayerType::SIMPLE_RNN; }
    std::string get_config() const override;
    uint32_t get_version() const override { return 2; }
    
    // ========================================================================
    // DIMENSIONI
    // ========================================================================
    int get_input_size() const override { return input_size_; }
    int get_output_size() const override { return units_; }
    void set_input_shape(int input_size) override;
    
    // ========================================================================
    // METODI SPECIFICI RNN
    // ========================================================================
    void reset_state() override;
    Eigen::MatrixXd get_hidden_state() const override;
    
    // ========================================================================
    // SERIALIZZAZIONE
    // ========================================================================
    void serialize(std::ostream& out) const override;
    void deserialize(std::istream& in) override;

private:
    // ========================================================================
    // METODI PRIVATI
    // ========================================================================
    Eigen::MatrixXd apply_activation(const Eigen::MatrixXd& z) const;
    Eigen::MatrixXd apply_activation_derivative(const Eigen::MatrixXd& z) const;
    
    std::shared_ptr<SimpleRNNCache> get_specific_cache() const {
        return std::dynamic_pointer_cast<SimpleRNNCache>(cache_);
    }
    
    // ========================================================================
    // MEMBRI
    // ========================================================================
    
    // Cache (solo dati temporanei)
    std::shared_ptr<SimpleRNNCache> cache_;
    
    // PESI - persistenti
    Eigen::MatrixXd kernel_;      // Pesi per input [input_size, units]
    Eigen::MatrixXd recurrent_;   // Pesi ricorrenti [units, units]
    Eigen::VectorXd bias_;        // Bias [units]
    
    // GRADIENTI - temporanei
    Eigen::MatrixXd weights_gradient_;
    Eigen::VectorXd bias_gradient_;
    
    // Configurazione
    int units_;
    int input_size_;
    std::string activation_;
    bool use_bias_;
    
    // Stato
    Eigen::MatrixXd hidden_state_;  // Stato nascosto corrente [batch, units]
};

} // namespace layers

#endif // SIMPLE_RNN_LAYER_H