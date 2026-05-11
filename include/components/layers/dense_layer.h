// include/components/layers/dense_layer.h
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "components/cache/dense_cache.h"
#include <Eigen/Dense>
#include <random>
#include <string>

namespace layers {

class DenseLayer : public Layer {
public:
    DenseLayer(int units, const std::string& activation = "relu", bool use_bias = true);
    ~DenseLayer() override = default;

    // ========================================================================
    // FORWARD/BACKWARD PASS
    // ========================================================================
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool training) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;  // ← 1 parametro!

    // ========================================================================
    // GESTIONE PESI (proprietà del layer)
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
    Eigen::VectorXd get_biases() const override { return bias_; }
    const Eigen::VectorXd& get_bias_gradient() const override { return bias_gradient_; }
    
    // Setter bias
    void set_biases(const Eigen::VectorXd& biases) override;
    void set_bias_gradient(const Eigen::VectorXd& gradient) override { bias_gradient_ = gradient; }
    
    // ========================================================================
    // PARAMETRI
    // ========================================================================
    int get_parameter_count() const override;
    
    // ========================================================================
    // CACHE MANAGEMENT (solo dati temporanei)
    // ========================================================================
    void clear_cache() override { if (cache_) cache_->clear(); }
    std::shared_ptr<LayerCache> get_cache() const override { return cache_; }
    void set_cache(std::shared_ptr<LayerCache> cache) override { 
        cache_ = std::dynamic_pointer_cast<DenseCache>(cache);
    }
    
    // ========================================================================
    // INFO LAYER
    // ========================================================================
    LayerType get_layer_type() const override { return LayerType::DENSE; }
    std::string get_type() const override { return "DenseLayer"; }
    std::string get_config() const override;
    uint32_t get_version() const override { return 2; }  // Versione aggiornata
    
    // ========================================================================
    // DIMENSIONI
    // ========================================================================
    int get_input_size() const override { return input_size_; }
    int get_output_size() const override { return units_; }
    void set_input_shape(int input_size) override;
    
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
    
    // ========================================================================
    // MEMBRI
    // ========================================================================
    
    // Cache (solo dati temporanei del forward)
    std::shared_ptr<DenseCache> cache_;
    
    // PESI - persistenti (solo qui!)
    Eigen::MatrixXd weights_;      // [input_size, units]
    Eigen::VectorXd bias_;         // [units]
    
    // GRADIENTI - temporanei (calcolati nel backward)
    Eigen::MatrixXd weights_gradient_;  // [input_size, units]
    Eigen::VectorXd bias_gradient_;     // [units]
    
    // Configurazione
    int units_;
    std::string activation_;
    bool use_bias_;
    int input_size_;
};

} // namespace layers

#endif // DENSE_LAYER_H