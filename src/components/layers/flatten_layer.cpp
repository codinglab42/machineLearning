#include "components/layers/flatten_layer.h"

namespace layers {

    FlattenLayer::FlattenLayer() : input_size_(0) {}

    Eigen::MatrixXd FlattenLayer::forward(const Eigen::MatrixXd& input, bool training) {
        // Crea cache se non esiste
        if (!cache_) {
            cache_ = std::make_shared<FlattenCache>();
        }
        
        // Salva la forma originale nella cache
        cache_->original_shape.clear();
        cache_->original_shape.push_back(input.rows());  // numero campioni
        cache_->original_shape.push_back(input.cols());  // numero features per campione
        
        cache_->input_cache = input;
        cache_->output_cache = input;  // Flatten non cambia l'input in questo caso
        
        return input;
    }

    Eigen::MatrixXd FlattenLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            throw std::runtime_error("FlattenLayer: cache not initialized. Call forward first.");
        }
        
        // Il gradiente passa attraverso invariato
        // La forma è già corretta per il layer precedente
        return gradient;
    }

    std::string FlattenLayer::get_config() const {
        return "FlattenLayer()";
    }

} // namespace layers
