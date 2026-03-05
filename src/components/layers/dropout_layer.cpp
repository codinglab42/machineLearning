#include "components/layers/dropout_layer.h"

namespace layers {

    DropoutLayer::DropoutLayer(double rate) 
        : rate_(rate), 
          scale_(1.0 / (1.0 - rate)),
          input_size_(0),
          rng_(std::random_device{}()) {
        
        if (rate < 0.0 || rate >= 1.0) {
            throw std::invalid_argument("DropoutLayer: rate must be in [0, 1)");
        }
    }

    Eigen::MatrixXd DropoutLayer::forward(const Eigen::MatrixXd& input, bool training) {
        // Crea cache se non esiste
        if (!cache_) {
            cache_ = std::make_shared<DropoutCache>();
        }
        
        cache_->input_cache = input;
        cache_->training = training;
        
        if (!training) {
            cache_->output_cache = input;
            return input;  // in inference, nessun dropout
        }
        
        input_size_ = input.cols();
        cache_->mask.resize(input.rows(), input.cols());
        
        // Genera maschera di dropout
        std::bernoulli_distribution dist(1.0 - rate_);
        
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                cache_->mask(i, j) = dist(rng_) ? scale_ : 0.0;
            }
        }
        
        cache_->output_cache = input.cwiseProduct(cache_->mask);
        return cache_->output_cache;
    }

    Eigen::MatrixXd DropoutLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            throw std::runtime_error("DropoutLayer: cache not initialized. Call forward first.");
        }
        
        if (!cache_->training) {
            return gradient;  // in inference mode, gradiente passa invariato
        }
        
        // Applica la stessa maschera al gradiente
        return gradient.cwiseProduct(cache_->mask);
    }

    std::string DropoutLayer::get_config() const {
        return "DropoutLayer(rate=" + std::to_string(rate_) + ")";
    }

} // namespace layers

