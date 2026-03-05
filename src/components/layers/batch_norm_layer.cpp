#include "components/layers/batch_norm_layer.h"

namespace layers {

    BatchNormLayer::BatchNormLayer(double epsilon, double momentum)
        : epsilon_(epsilon),
          momentum_(momentum),
          input_size_(0) {
        
        if (epsilon <= 0) throw std::invalid_argument("BatchNormLayer: epsilon must be > 0");
        if (momentum <= 0 || momentum >= 1) throw std::invalid_argument("BatchNormLayer: momentum must be in (0, 1)");
    }

    void BatchNormLayer::set_input_shape(int input_size) {
        input_size_ = input_size;
        
        // Inizializza parametri
        gamma_ = Eigen::VectorXd::Ones(input_size);
        beta_ = Eigen::VectorXd::Zero(input_size);
        running_mean_ = Eigen::VectorXd::Zero(input_size);
        running_var_ = Eigen::VectorXd::Ones(input_size);
    }

    Eigen::MatrixXd BatchNormLayer::forward(const Eigen::MatrixXd& input, bool training) {
        // Crea cache se non esiste
        if (!cache_) {
            cache_ = std::make_shared<BatchNormCache>();
        }
        
        cache_->input_cache = input;
        cache_->training = training;
        
        int batch_size = input.rows();
        
        if (training) {
            // Calcola media e varianza del batch
            cache_->batch_mean = input.colwise().mean();
            
            // Centratura
            cache_->x_centered = input.rowwise() - cache_->batch_mean.transpose();
            
            // Varianza (con correzione di Bessel per sample variance)
            cache_->batch_var = (cache_->x_centered.array().square()).colwise().sum() / (batch_size - 1);
            
            // Normalizzazione
            cache_->inv_std = (cache_->batch_var.array() + epsilon_).sqrt().inverse();
            cache_->x_norm = cache_->x_centered.array().rowwise() * cache_->inv_std.transpose().array();
            
            // Aggiorna statistiche running (media mobile)
            running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * cache_->batch_mean;
            running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * cache_->batch_var;
        } else {
            // Usa statistiche running per inference
            Eigen::MatrixXd centered = input.rowwise() - running_mean_.transpose();
            cache_->x_norm = centered.array().rowwise() / (running_var_.transpose().array() + epsilon_).sqrt();
        }
        
        // Scala e shift
        Eigen::MatrixXd output = cache_->x_norm.array().rowwise() * gamma_.transpose().array();
        output = output.rowwise() + beta_.transpose();
        
        cache_->output_cache = output;
        
        return output;
    }

    Eigen::MatrixXd BatchNormLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            throw std::runtime_error("BatchNormLayer: cache not initialized. Call forward first.");
        }
        
        auto* bn_cache = get_specific_cache().get();
        int batch_size = gradient.rows();
        
        // Gradiente per gamma e beta
        Eigen::VectorXd dgamma = (bn_cache->x_norm.array() * gradient.array()).colwise().sum();
        Eigen::VectorXd dbeta = gradient.colwise().sum();
        
        // Gradiente rispetto all'input normalizzato
        Eigen::MatrixXd dx_norm = gradient.array().rowwise() * gamma_.transpose().array();
        
        if (bn_cache->training) {
            // Gradiente rispetto alla varianza
            Eigen::VectorXd dvar = (dx_norm.array().rowwise() * 
                                   (-0.5) * bn_cache->inv_std.transpose().array().pow(3) * 
                                   bn_cache->x_centered.array()).colwise().sum();
            
            // Gradiente rispetto alla media
            Eigen::MatrixXd dx_centered1 = dx_norm.array().rowwise() * bn_cache->inv_std.transpose().array();
            Eigen::VectorXd dmean = (-dx_centered1.colwise().sum().transpose()).transpose() - 
                                    (2.0 / (batch_size - 1)) * dvar.asDiagonal() * 
                                    bn_cache->x_centered.colwise().sum().transpose();
            
            // Gradiente finale rispetto all'input
            Eigen::MatrixXd dx = dx_centered1 + 
                                (2.0 / (batch_size - 1)) * bn_cache->x_centered.array().rowwise() * dvar.transpose().array() +
                                (1.0 / batch_size) * dmean.transpose().replicate(batch_size, 1);
            
            // Aggiorna parametri
            gamma_ -= learning_rate * dgamma / batch_size;
            beta_ -= learning_rate * dbeta / batch_size;
            
            return dx;
        } else {
            // In inference mode, il gradiente passa direttamente
            return dx_norm;
        }
    }

    Eigen::MatrixXd BatchNormLayer::get_weights() const {
        Eigen::MatrixXd weights(gamma_.size(), 2);
        weights.col(0) = gamma_;
        weights.col(1) = beta_;
        return weights;
    }

    void BatchNormLayer::set_weights(const Eigen::MatrixXd& weights) {
        if (weights.rows() != gamma_.size() || weights.cols() != 2) {
            throw std::invalid_argument("BatchNormLayer: invalid weights dimension");
        }
        gamma_ = weights.col(0);
        beta_ = weights.col(1);
    }

    std::string BatchNormLayer::get_config() const {
        return "BatchNormLayer(epsilon=" + std::to_string(epsilon_) +
               ", momentum=" + std::to_string(momentum_) + ")";
    }

} // namespace layers