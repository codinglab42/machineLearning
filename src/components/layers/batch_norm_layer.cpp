#include "components/layers/batch_norm_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

    BatchNormLayer::BatchNormLayer(double epsilon, double momentum)
        : epsilon_(epsilon), momentum_(momentum), input_size_(0) {
        
        ML_CHECK_PARAM(epsilon > 0, "epsilon", "must be > 0", "BatchNormLayer");
        ML_CHECK_PARAM(momentum > 0 && momentum < 1, "momentum", "must be in (0, 1)", "BatchNormLayer");
        cache_ = nullptr;
    }

    void BatchNormLayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "BatchNormLayer");
        
        input_size_ = input_size;
        
        gamma_ = Eigen::VectorXd::Ones(input_size);
        beta_ = Eigen::VectorXd::Zero(input_size);
        running_mean_ = Eigen::VectorXd::Zero(input_size);
        running_var_ = Eigen::VectorXd::Ones(input_size);
    }

    Eigen::MatrixXd BatchNormLayer::forward(const Eigen::MatrixXd& input) {
        return forward(input, false);
    }

    Eigen::MatrixXd BatchNormLayer::forward(const Eigen::MatrixXd& input, bool training) {
        ML_CHECK_NOT_EMPTY(input, "input", "BatchNormLayer");
        
        if (input.cols() != input_size_) {
            ML_THROW_DIMENSION_MISMATCH("forward input",
                input.rows(), input_size_,
                input.rows(), input.cols(), "BatchNormLayer");
        }
        
        if (!cache_) {
            cache_ = std::make_shared<BatchNormCache>();
        }
        
        cache_->input_cache = input;
        cache_->training = training;
        
        int batch_size = input.rows();
        
        if (training) {
            cache_->batch_mean = input.colwise().mean();
            cache_->x_centered = input.rowwise() - cache_->batch_mean.transpose();
            cache_->batch_var = (cache_->x_centered.array().square()).colwise().sum() / (batch_size - 1);
            cache_->inv_std = (cache_->batch_var.array() + epsilon_).sqrt().inverse();
            cache_->x_norm = cache_->x_centered.array().rowwise() * cache_->inv_std.transpose().array();
            
            running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * cache_->batch_mean;
            running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * cache_->batch_var;
        } else {
            Eigen::MatrixXd centered = input.rowwise() - running_mean_.transpose();
            cache_->x_norm = centered.array().rowwise() / (running_var_.transpose().array() + epsilon_).sqrt();
        }
        
        Eigen::MatrixXd output = cache_->x_norm.array().rowwise() * gamma_.transpose().array();
        output = output.rowwise() + beta_.transpose();
        
        cache_->output_cache = output;
        return output;
    }

    Eigen::MatrixXd BatchNormLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("BatchNormLayer", "cache not initialized. Call forward first.");
    }
    
    auto bn_cache = get_specific_cache();
    int batch_size = gradient.rows();
    
    // 1. Gradiente per i parametri (gamma e beta)
    // dgamma = sum(gradient * x_norm)
    Eigen::VectorXd dgamma = (bn_cache->x_norm.array() * gradient.array()).matrix().colwise().sum();
    Eigen::VectorXd dbeta = gradient.colwise().sum();
    
    // 2. Gradiente rispetto all'input normalizzato (dx_norm)
    Eigen::MatrixXd dx_norm = gradient.array().rowwise() * gamma_.transpose().array();
    
    if (bn_cache->training) {
        // Prepariamo i termini intermedi come array per evitare problemi di broadcasting
        Eigen::ArrayXd inv_std = bn_cache->inv_std.array();
        
        // dvar = sum(dx_norm * (x - mean) * -0.5 * inv_std^3)
        Eigen::VectorXd dvar = (dx_norm.array() * bn_cache->x_centered.array()).colwise().sum();
        dvar.array() *= -0.5 * inv_std.pow(3);
        
        // dx_centered1 = dx_norm * inv_std
        Eigen::MatrixXd dx_centered1 = dx_norm.array().rowwise() * inv_std.transpose();
        
        // dmean = sum(-dx_centered1) + dvar * sum(-2 * (x - mean)) / (N-1)
        // Nota: sum(x - mean) è matematicamente 0, ma lo calcoliamo per stabilità numerica
        Eigen::VectorXd dmean = -dx_centered1.colwise().sum();
        Eigen::VectorXd sum_centered = bn_cache->x_centered.colwise().sum();
        dmean.array() += dvar.array() * (sum_centered.array() * -2.0 / (batch_size - 1));
        
        // 3. Gradiente finale rispetto all'input (dx)
        Eigen::MatrixXd dx = dx_centered1;
        // Aggiungiamo il contributo di dvar
        dx.array() += bn_cache->x_centered.array().rowwise() * (dvar.array() * 2.0 / (batch_size - 1)).transpose();
        // Aggiungiamo il contributo di dmean
        dx.array() += dmean.transpose().replicate(batch_size, 1).array() / batch_size;
        
        // Aggiornamento pesi (Gamma e Beta)
        gamma_ -= learning_rate * dgamma / batch_size;
        beta_ -= learning_rate * dbeta / batch_size;
        
        return dx;
    }
    
    // In modalità inference, il gradiente è semplicemente la propagazione attraverso i valori fissi
    Eigen::ArrayXd running_inv_std = (running_var_.array() + epsilon_).sqrt().inverse();
    return dx_norm.array().rowwise() * running_inv_std.transpose();
}
    Eigen::MatrixXd BatchNormLayer::get_weights() const {
        Eigen::MatrixXd weights(gamma_.size(), 2);
        weights.col(0) = gamma_;
        weights.col(1) = beta_;
        return weights;
    }

    void BatchNormLayer::set_weights(const Eigen::MatrixXd& weights) {
        if (weights.rows() != gamma_.size() || weights.cols() != 2) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "BatchNormLayer");
        }
        gamma_ = weights.col(0);
        beta_ = weights.col(1);
    }

    int BatchNormLayer::get_parameter_count() const {
        return gamma_.size() + beta_.size();
    }

    void BatchNormLayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&momentum_), sizeof(double));
        
        for (int i = 0; i < gamma_.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&gamma_(i)), sizeof(double));
        }
        for (int i = 0; i < beta_.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&beta_(i)), sizeof(double));
        }
        for (int i = 0; i < running_mean_.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&running_mean_(i)), sizeof(double));
        }
        for (int i = 0; i < running_var_.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&running_var_(i)), sizeof(double));
        }
    }

    void BatchNormLayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        in.read(reinterpret_cast<char*>(&epsilon_), sizeof(double));
        in.read(reinterpret_cast<char*>(&momentum_), sizeof(double));
        
        gamma_.resize(input_size_);
        beta_.resize(input_size_);
        running_mean_.resize(input_size_);
        running_var_.resize(input_size_);
        
        for (int i = 0; i < gamma_.size(); ++i) {
            in.read(reinterpret_cast<char*>(&gamma_(i)), sizeof(double));
        }
        for (int i = 0; i < beta_.size(); ++i) {
            in.read(reinterpret_cast<char*>(&beta_(i)), sizeof(double));
        }
        for (int i = 0; i < running_mean_.size(); ++i) {
            in.read(reinterpret_cast<char*>(&running_mean_(i)), sizeof(double));
        }
        for (int i = 0; i < running_var_.size(); ++i) {
            in.read(reinterpret_cast<char*>(&running_var_(i)), sizeof(double));
        }
    }

    std::string BatchNormLayer::get_config() const {
        std::ostringstream oss;
        oss << "BatchNormLayer(epsilon=" << epsilon_
            << ", momentum=" << momentum_
            << ", input_size=" << input_size_ << ")";
        return oss.str();
    }

} // namespace layers