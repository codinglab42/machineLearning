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
        int feature_size = input.cols();
        
        if (training) {
            // Calcola media del batch
            cache_->batch_mean = Eigen::VectorXd::Zero(feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    cache_->batch_mean(j) += input(i, j);
                }
            }
            for (int j = 0; j < feature_size; ++j) {
                cache_->batch_mean(j) /= batch_size;
            }
            
            // Centratura
            cache_->x_centered.resize(batch_size, feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    cache_->x_centered(i, j) = input(i, j) - cache_->batch_mean(j);
                }
            }
            
            // Calcola varianza
            cache_->batch_var = Eigen::VectorXd::Zero(feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    cache_->batch_var(j) += cache_->x_centered(i, j) * cache_->x_centered(i, j);
                }
            }
            for (int j = 0; j < feature_size; ++j) {
                cache_->batch_var(j) /= (batch_size - 1);
            }
            
            // Calcola inv_std
            cache_->inv_std.resize(feature_size);
            for (int j = 0; j < feature_size; ++j) {
                cache_->inv_std(j) = 1.0 / std::sqrt(cache_->batch_var(j) + epsilon_);
            }
            
            // Normalizza
            cache_->x_norm.resize(batch_size, feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    cache_->x_norm(i, j) = cache_->x_centered(i, j) * cache_->inv_std(j);
                }
            }
            
            // Aggiorna statistiche running
            for (int j = 0; j < feature_size; ++j) {
                running_mean_(j) = momentum_ * running_mean_(j) + (1.0 - momentum_) * cache_->batch_mean(j);
                running_var_(j) = momentum_ * running_var_(j) + (1.0 - momentum_) * cache_->batch_var(j);
            }
        } else {
            // Usa statistiche running per inference
            cache_->x_norm.resize(batch_size, feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    double centered = input(i, j) - running_mean_(j);
                    cache_->x_norm(i, j) = centered / std::sqrt(running_var_(j) + epsilon_);
                }
            }
        }
        
        // Scala e shift
        Eigen::MatrixXd output(batch_size, feature_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                output(i, j) = cache_->x_norm(i, j) * gamma_(j) + beta_(j);
            }
        }
        
        cache_->output_cache = output;
        return output;
    }

    Eigen::MatrixXd BatchNormLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("BatchNormLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "BatchNormLayer");
        
        auto bn_cache = get_specific_cache();
        
        if (gradient.rows() != bn_cache->input_cache.rows() || 
            gradient.cols() != bn_cache->input_cache.cols()) {
            ML_THROW_DIMENSION_MISMATCH("backward gradient",
                bn_cache->input_cache.rows(), bn_cache->input_cache.cols(),
                gradient.rows(), gradient.cols(), "BatchNormLayer");
        }
        
        int batch_size = gradient.rows();
        int feature_size = gradient.cols();
        
        // Gradiente per gamma e beta
        Eigen::VectorXd dgamma = Eigen::VectorXd::Zero(feature_size);
        Eigen::VectorXd dbeta = Eigen::VectorXd::Zero(feature_size);
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                dgamma(j) += bn_cache->x_norm(i, j) * gradient(i, j);
                dbeta(j) += gradient(i, j);
            }
        }
        
        // Gradiente rispetto all'input normalizzato
        Eigen::MatrixXd dx_norm(batch_size, feature_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < feature_size; ++j) {
                dx_norm(i, j) = gradient(i, j) * gamma_(j);
            }
        }
        
        if (bn_cache->training) {
            // Gradiente rispetto alla varianza
            Eigen::VectorXd dvar = Eigen::VectorXd::Zero(feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    double inv_std_j = bn_cache->inv_std(j);
                    dvar(j) += dx_norm(i, j) * (-0.5) * std::pow(inv_std_j, 3) * bn_cache->x_centered(i, j);
                }
            }
            
            // Gradiente rispetto alla media (prima parte)
            Eigen::VectorXd dmean = Eigen::VectorXd::Zero(feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    dmean(j) += -dx_norm(i, j) * bn_cache->inv_std(j);
                }
            }
            
            // Seconda parte del gradiente rispetto alla media
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    dmean(j) += (-2.0 / (batch_size - 1)) * dvar(j) * bn_cache->x_centered(i, j);
                }
            }
            
            // Gradiente finale rispetto all'input
            Eigen::MatrixXd dx(batch_size, feature_size);
            for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < feature_size; ++j) {
                    dx(i, j) = dx_norm(i, j) * bn_cache->inv_std(j) +
                               (2.0 / (batch_size - 1)) * bn_cache->x_centered(i, j) * dvar(j) +
                               (1.0 / batch_size) * dmean(j);
                }
            }
            
            // Aggiorna parametri
            for (int j = 0; j < feature_size; ++j) {
                gamma_(j) -= learning_rate * dgamma(j) / batch_size;
                beta_(j) -= learning_rate * dbeta(j) / batch_size;
            }
            
            return dx;
        }
        
        return dx_norm;
    }

    Eigen::MatrixXd BatchNormLayer::get_weights() const {
        Eigen::MatrixXd weights(gamma_.size(), 2);
        for (int i = 0; i < gamma_.size(); ++i) {
            weights(i, 0) = gamma_(i);
            weights(i, 1) = beta_(i);
        }
        return weights;
    }

    void BatchNormLayer::set_weights(const Eigen::MatrixXd& weights) {
        if (weights.rows() != gamma_.size() || weights.cols() != 2) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "BatchNormLayer");
        }
        for (int i = 0; i < gamma_.size(); ++i) {
            gamma_(i) = weights(i, 0);
            beta_(i) = weights(i, 1);
        }
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