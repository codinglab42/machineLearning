// src/components/layers/dense_layer.cpp
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/dense_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

DenseLayer::DenseLayer(int units, const std::string& activation, bool use_bias)
    : units_(units), activation_(activation), use_bias_(use_bias), input_size_(0) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "DenseLayer");
    cache_ = nullptr;
    
    // Inizializza gradienti vuoti
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

void DenseLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "DenseLayer");
    
    input_size_ = input_size;
    
    // Inizializzazione Xavier
    double scale = std::sqrt(6.0 / (input_size + units_));
    
    weights_.resize(input_size, units_);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-scale, scale);
    
    for (int i = 0; i < weights_.rows(); ++i) {
        for (int j = 0; j < weights_.cols(); ++j) {
            weights_(i, j) = dist(gen);
        }
    }
    
    if (use_bias_) {
        bias_.setZero(units_);
    }
    
    // Inizializza cache
    if (!cache_) {
        cache_ = std::make_shared<DenseCache>();
    }
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "DenseLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input", 
            input.rows(), input_size_,
            input.rows(), input.cols(), "DenseLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<DenseCache>();
    }
    
    // Calcolo forward
    Eigen::MatrixXd z = input * weights_;
    
    if (use_bias_) {
        z.rowwise() += bias_.transpose();
    }
    
    Eigen::MatrixXd output = apply_activation(z);
    
    // Salva nella cache (solo dati temporanei!)
    cache_->input_cache = input;
    cache_->z_cache = z;
    cache_->output_cache = output;
    
    return output;
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("DenseLayer", "cache not initialized. Call forward first.");
    }
    
    if (gradient.rows() != cache_->input_cache.rows() || 
        gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient",
            cache_->input_cache.rows(), units_,
            gradient.rows(), gradient.cols(), "DenseLayer");
    }
    
    // Calcola dZ = gradient * derivata_attivazione
    Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(cache_->z_cache).array();
    
    // Calcola gradienti per pesi
    Eigen::MatrixXd weight_grad = cache_->input_cache.transpose() * dZ;
    
    if (use_bias_) {
        // Calcola gradiente per bias
        Eigen::VectorXd bias_grad = dZ.colwise().sum();
        
        // UNIFICA: gradiente pesi + bias in un'unica matrice [input_size, units + 1]
        weights_gradient_.resize(weight_grad.rows(), weight_grad.cols() + 1);
        weights_gradient_.leftCols(weight_grad.cols()) = weight_grad;
        weights_gradient_.col(weight_grad.cols()) = bias_grad;
    } else {
        weights_gradient_ = weight_grad;
    }
    
    // Calcola gradiente per input (da propagare indietro)
    Eigen::MatrixXd dX = dZ * weights_.transpose();
    
    return dX;
}

void DenseLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (use_bias_) {
        if (weights.cols() != weights_.cols() + 1) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "DenseLayer");
        }
        weights_ = weights.leftCols(weights_.cols());
        bias_ = weights.col(weights_.cols());
    } else {
        if (weights.cols() != weights_.cols()) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "DenseLayer");
        }
        weights_ = weights;
    }
}

Eigen::MatrixXd DenseLayer::get_weights() const { 
    if (use_bias_) {
        // Restituisce [input_size, units + 1] - pesi + bias nell'ultima colonna
        Eigen::MatrixXd weights_with_bias(weights_.rows(), weights_.cols() + 1);
        weights_with_bias.leftCols(weights_.cols()) = weights_;
        weights_with_bias.col(weights_.cols()) = bias_;
        return weights_with_bias;
    }
    return weights_;
}



void DenseLayer::set_biases(const Eigen::VectorXd& biases) {
    if (use_bias_) {
        if (biases.size() != units_) {
            ML_THROW_PARAMETER_ERROR("biases", "invalid size", "DenseLayer");
        }
        bias_ = biases;
    }
}

int DenseLayer::get_parameter_count() const {
    return weights_.size() + (use_bias_ ? bias_.size() : 0);
}

Eigen::MatrixXd DenseLayer::apply_activation(const Eigen::MatrixXd& z) const {
    if (activation_ == "relu") {
        return z.cwiseMax(0.0);
    } else if (activation_ == "sigmoid") {
        return 1.0 / (1.0 + (-z).array().exp());
    } else if (activation_ == "tanh") {
        return z.array().tanh();
    } else if (activation_ == "softmax") {
        Eigen::MatrixXd exp_z = z.array().exp();
        Eigen::VectorXd sum = exp_z.rowwise().sum();
        return exp_z.array().colwise() / sum.array();
    } else if (activation_ == "linear") {
        return z;
    }
    return z.cwiseMax(0.0);
}

Eigen::MatrixXd DenseLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
    if (activation_ == "relu") {
        return (z.array() > 0.0).cast<double>();
    } else if (activation_ == "sigmoid") {
        Eigen::MatrixXd sig = 1.0 / (1.0 + (-z).array().exp());
        return sig.array() * (1.0 - sig.array());
    } else if (activation_ == "tanh") {
        Eigen::MatrixXd tanh_z = z.array().tanh();
        return 1.0 - tanh_z.array().square();
    } else if (activation_ == "linear") {
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    } else if (activation_ == "softmax") {
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    }
    return (z.array() > 0.0).cast<double>();
}

void DenseLayer::serialize(std::ostream& out) const {
    // Usiamo un formato binario semplice e robusto
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
    
    size_t act_len = activation_.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(activation_.c_str(), act_len);
    
    out.write(reinterpret_cast<const char*>(&use_bias_), sizeof(bool));
    
    // Serializza pesi (solo i pesi, non i gradienti!)
    Eigen::MatrixXd weights_to_save = get_weights();
    int rows = weights_to_save.rows();
    int cols = weights_to_save.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(weights_to_save.data()), rows * cols * sizeof(double));
    
    if (use_bias_) {
        int bias_size = bias_.size();
        out.write(reinterpret_cast<const char*>(&bias_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(bias_.data()), bias_size * sizeof(double));
    }
}

void DenseLayer::deserialize(std::istream& in) {
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    in.read(reinterpret_cast<char*>(&units_), sizeof(int));
    
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    activation_.resize(act_len);
    in.read(&activation_[0], act_len);
    
    in.read(reinterpret_cast<char*>(&use_bias_), sizeof(bool));
    
    // IMPORTANTE: Inizializza le matrici interne PRIMA di set_weights!
    weights_.resize(input_size_, units_);
    if (use_bias_) {
        bias_.resize(units_);
    }
    
    // Leggi la matrice dei pesi
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));

    Eigen::MatrixXd loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    
    set_weights(loaded_weights);
    
    // Ricrea cache
    cache_ = std::make_shared<DenseCache>();
}

std::string DenseLayer::get_config() const {
    std::ostringstream oss;
    oss << "DenseLayer(units=" << units_ 
        << ", activation=" << activation_
        << ", use_bias=" << (use_bias_ ? "true" : "false")
        << ", input_size=" << input_size_ << ")";
    return oss.str();
}

} // namespace layers