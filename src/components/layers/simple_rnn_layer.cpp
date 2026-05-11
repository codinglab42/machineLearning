// src/components/layers/simple_rnn_layer.cpp
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/simple_rnn_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

SimpleRNNLayer::SimpleRNNLayer(int units, int input_size, 
                             const std::string& activation,
                             bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation), 
      use_bias_(use_bias) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "SimpleRNNLayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "SimpleRNNLayer");

    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
    
    double scale = std::sqrt(2.0 / (input_size + units));
    
    kernel_.resize(input_size, units);
    recurrent_.resize(units, units);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    for (int i = 0; i < kernel_.rows(); ++i) {
        for (int j = 0; j < kernel_.cols(); ++j) {
            kernel_(i, j) = dist(gen);
        }
    }
    
    for (int i = 0; i < recurrent_.rows(); ++i) {
        for (int j = 0; j < recurrent_.cols(); ++j) {
            recurrent_(i, j) = dist(gen);
        }
    }
    
    if (use_bias_) {
        bias_.resize(units);
        bias_.setZero();
    }
    
    // Inizializza gradienti
    weights_gradient_.resize(kernel_.rows() + recurrent_.rows(), 
                             kernel_.cols() + recurrent_.cols() + (use_bias_ ? 1 : 0));
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units);
        bias_gradient_.setZero();
    }
    
    hidden_state_.resize(0, 0);
    cache_ = nullptr;

    // VERIFICA dimensioni
    std::cout << "=== CONSTRUCTOR ===" << std::endl;
    std::cout << "kernel_ = " << kernel_.rows() << "x" << kernel_.cols() << std::endl;
    std::cout << "Expected: " << input_size << "x" << units << std::endl;
    std::cout << "recurrent_ = " << recurrent_.rows() << "x" << recurrent_.cols() << std::endl;
    std::cout << "Expected: " << units << "x" << units << std::endl;
}

void SimpleRNNLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "SimpleRNNLayer");

    std::cout << "=== set_input_shape ===" << std::endl;
    std::cout << "input_size param = " << input_size << std::endl;
    std::cout << "current input_size_ = " << input_size_ << std::endl;
    std::cout << "kernel_ before: " << kernel_.rows() << "x" << kernel_.cols() << std::endl;
    
    if (input_size_ != input_size) {
        input_size_ = input_size;
        
        // Ridimensiona i pesi del kernel
        Eigen::MatrixXd new_kernel(input_size, units_);
        double scale = std::sqrt(2.0 / (input_size + units_));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, scale);
        
        for (int i = 0; i < new_kernel.rows(); ++i) {
            for (int j = 0; j < new_kernel.cols(); ++j) {
                new_kernel(i, j) = dist(gen);
            }
        }
        kernel_ = new_kernel;

        std::cout << "kernel_ after resize: " << kernel_.rows() << "x" << kernel_.cols() << std::endl;
        
        // I pesi ricorrenti rimangono [units, units]
        // I bias rimangono [units]
    }
}

void SimpleRNNLayer::reset_state() {
    hidden_state_.resize(0, 0);
}

Eigen::MatrixXd SimpleRNNLayer::get_hidden_state() const {
    return hidden_state_;
}

Eigen::MatrixXd SimpleRNNLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd SimpleRNNLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "SimpleRNNLayer");

    std::cout << "=== forward DEBUG ===" << std::endl;
    std::cout << "kernel_ at forward start: " << kernel_.rows() << "x" << kernel_.cols() << std::endl;
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "SimpleRNNLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<SimpleRNNCache>();
    }
    
    int batch_size = input.rows();
    int timesteps = 1;
    
    cache_->input_cache = input;
    cache_->output_cache.resize(batch_size, units_);
    cache_->timesteps = timesteps;
    cache_->batch_size = batch_size;
    cache_->input_size = input_size_;
    cache_->hidden_size = units_;
    cache_->training = training;
    
    if (training) {
        cache_->hidden_states.clear();
        cache_->pre_activations.clear();
        cache_->z_values.clear();
    }
    
    if (hidden_state_.rows() != batch_size || hidden_state_.cols() != units_) {
        hidden_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
    }
    
    Eigen::MatrixXd z = input * kernel_ + hidden_state_ * recurrent_;
    
    if (use_bias_) {
        z.rowwise() += bias_.transpose();
    }
    
    if (training) {
        cache_->z_values.push_back(z);
    }
    
    hidden_state_ = apply_activation(z);
    
    if (training) {
        cache_->hidden_states.push_back(hidden_state_);
        cache_->pre_activations.push_back(z);
    }
    
    cache_->output_cache = hidden_state_;
    return hidden_state_;
}

Eigen::MatrixXd SimpleRNNLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("SimpleRNNLayer", "cache not initialized. Call forward first.");
    }
    
    auto rnn_cache = get_specific_cache();

    if (!rnn_cache->training) {
        return gradient;
    }
    
    int batch_size = rnn_cache->batch_size;
    
    if (gradient.rows() != batch_size || gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient",
            batch_size, units_,
            gradient.rows(), gradient.cols(), "SimpleRNNLayer");
    }
    
    const Eigen::MatrixXd& z = rnn_cache->z_values[0];
    const Eigen::MatrixXd& prev_h = (rnn_cache->hidden_states.size() > 1) ? 
                                     rnn_cache->hidden_states[0] : 
                                     Eigen::MatrixXd::Zero(batch_size, units_);
    
    Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(z).array();
    
    // Calcolo gradienti per kernel e recurrent
    Eigen::MatrixXd dKernel = rnn_cache->input_cache.transpose() * dZ;
    Eigen::MatrixXd dRecurrent = prev_h.transpose() * dZ;
    
    if (use_bias_) {
        // UNIFICA: get_weights() restituisce [input_size + units, units + 1]
        int total_rows = kernel_.rows() + recurrent_.rows();
        int total_cols = units_ + 1;
        
        Eigen::VectorXd dBias = dZ.colwise().sum();
        
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        
        // Kernel gradient
        weights_gradient_.block(0, 0, dKernel.rows(), dKernel.cols()) = dKernel;
        
        // Recurrent gradient
        weights_gradient_.block(kernel_.rows(), 0, dRecurrent.rows(), dRecurrent.cols()) = dRecurrent;
        
        // Bias gradient (ultima colonna)
        weights_gradient_.col(units_).head(dBias.size()) = dBias;
        
        bias_gradient_.resize(0);  // Non serve più separatamente
    } else {
        int total_rows = kernel_.rows() + recurrent_.rows();
        int total_cols = units_;
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        weights_gradient_.block(0, 0, dKernel.rows(), dKernel.cols()) = dKernel;
        weights_gradient_.block(kernel_.rows(), 0, dRecurrent.rows(), dRecurrent.cols()) = dRecurrent;
    }
    
    // Calcola dX - deve avere dimensioni [batch_size, input_size]
    Eigen::MatrixXd dX = dZ * kernel_.transpose();
    
    return dX;
}

Eigen::MatrixXd SimpleRNNLayer::apply_activation(const Eigen::MatrixXd& z) const {
    if (activation_ == "tanh") {
        return z.array().tanh();
    } else if (activation_ == "relu") {
        return z.cwiseMax(0.0);
    } else if (activation_ == "sigmoid") {
        return 1.0 / (1.0 + (-z).array().exp());
    } else if (activation_ == "linear") {
        return z;
    }
    return z.array().tanh();
}

Eigen::MatrixXd SimpleRNNLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
    if (activation_ == "tanh") {
        Eigen::MatrixXd tanh_z = z.array().tanh();
        return 1.0 - tanh_z.array().square();
    } else if (activation_ == "relu") {
        return (z.array() > 0.0).cast<double>();
    } else if (activation_ == "sigmoid") {
        Eigen::MatrixXd sig = 1.0 / (1.0 + (-z).array().exp());
        return sig.array() * (1.0 - sig.array());
    } else if (activation_ == "linear") {
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    }
    return 1.0 - z.array().tanh().square();
}

// include/components/layers/simple_rnn_layer.h
// Aggiungi/modifica questi metodi:

Eigen::MatrixXd SimpleRNNLayer::get_weights() const {
    int total_rows = kernel_.rows() + recurrent_.rows();
    
    if (use_bias_) {
        // Restituisce [input_size + units, units + 1]
        int total_cols = units_ + 1;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.setZero();
        
        // Kernel weights (input_size x units)
        weights.block(0, 0, kernel_.rows(), units_) = kernel_;
        
        // Recurrent weights (units x units)
        weights.block(kernel_.rows(), 0, recurrent_.rows(), units_) = recurrent_;
        
        // Bias (units) nell'ultima colonna
        weights.col(units_).head(bias_.size()) = bias_;
        
        return weights;
    } else {
        // Restituisce [input_size + units, units]
        int total_cols = units_;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.block(0, 0, kernel_.rows(), units_) = kernel_;
        weights.block(kernel_.rows(), 0, recurrent_.rows(), units_) = recurrent_;
        return weights;
    }
}

void SimpleRNNLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (use_bias_) {
        int expected_cols = units_ + 1;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "SimpleRNNLayer");
        }
        
        // Estrai kernel (input_size x units)
        kernel_ = weights.block(0, 0, input_size_, units_);
        
        // Estrai recurrent (units x units)
        recurrent_ = weights.block(input_size_, 0, units_, units_);
        
        // Estrai bias dall'ultima colonna
        bias_ = weights.col(units_).head(units_);
    } else {
        int expected_cols = units_;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "SimpleRNNLayer");
        }
        
        kernel_ = weights.block(0, 0, input_size_, units_);
        recurrent_ = weights.block(input_size_, 0, units_, units_);
    }
}

int SimpleRNNLayer::get_parameter_count() const {
    return kernel_.size() + recurrent_.size() + (use_bias_ ? bias_.size() : 0);
}

Eigen::VectorXd SimpleRNNLayer::get_biases() const {
    return bias_;
}

void SimpleRNNLayer::set_biases(const Eigen::VectorXd& biases) {
    if (biases.size() != units_) {
        ML_THROW_PARAMETER_ERROR("biases", "size must equal units", "SimpleRNNLayer");
    }
    bias_ = biases;
}

// src/components/layers/simple_rnn_layer.cpp

void SimpleRNNLayer::serialize(std::ostream& out) const {
    // Versione
    uint32_t version = get_version();
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Scrivi configurazione
    out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    
    bool has_bias = use_bias_;
    out.write(reinterpret_cast<const char*>(&has_bias), sizeof(bool));
    
    size_t act_len = activation_.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(activation_.c_str(), act_len);
    
    // Serializza usando get_weights() che ora include i bias
    Eigen::MatrixXd weights_to_save = get_weights();
    int rows = weights_to_save.rows();
    int cols = weights_to_save.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(weights_to_save.data()), 
            rows * cols * sizeof(double));
}

void SimpleRNNLayer::deserialize(std::istream& in) {
    // Leggi versione
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Leggi configurazione
    in.read(reinterpret_cast<char*>(&units_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    
    bool has_bias;
    in.read(reinterpret_cast<char*>(&has_bias), sizeof(bool));
    use_bias_ = has_bias;
    
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    activation_.resize(act_len);
    in.read(&activation_[0], act_len);
    
    // Leggi la matrice dei pesi
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    Eigen::MatrixXd loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    
    // Ridimensiona matrici
    kernel_.resize(input_size_, units_);
    recurrent_.resize(units_, units_);
    if (use_bias_) {
        bias_.resize(units_);
    }
    
    // Usa set_weights per decomporre
    set_weights(loaded_weights);
    
    // Inizializza gradienti
    int total_rows = kernel_.rows() + recurrent_.rows();
    int total_cols = kernel_.cols() + recurrent_.cols() + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units_);
        bias_gradient_.setZero();
    }
    
    hidden_state_.resize(0, 0);
    cache_ = std::make_shared<SimpleRNNCache>();
}

std::string SimpleRNNLayer::get_config() const {
    std::ostringstream oss;
    oss << "SimpleRNNLayer(units=" << units_
        << ", input_size=" << input_size_
        << ", activation=" << activation_
        << ", use_bias=" << (use_bias_ ? "true" : "false") << ")";
    return oss.str();
}

} // namespace layers