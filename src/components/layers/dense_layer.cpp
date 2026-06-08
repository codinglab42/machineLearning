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
    : units_(units), activation_(activation), use_bias_(use_bias) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "DenseLayer");
    cache_ = nullptr;
    
    // Inizializza gradienti vuoti
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

void DenseLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "DenseLayer");
    
    // Se non è cambiato, esci (utile per deserializzazione)
    if (input_size_ == input_size && weights_.size() > 0) {
        return;
    }
    
    input_size_ = input_size;
    output_size_ = units_;
    
    // Solo ridimensiona, non inizializzare ancora!
    weights_.resize(input_size, units_);
    
    if (use_bias_) {
        bias_.resize(units_);
        bias_.setZero();  // Bias a zero di default
    }
    
    // ⭐ IMPORTANTE: weights_gradient_ deve avere DIMENSIONI CORRETTE
    if (use_bias_) {
        // Quando c'è bias, weights_gradient_ deve avere una colonna in più
        weights_gradient_.resize(input_size, units_ + 1);
    } else {
        weights_gradient_.resize(input_size, units_);
    }
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units_);
        bias_gradient_.setZero();
    }
    
    // Cache
    if (!cache_) {
        cache_ = std::make_shared<DenseCache>();
    }
}

void DenseLayer::initialize_weights() {
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "must be > 0", "DenseLayer");
    ML_CHECK_PARAM(units_ > 0, "units", "must be > 0", "DenseLayer");
    
    // Inizializzazione Xavier
    double scale = std::sqrt(2.0 / (input_size_ + units_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    for (int i = 0; i < weights_.rows(); ++i) {
        for (int j = 0; j < weights_.cols(); ++j) {
            weights_(i, j) = dist(gen);
        }
    }
    
    if (use_bias_) { bias_.setZero(); }
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
    
    // CALCOLA dZ - per activation "linear" la derivata è 1
    Eigen::MatrixXd dZ;
    if (activation_ == "linear") {
        dZ = gradient;  // derivata di linear è 1
    } else {
        dZ = gradient.array() * apply_activation_derivative(cache_->z_cache).array();
    }
    
    // Calcola gradienti per pesi
    Eigen::MatrixXd weight_grad = cache_->input_cache.transpose() * dZ;
    
    if (use_bias_) {
        Eigen::VectorXd bias_grad = dZ.colwise().sum();
        
        // ⭐ Non ridimensionare weights_gradient_ qui!
        // Deve già avere le dimensioni corrette da set_input_shape()
        if (weights_gradient_.rows() != weight_grad.rows() || 
            weights_gradient_.cols() != weight_grad.cols() + 1) {
            weights_gradient_.resize(weight_grad.rows(), weight_grad.cols() + 1);
        }
        weights_gradient_.leftCols(weight_grad.cols()) = weight_grad;
        weights_gradient_.col(weight_grad.cols()) = bias_grad;
        
        bias_gradient_ = bias_grad;
    } else {
        if (weights_gradient_.rows() != weight_grad.rows() || 
            weights_gradient_.cols() != weight_grad.cols()) {
            weights_gradient_.resize(weight_grad.rows(), weight_grad.cols());
        }
        weights_gradient_ = weight_grad;
    }
    
    // Calcola gradiente per input
    Eigen::MatrixXd dX = dZ * weights_.transpose();
    
    return dX;
}

void DenseLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (use_bias_) {
        // weights deve avere colonne = units + 1 (ultima colonna è bias)
        if (weights.cols() != units_ + 1) {
            ML_THROW_PARAMETER_ERROR("weights", 
                "expected " + std::to_string(units_ + 1) + " columns, got " + std::to_string(weights.cols()), 
                "DenseLayer");
        }
        weights_ = weights.leftCols(units_);
        bias_ = weights.col(units_);
    } else {
        if (weights.cols() != units_) {
            ML_THROW_PARAMETER_ERROR("weights", 
                "expected " + std::to_string(units_) + " columns, got " + std::to_string(weights.cols()), 
                "DenseLayer");
        }
        weights_ = weights;
    }
}

Eigen::MatrixXd DenseLayer::get_weights() const { 
    if (use_bias_) {
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
    // Clip per stabilità
    Eigen::MatrixXd z_clipped = z.cwiseMax(-10.0).cwiseMin(10.0);
    
    if (activation_ == "relu") {
        return z.cwiseMax(0.0);
    } else if (activation_ == "sigmoid") {
        return (1.0 / (1.0 + (-z_clipped).array().exp())).matrix();
    } else if (activation_ == "tanh") {
        return z_clipped.array().tanh().matrix();
    } else if (activation_ == "softmax") {
        Eigen::MatrixXd exp_z = z_clipped.array().exp().matrix();
        Eigen::VectorXd sum = exp_z.rowwise().sum();
        return (exp_z.array().colwise() / sum.array()).matrix();
    } else if (activation_ == "linear") {
        return z;
    }
    return z.cwiseMax(0.0);
}

Eigen::MatrixXd DenseLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
    // ⭐ Clip anche per la derivata!
    Eigen::MatrixXd z_clipped = z.cwiseMax(-10.0).cwiseMin(10.0);
    
    if (activation_ == "relu") {
        return (z.array() > 0.0).cast<double>().matrix();
    } else if (activation_ == "sigmoid") {
        Eigen::MatrixXd sig = (1.0 / (1.0 + (-z_clipped).array().exp())).matrix();
        return (sig.array() * (1.0 - sig.array())).matrix();
    } else if (activation_ == "tanh") {
        Eigen::MatrixXd tanh_z = z_clipped.array().tanh().matrix();
        return (1.0 - tanh_z.array().square()).matrix();
    } else if (activation_ == "linear") {
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    } else if (activation_ == "softmax") {
        // Per softmax, la derivata è Jacobiana, ma per semplicità
        return Eigen::MatrixXd::Ones(z.rows(), z.cols());
    }
    return (z.array() > 0.0).cast<double>().matrix();
}

void DenseLayer::serialize(std::ostream& out) const {
    // 1. Scrivi dimensioni (int32_t)
    int32_t input_size = input_size_;
    int32_t units = units_;
    out.write(reinterpret_cast<const char*>(&input_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&units), sizeof(int32_t));
    
    // 2. Scrivi activation (int32_t per lunghezza)
    int32_t act_len = static_cast<int32_t>(activation_.size());
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(int32_t));
    out.write(activation_.c_str(), act_len);
    
    // 3. Scrivi use_bias (int8_t)
    int8_t use_bias_flag = use_bias_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_bias_flag), sizeof(int8_t));
    
    // 4. Scrivi weights
    int32_t rows = weights_.rows();
    int32_t cols = weights_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(weights_.data()), 
             rows * cols * sizeof(double));
    
    // 5. Scrivi bias (se presente)
    if (use_bias_) {
        int32_t bias_rows = bias_.rows();
        int32_t bias_cols = bias_.cols();
        out.write(reinterpret_cast<const char*>(&bias_rows), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&bias_cols), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(bias_.data()), 
                 bias_rows * bias_cols * sizeof(double));
    }
    
    if (!out.good()) {
        ML_THROW_SERIALIZATION_ERROR("Failed to write DenseLayer", "DenseLayer");
    }
}


void DenseLayer::deserialize(std::istream& in) {
    // 1. Leggi dimensioni (usa int32_t)
    int32_t input_size, units;
    in.read(reinterpret_cast<char*>(&input_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&units), sizeof(int32_t));
    
    if (input_size <= 0 || units <= 0) {
        ML_THROW_DESERIALIZATION_ERROR(
            "Invalid dimensions: input_size=" + std::to_string(input_size) +
            ", units=" + std::to_string(units), "DenseLayer");
    }
    
    input_size_ = input_size;
    units_ = units;
    output_size_ = units;  // Importante!
    
    // 2. Leggi activation (usa int32_t per lunghezza)
    int32_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(int32_t));
    
    if (act_len < 0 || act_len > 1024) {
        ML_THROW_DESERIALIZATION_ERROR(
            "Invalid activation length: " + std::to_string(act_len), "DenseLayer");
    }
    
    std::vector<char> buffer(act_len + 1, '\0');
    in.read(buffer.data(), act_len);
    activation_ = std::string(buffer.data());
    
    // 3. Leggi use_bias (usa int8_t invece di bool per dimensione fissa)
    int8_t use_bias_flag;
    in.read(reinterpret_cast<char*>(&use_bias_flag), sizeof(int8_t));
    use_bias_ = (use_bias_flag != 0);
    
    // 4. Alloca le matrici con le dimensioni corrette PRIMA di leggere i pesi
    // Alloca strutture
    weights_.resize(input_size_, units_);
    weights_gradient_.resize(input_size_, units_);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_.resize(units_);
        bias_gradient_.resize(units_);
        bias_gradient_.setZero();
    }
    
    // 5. Leggi la matrice dei pesi
    int32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    
    // Validazione dimensioni
    if (rows != input_size_ || cols != units_) {
        ML_THROW_DESERIALIZATION_ERROR(
            "Weight dimensions mismatch: expected " + 
            std::to_string(input_size_) + "x" + std::to_string(units_) +
            ", got " + std::to_string(rows) + "x" + std::to_string(cols), "DenseLayer");
    }
    
    Eigen::MatrixXd loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    
    // Assegna direttamente, senza chiamare set_weights
    if (use_bias_) {
        // Se loaded_weights ha una colonna in più, contiene anche bias
        if (cols == units_ + 1) {
            weights_ = loaded_weights.leftCols(units_);
            bias_ = loaded_weights.col(units_);
        } else {
            weights_ = loaded_weights;
            // bias_ rimane zero
        }
    } else {
        weights_ = loaded_weights;
    }
    
    // 6. Leggi bias (se presente)
    // Leggi bias (solo se serializzato separatamente)
    if (use_bias_) {
        int32_t bias_rows, bias_cols;
        if (in.peek() != EOF) {
            in.read(reinterpret_cast<char*>(&bias_rows), sizeof(int32_t));
            in.read(reinterpret_cast<char*>(&bias_cols), sizeof(int32_t));
            
            if (bias_rows == 1 && bias_cols == units_) {
                Eigen::MatrixXd loaded_bias(bias_rows, bias_cols);
                in.read(reinterpret_cast<char*>(loaded_bias.data()), bias_rows * bias_cols * sizeof(double));
                bias_ = loaded_bias.row(0);
            }
        }
    }
    
    // 7. Ricrea cache
    cache_ = std::make_shared<DenseCache>();
    
    // 8. Verifica finale
    if (!in.good() && !in.eof()) {
        ML_THROW_SERIALIZATION_ERROR("Stream error while reading DenseLayer", "DenseLayer");
    }
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