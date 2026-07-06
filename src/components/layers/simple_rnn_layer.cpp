#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/simple_rnn_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

// ============================================================================
// COSTRUTTORE
// ============================================================================

SimpleRNNLayer::SimpleRNNLayer(int units, int input_size, 
                             const std::string& activation,
                             bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation), 
      use_bias_(use_bias), return_sequences_(false) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "SimpleRNNLayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "SimpleRNNLayer");

    kernel_.resize(input_size, units);
    recurrent_.resize(units, units);
    
    double scale = std::sqrt(2.0 / (input_size + units));
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
    
    int total_rows = kernel_.rows() + recurrent_.rows();
    int total_cols = units_ + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units);
        bias_gradient_.setZero();
    }
    
    hidden_state_.resize(0, 0);
    cache_ = nullptr;
}

// ============================================================================
// DIMENSIONI
// ============================================================================

void SimpleRNNLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "SimpleRNNLayer");
    
    input_size_ = input_size;
    
    kernel_.resize(input_size_, units_);
    recurrent_.resize(units_, units_);
    
    if (use_bias_) {
        bias_.resize(units_);
        bias_.setZero();
    }
    
    // ALLOCAZIONE GRADIENTI PURA (Pesi e Bias separati)
    int total_rows = input_size_ + units_;
    weights_gradient_.resize(total_rows, units_);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units_);
        bias_gradient_.setZero();
    } else {
        bias_gradient_.resize(0);
    }
    
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
}

// ============================================================================
// INIZIALIZZAZIONE PESI
// ============================================================================

void SimpleRNNLayer::initialize_weights() {
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "must be > 0", "SimpleRNNLayer");
    ML_CHECK_PARAM(units_ > 0, "units", "must be > 0", "SimpleRNNLayer");
    
    double scale = std::sqrt(2.0 / input_size_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    for (int i = 0; i < kernel_.rows(); ++i) {
        for (int j = 0; j < kernel_.cols(); ++j) {
            kernel_(i, j) = dist(gen);
        }
    }
    
    double recurrent_scale = std::sqrt(1.0 / units_);
    std::normal_distribution<double> recurrent_dist(0.0, recurrent_scale);
    
    for (int i = 0; i < recurrent_.rows(); ++i) {
        for (int j = 0; j < recurrent_.cols(); ++j) {
            recurrent_(i, j) = recurrent_dist(gen);
        }
    }
}

// ============================================================================
// STATO
// ============================================================================

void SimpleRNNLayer::reset_state() {
    hidden_state_.resize(0, 0);
}

Eigen::MatrixXd SimpleRNNLayer::get_hidden_state() const {
    return hidden_state_;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

Eigen::MatrixXd SimpleRNNLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd SimpleRNNLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "SimpleRNNLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "SimpleRNNLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<SimpleRNNCache>();
    }
    
    int batch_size = input.rows();
    
    cache_->input_cache = input;
    cache_->output_cache.resize(batch_size, units_);
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

// ============================================================================
// BACKWARD PASS
// ============================================================================

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
    Eigen::MatrixXd prev_h = Eigen::MatrixXd::Zero(batch_size, units_);
    if (rnn_cache->hidden_states.size() > 1) {
        prev_h = rnn_cache->hidden_states[0];
    }
    
    // Calcolo gradiente rispetto alla pre-attivazione
    Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(z).array();
    
    // Calcolo dei gradienti dei pesi
    Eigen::MatrixXd dKernel = rnn_cache->input_cache.transpose() * dZ;
    Eigen::MatrixXd dRecurrent = prev_h.transpose() * dZ;
    
    // Assemblaggio geometrico rigido di weights_gradient_
    if (weights_gradient_.rows() != (input_size_ + units_) || weights_gradient_.cols() != units_) {
        weights_gradient_.resize(input_size_ + units_, units_);
    }
    weights_gradient_.block(0, 0, input_size_, units_) = dKernel;
    weights_gradient_.block(input_size_, 0, units_, units_) = dRecurrent;
    
    // Calcolo e isolamento del gradiente del Bias
    if (use_bias_) {
        if (bias_gradient_.size() != units_) {
            bias_gradient_.resize(units_);
        }
        bias_gradient_ = dZ.colwise().sum().transpose();
    }
    
    // Gradiente rispetto all'input da propagare indietro
    return dZ * kernel_.transpose();
}

// ============================================================================
// ATTIVAZIONI
// ============================================================================

// In simple_rnn_layer.cpp

Eigen::MatrixXd SimpleRNNLayer::apply_activation(const Eigen::MatrixXd& z) const {
    Eigen::MatrixXd z_clipped = z.cwiseMax(-10.0).cwiseMin(10.0);
    
    if (activation_ == "tanh") {
        return z_clipped.array().tanh().matrix();
    } else if (activation_ == "relu") {
        return z.cwiseMax(0.0);
    } else if (activation_ == "sigmoid") {
        return (1.0 / (1.0 + (-z_clipped).array().exp())).matrix();
    }
    return z_clipped.array().tanh().matrix();
}

Eigen::MatrixXd SimpleRNNLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
    // ⭐ Clip anche per la derivata!
    Eigen::MatrixXd z_clipped = z.cwiseMax(-10.0).cwiseMin(10.0);
    
    if (activation_ == "tanh") {
        Eigen::MatrixXd tanh_z = z_clipped.array().tanh().matrix();
        return (1.0 - tanh_z.array().square()).matrix();
    } else if (activation_ == "relu") {
        return (z.array() > 0.0).cast<double>().matrix();
    } else if (activation_ == "sigmoid") {
        Eigen::MatrixXd sig = (1.0 / (1.0 + (-z_clipped).array().exp())).matrix();
        return (sig.array() * (1.0 - sig.array())).matrix();
    }
    return (1.0 - z_clipped.array().tanh().square()).matrix();
}

// ============================================================================
// GETTER/SETTER PESI
// ============================================================================

Eigen::MatrixXd SimpleRNNLayer::get_weights() const {
    int total_rows = input_size_ + units_;
    Eigen::MatrixXd weights(total_rows, units_);
    weights.setZero();
    
    weights.block(0, 0, input_size_, units_) = kernel_;
    weights.block(input_size_, 0, units_, units_) = recurrent_;
    
    return weights;
}

void SimpleRNNLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (weights.rows() != (input_size_ + units_) || weights.cols() != units_) {
        ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "SimpleRNNLayer");
    }
    kernel_ = weights.block(0, 0, input_size_, units_);
    recurrent_ = weights.block(input_size_, 0, units_, units_);
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

int SimpleRNNLayer::get_parameter_count() const {
    return kernel_.size() + recurrent_.size() + (use_bias_ ? bias_.size() : 0);
}

// ============================================================================
// SERIALIZZAZIONE
// ============================================================================

void SimpleRNNLayer::serialize(std::ostream& out) const {
    int32_t input_size = input_size_;
    int32_t units = units_;
    out.write(reinterpret_cast<const char*>(&input_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&units), sizeof(int32_t));
    
    int32_t act_len = static_cast<int32_t>(activation_.size());
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(int32_t));
    out.write(activation_.c_str(), act_len);
    
    int8_t use_bias_flag = use_bias_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_bias_flag), sizeof(int8_t));
    
    int8_t return_seq_flag = return_sequences_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&return_seq_flag), sizeof(int8_t));
    
    int32_t rows = kernel_.rows();
    int32_t cols = kernel_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(kernel_.data()), rows * cols * sizeof(double));
    
    rows = recurrent_.rows();
    cols = recurrent_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(recurrent_.data()), rows * cols * sizeof(double));
    
    if (use_bias_) {
        rows = 1;
        cols = bias_.size();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(bias_.data()), rows * cols * sizeof(double));
    }
    
    if (!out.good()) {
        throw ml_exception::SerializationException("Failed to write SimpleRNNLayer", "SimpleRNNLayer");
    }
}

void SimpleRNNLayer::deserialize(std::istream& in) {
    int32_t input_size, units;
    in.read(reinterpret_cast<char*>(&input_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&units), sizeof(int32_t));
    
    if (input_size <= 0 || units <= 0) {
        throw ml_exception::DeserializationException(
            "Invalid dimensions: input_size=" + std::to_string(input_size) +
            ", units=" + std::to_string(units), "SimpleRNNLayer");
    }
    
    input_size_ = input_size;
    units_ = units;
    output_size_ = units;
    
    int32_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(int32_t));
    std::vector<char> buffer(act_len + 1, '\0');
    in.read(buffer.data(), act_len);
    activation_ = std::string(buffer.data());
    
    int8_t use_bias_flag;
    in.read(reinterpret_cast<char*>(&use_bias_flag), sizeof(int8_t));
    use_bias_ = (use_bias_flag != 0);
    
    int8_t return_seq_flag;
    in.read(reinterpret_cast<char*>(&return_seq_flag), sizeof(int8_t));
    return_sequences_ = (return_seq_flag != 0);
    
    kernel_.resize(input_size_, units_);
    recurrent_.resize(units_, units_);
    
    int total_rows = input_size_ + units_;
    int total_cols = units_ + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_.resize(units_);
        bias_gradient_.resize(units_);
        bias_gradient_.setZero();
    }
    
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
    
    int32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    
    if (rows != input_size_ || cols != units_) {
        throw ml_exception::DeserializationException("Kernel dimensions mismatch", "SimpleRNNLayer");
    }
    
    Eigen::MatrixXd loaded_kernel(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_kernel.data()), rows * cols * sizeof(double));
    kernel_ = loaded_kernel;
    
    in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    
    if (rows != units_ || cols != units_) {
        throw ml_exception::DeserializationException("Recurrent kernel dimensions mismatch", "SimpleRNNLayer");
    }
    
    Eigen::MatrixXd loaded_recurrent(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_recurrent.data()), rows * cols * sizeof(double));
    recurrent_ = loaded_recurrent;
    
    if (use_bias_) {
        in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        
        if (rows == 1 && cols == units_) {
            Eigen::MatrixXd loaded_bias(rows, cols);
            in.read(reinterpret_cast<char*>(loaded_bias.data()), rows * cols * sizeof(double));
            bias_ = loaded_bias.row(0);
        } else {
            throw ml_exception::DeserializationException("Bias dimensions mismatch", "SimpleRNNLayer");
        }
    }
    
    cache_ = std::make_shared<SimpleRNNCache>();
    
    if (!in.good() && !in.eof()) {
        throw ml_exception::DeserializationException("Stream error", "SimpleRNNLayer");
    }
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