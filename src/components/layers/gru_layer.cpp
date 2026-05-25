#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/gru_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

// ============================================================================
// COSTRUTTORI
// ============================================================================

GRULayer::GRULayer(int units, int input_size, 
                   const std::string& activation,
                   const std::string& recurrent_activation,
                   bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation),
      recurrent_activation_(recurrent_activation), use_bias_(use_bias) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "GRULayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
    
    double scale = std::sqrt(2.0 / (input_size + units));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    auto init = [&](Eigen::MatrixXd& mat, int rows, int cols) {
        mat.resize(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                mat(i, j) = dist(gen);
    };
    
    // Kernel weights (input to hidden) per i 3 gate
    init(kernel_r, input_size, units);
    init(kernel_z, input_size, units);
    init(kernel_h, input_size, units);
    
    // Recurrent weights (hidden to hidden) per i 3 gate
    init(recurrent_r, units, units);
    init(recurrent_z, units, units);
    init(recurrent_h, units, units);
    
    if (use_bias_) {
        bias_r.setZero(units);
        bias_z.setZero(units);
        bias_h.setZero(units);
    }
    
    hidden_state_.resize(0, 0);
    cache_ = nullptr;
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
}

// ============================================================================
// DIMENSIONI
// ============================================================================

void GRULayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
    
    if (input_size_ == input_size && kernel_r.size() > 0) {
        return;
    }
    
    input_size_ = input_size;
    output_size_ = units_;
    
    // Ridimensiona kernel weights
    kernel_r.resize(input_size_, units_);
    kernel_z.resize(input_size_, units_);
    kernel_h.resize(input_size_, units_);
    
    // Ridimensiona recurrent weights
    recurrent_r.resize(units_, units_);
    recurrent_z.resize(units_, units_);
    recurrent_h.resize(units_, units_);
    
    // Ridimensiona bias
    if (use_bias_) {
        bias_r.resize(units_);
        bias_z.resize(units_);
        bias_h.resize(units_);
        bias_r.setZero();
        bias_z.setZero();
        bias_h.setZero();
    }
    
    // Ridimensiona gradienti
    int total_rows = (input_size_ + units_) * 3;
    int total_cols = 3 * units_ + (use_bias_ ? 3 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(3 * units_);
        bias_gradient_.setZero();
    }
    
    // Ridimensiona stato nascosto
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
}

// ============================================================================
// INIZIALIZZAZIONE PESI
// ============================================================================

void GRULayer::initialize_weights() {
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "must be > 0", "GRULayer");
    ML_CHECK_PARAM(units_ > 0, "units", "must be > 0", "GRULayer");
    
    double scale = std::sqrt(2.0 / input_size_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    auto init = [&](Eigen::MatrixXd& mat) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = dist(gen);
    };
    
    init(kernel_r);
    init(kernel_z);
    init(kernel_h);
    
    double rec_scale = std::sqrt(1.0 / units_);
    std::normal_distribution<double> rec_dist(0.0, rec_scale);
    
    auto init_rec = [&](Eigen::MatrixXd& mat) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = rec_dist(gen);
    };
    
    init_rec(recurrent_r);
    init_rec(recurrent_z);
    init_rec(recurrent_h);
    
    if (use_bias_) {
        bias_r.setZero();
        bias_z.setZero();
        bias_h.setZero();
    }
}

// ============================================================================
// STATO
// ============================================================================

void GRULayer::reset_state() {
    hidden_state_.resize(0, 0);
    if (cache_) cache_->clear();
}

Eigen::MatrixXd GRULayer::get_hidden_state() const {
    return hidden_state_;
}

// ============================================================================
// FUNZIONI DI ATTIVAZIONE
// ============================================================================

Eigen::MatrixXd GRULayer::sigmoid(const Eigen::MatrixXd& x) const {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::MatrixXd GRULayer::sigmoid_derivative(const Eigen::MatrixXd& x) const {
    Eigen::MatrixXd sig = sigmoid(x);
    return sig.array() * (1.0 - sig.array());
}

Eigen::MatrixXd GRULayer::tanh(const Eigen::MatrixXd& x) const {
    return x.array().tanh();
}

Eigen::MatrixXd GRULayer::tanh_derivative(const Eigen::MatrixXd& x) const {
    Eigen::MatrixXd t = tanh(x);
    return 1.0 - t.array().square();
}

// ============================================================================
// FORWARD PASS
// ============================================================================

Eigen::MatrixXd GRULayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd GRULayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "GRULayer");
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "layer not initialized", "GRULayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input", input.rows(), input_size_,
                                    input.rows(), input.cols(), "GRULayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<GRUCache>();
    }
    
    int batch_size = input.rows();
    
    cache_->input_cache = input;
    cache_->output_cache.resize(batch_size, units_);
    cache_->batch_size = batch_size;
    cache_->timesteps = 1;
    cache_->input_size = input_size_;
    cache_->hidden_size = units_;
    cache_->training = training;
    
    if (training) {
        cache_->hidden_states.clear();
        cache_->reset_gates.clear();
        cache_->update_gates.clear();
        cache_->candidate_hidden.clear();
        cache_->z_r.clear();
        cache_->z_z.clear();
        cache_->z_h.clear();
    }
    
    if (hidden_state_.rows() != batch_size || hidden_state_.cols() != units_) {
        hidden_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
    }
    
    // Calcolo reset gate (r_t) e update gate (z_t)
    Eigen::MatrixXd z_r = input * kernel_r + hidden_state_ * recurrent_r;
    Eigen::MatrixXd z_z = input * kernel_z + hidden_state_ * recurrent_z;
    
    if (use_bias_) {
        z_r.rowwise() += bias_r.transpose();
        z_z.rowwise() += bias_z.transpose();
    }
    
    Eigen::MatrixXd r_t = sigmoid(z_r);
    Eigen::MatrixXd z_t = sigmoid(z_z);
    
    // Calcolo candidate hidden (h_tilde)
    Eigen::MatrixXd h_prev_weighted = r_t.array() * hidden_state_.array();
    Eigen::MatrixXd z_h = input * kernel_h + h_prev_weighted * recurrent_h;
    
    if (use_bias_) {
        z_h.rowwise() += bias_h.transpose();
    }
    
    Eigen::MatrixXd h_tilde = tanh(z_h);
    
    // Aggiornamento hidden state
    Eigen::MatrixXd h_t = (1.0 - z_t.array()) * hidden_state_.array() + z_t.array() * h_tilde.array();
    
    if (training) {
        cache_->hidden_states.push_back(h_t);
        cache_->reset_gates.push_back(r_t);
        cache_->update_gates.push_back(z_t);
        cache_->candidate_hidden.push_back(h_tilde);
        cache_->z_r.push_back(z_r);
        cache_->z_z.push_back(z_z);
        cache_->z_h.push_back(z_h);
    }
    
    hidden_state_ = h_t;
    cache_->output_cache = h_t;
    
    return h_t;
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

Eigen::MatrixXd GRULayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_ || !cache_->training) {
        return gradient;
    }
    
    auto gru_cache = get_specific_cache();
    if (!gru_cache) {
        return gradient;
    }
    
    int batch_size = gru_cache->batch_size;
    
    if (gradient.rows() != batch_size || gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient", batch_size, units_,
                                    gradient.rows(), gradient.cols(), "GRULayer");
    }
    
    // Recupera valori dalla cache
    const Eigen::MatrixXd& r_t = gru_cache->reset_gates[0];
    const Eigen::MatrixXd& z_t = gru_cache->update_gates[0];
    const Eigen::MatrixXd& h_tilde = gru_cache->candidate_hidden[0];
    const Eigen::MatrixXd& z_r = gru_cache->z_r[0];
    const Eigen::MatrixXd& z_z = gru_cache->z_z[0];
    const Eigen::MatrixXd& z_h = gru_cache->z_h[0];
    
    const Eigen::MatrixXd& prev_h = (gru_cache->hidden_states.size() > 1) ? 
                                    gru_cache->hidden_states[0] : 
                                    Eigen::MatrixXd::Zero(batch_size, units_);
    
    const Eigen::MatrixXd& input = gru_cache->input_cache;
    
    Eigen::MatrixXd dH = gradient;
    
    // Gradiente per update gate
    Eigen::MatrixXd dZ_t = dH.array() * (h_tilde - prev_h).array() * sigmoid_derivative(z_z).array();
    
    // Gradiente per candidate hidden
    Eigen::MatrixXd dH_tilde = dH.array() * z_t.array() * tanh_derivative(z_h).array();
    
    // Gradiente per reset gate
    Eigen::MatrixXd dR_t = (dH_tilde * recurrent_h.transpose()).array() * 
                           prev_h.array() * sigmoid_derivative(z_r).array();
    
    // Gradienti per kernel weights
    Eigen::MatrixXd dKernel_r = input.transpose() * dR_t;
    Eigen::MatrixXd dKernel_z = input.transpose() * dZ_t;
    Eigen::MatrixXd dKernel_h = input.transpose() * dH_tilde;
    
    // Gradienti per recurrent weights
    Eigen::MatrixXd h_weighted = r_t.array() * prev_h.array();
    Eigen::MatrixXd dRecurrent_r = prev_h.transpose() * dR_t;
    Eigen::MatrixXd dRecurrent_z = prev_h.transpose() * dZ_t;
    Eigen::MatrixXd dRecurrent_h = h_weighted.transpose() * dH_tilde;
    
    // Aggiorna weights_gradient_
    int total_rows = (input_size_ + units_) * 3;
    int total_cols = 3 * units_ + (use_bias_ ? 3 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    // Kernel gradients
    weights_gradient_.block(0, 0, dKernel_r.rows(), units_) = dKernel_r;
    weights_gradient_.block(0, units_, dKernel_z.rows(), units_) = dKernel_z;
    weights_gradient_.block(0, 2 * units_, dKernel_h.rows(), units_) = dKernel_h;
    
    // Recurrent gradients
    weights_gradient_.block(kernel_r.rows(), 0, dRecurrent_r.rows(), units_) = dRecurrent_r;
    weights_gradient_.block(kernel_r.rows(), units_, dRecurrent_z.rows(), units_) = dRecurrent_z;
    weights_gradient_.block(kernel_r.rows(), 2 * units_, dRecurrent_h.rows(), units_) = dRecurrent_h;
    
    // Bias gradients
    if (use_bias_) {
        weights_gradient_.col(3 * units_).head(batch_size) = dR_t.colwise().sum();
        weights_gradient_.col(3 * units_ + 1).head(batch_size) = dZ_t.colwise().sum();
        weights_gradient_.col(3 * units_ + 2).head(batch_size) = dH_tilde.colwise().sum();
    }
    
    // Gradiente rispetto all'input
    Eigen::MatrixXd dX = dR_t * kernel_r.transpose() + 
                         dZ_t * kernel_z.transpose() + 
                         dH_tilde * kernel_h.transpose();
    
    return dX;
}

// ============================================================================
// GETTER/SETTER PESI
// ============================================================================

Eigen::MatrixXd GRULayer::get_weights() const {
    int total_rows = (input_size_ + units_) * 3;
    int total_cols = 3 * units_ + (use_bias_ ? 3 : 0);
    Eigen::MatrixXd weights(total_rows, total_cols);
    weights.setZero();
    
    int row = 0;
    auto copy = [&](const Eigen::MatrixXd& mat) {
        weights.block(row, 0, mat.rows(), mat.cols()) = mat;
        row += mat.rows();
    };
    
    copy(kernel_r);
    copy(kernel_z);
    copy(kernel_h);
    copy(recurrent_r);
    copy(recurrent_z);
    copy(recurrent_h);
    
    if (use_bias_) {
        weights.col(3 * units_).head(units_) = bias_r;
        weights.col(3 * units_ + 1).head(units_) = bias_z;
        weights.col(3 * units_ + 2).head(units_) = bias_h;
    }
    
    return weights;
}

void GRULayer::set_weights(const Eigen::MatrixXd& weights) {
    int row = 0;
    auto extract = [&](Eigen::MatrixXd& mat, int rows, int cols) {
        mat = weights.block(row, 0, rows, cols);
        row += rows;
    };
    
    extract(kernel_r, input_size_, units_);
    extract(kernel_z, input_size_, units_);
    extract(kernel_h, input_size_, units_);
    extract(recurrent_r, units_, units_);
    extract(recurrent_z, units_, units_);
    extract(recurrent_h, units_, units_);
    
    if (use_bias_ && weights.cols() > 3 * units_) {
        bias_r = weights.col(3 * units_).head(units_);
        bias_z = weights.col(3 * units_ + 1).head(units_);
        bias_h = weights.col(3 * units_ + 2).head(units_);
    }
}

Eigen::VectorXd GRULayer::get_biases() const {
    if (!use_bias_) return Eigen::VectorXd();
    Eigen::VectorXd biases(3 * units_);
    biases.segment(0, units_) = bias_r;
    biases.segment(units_, units_) = bias_z;
    biases.segment(2 * units_, units_) = bias_h;
    return biases;
}

void GRULayer::set_biases(const Eigen::VectorXd& biases) {
    if (!use_bias_) return;
    if (biases.size() != 3 * units_) {
        ML_THROW_PARAMETER_ERROR("biases", "size must be 3*units", "GRULayer");
    }
    bias_r = biases.segment(0, units_);
    bias_z = biases.segment(units_, units_);
    bias_h = biases.segment(2 * units_, units_);
}

int GRULayer::get_parameter_count() const {
    return 3 * (input_size_ * units_ + units_ * units_) + (use_bias_ ? 3 * units_ : 0);
}

// ============================================================================
// SERIALIZZAZIONE
// ============================================================================

void GRULayer::serialize(std::ostream& out) const {
    int32_t in_size = input_size_, u = units_;
    out.write(reinterpret_cast<const char*>(&in_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&u), sizeof(int32_t));
    
    auto write_str = [&](const std::string& s) {
        int32_t len = static_cast<int32_t>(s.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
        out.write(s.c_str(), len);
    };
    
    write_str(activation_);
    write_str(recurrent_activation_);
    
    int8_t use_bias_flag = use_bias_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_bias_flag), sizeof(int8_t));
    
    auto write_mat = [&](const Eigen::MatrixXd& m) {
        int32_t rows = static_cast<int32_t>(m.rows());
        int32_t cols = static_cast<int32_t>(m.cols());
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
    };
    
    // Scrivi kernel weights
    write_mat(kernel_r);
    write_mat(kernel_z);
    write_mat(kernel_h);
    
    // Scrivi recurrent weights
    write_mat(recurrent_r);
    write_mat(recurrent_z);
    write_mat(recurrent_h);
    
    // Scrivi bias
    if (use_bias_) {
        write_mat(bias_r);
        write_mat(bias_z);
        write_mat(bias_h);
    }
    
    if (!out.good()) {
        throw ml_exception::SerializationException("Failed to write GRULayer", "GRULayer");
    }
}

void GRULayer::deserialize(std::istream& in) {
    int32_t in_size, u;
    in.read(reinterpret_cast<char*>(&in_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&u), sizeof(int32_t));
    
    if (in_size <= 0 || u <= 0) {
        throw ml_exception::DeserializationException(
            "Invalid dimensions: input_size=" + std::to_string(in_size) +
            ", units=" + std::to_string(u), "GRULayer");
    }
    
    input_size_ = in_size;
    units_ = u;
    output_size_ = u;
    
    auto read_str = [&]() {
        int32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        std::vector<char> buf(len + 1, '\0');
        in.read(buf.data(), len);
        return std::string(buf.data());
    };
    
    activation_ = read_str();
    recurrent_activation_ = read_str();
    
    int8_t use_bias_flag;
    in.read(reinterpret_cast<char*>(&use_bias_flag), sizeof(int8_t));
    use_bias_ = (use_bias_flag != 0);
    
    auto read_mat = [&]() {
        int32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        Eigen::MatrixXd m(rows, cols);
        in.read(reinterpret_cast<char*>(m.data()), rows * cols * sizeof(double));
        return m;
    };
    
    // Leggi kernel weights
    kernel_r = read_mat();
    kernel_z = read_mat();
    kernel_h = read_mat();
    
    // Leggi recurrent weights
    recurrent_r = read_mat();
    recurrent_z = read_mat();
    recurrent_h = read_mat();
    
    // Leggi bias
    if (use_bias_) {
        bias_r = read_mat();
        bias_z = read_mat();
        bias_h = read_mat();
    }
    
    // Alloca gradienti
    int total_rows = (input_size_ + units_) * 3;
    int total_cols = 3 * units_ + (use_bias_ ? 3 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(3 * units_);
        bias_gradient_.setZero();
    }
    
    // Inizializza stato
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
    
    // Ricrea cache
    cache_ = std::make_shared<GRUCache>();
    
    if (!in.good() && !in.eof()) {
        throw ml_exception::DeserializationException("Stream error", "GRULayer");
    }
}

// ============================================================================
// CONFIG
// ============================================================================

std::string GRULayer::get_config() const {
    std::ostringstream oss;
    oss << "GRULayer(units=" << units_
        << ", input_size=" << input_size_
        << ", activation=" << activation_
        << ", recurrent_activation=" << recurrent_activation_
        << ", use_bias=" << (use_bias_ ? "true" : "false") << ")";
    return oss.str();
}

} // namespace layers