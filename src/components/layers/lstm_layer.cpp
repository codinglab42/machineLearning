#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/lstm_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

// ============================================================================
// COSTRUTTORI
// ============================================================================

LSTMLayer::LSTMLayer(int units, int input_size, 
                     const std::string& activation,
                     const std::string& recurrent_activation,
                     bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation),
      recurrent_activation_(recurrent_activation), use_bias_(use_bias),
      return_sequences_(false) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "LSTMLayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "LSTMLayer");
    
    // Inizializzazione Xavier
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
    
    // Kernel weights (input to hidden)
    init(kernel_i_, input_size, units);
    init(kernel_f_, input_size, units);
    init(kernel_c_, input_size, units);
    init(kernel_o_, input_size, units);
    
    // Recurrent weights (hidden to hidden)
    init(recurrent_i_, units, units);
    init(recurrent_f_, units, units);
    init(recurrent_c_, units, units);
    init(recurrent_o_, units, units);
    
    // Bias
    if (use_bias_) {
        bias_i_.resize(units);
        bias_f_.resize(units);
        bias_c_.resize(units);
        bias_o_.resize(units);
        bias_i_.setZero();
        bias_f_.setOnes();   // Forget gate bias = 1 (consigliato)
        bias_c_.setZero();
        bias_o_.setZero();
    }
    
    // Gradienti
    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
    
    hidden_state_.resize(0, 0);
    cell_state_.resize(0, 0);
    cache_ = nullptr;
}

// ============================================================================
// DIMENSIONI
// ============================================================================

void LSTMLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "LSTMLayer");
    
    if (input_size_ == input_size && kernel_i_.size() > 0) {
        return;
    }
    
    input_size_ = input_size;
    output_size_ = units_;
    
    // Ridimensiona kernel weights
    kernel_i_.resize(input_size_, units_);
    kernel_f_.resize(input_size_, units_);
    kernel_c_.resize(input_size_, units_);
    kernel_o_.resize(input_size_, units_);
    
    // Ridimensiona recurrent weights
    recurrent_i_.resize(units_, units_);
    recurrent_f_.resize(units_, units_);
    recurrent_c_.resize(units_, units_);
    recurrent_o_.resize(units_, units_);
    
    // Ridimensiona bias
    if (use_bias_) {
        bias_i_.resize(units_);
        bias_f_.resize(units_);
        bias_c_.resize(units_);
        bias_o_.resize(units_);
    }
    
    // Ridimensiona gradienti
    int total_rows = (input_size_ + units_) * 4;
    int total_cols = units_ + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units_ * 4);
        bias_gradient_.setZero();
    }
    
    // Ridimensiona stati
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
    cell_state_.resize(1, units_);
    cell_state_.setZero();
}

// ============================================================================
// INIZIALIZZAZIONE PESI
// ============================================================================

void LSTMLayer::initialize_weights() {
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "must be > 0", "LSTMLayer");
    ML_CHECK_PARAM(units_ > 0, "units", "must be > 0", "LSTMLayer");
    
    double scale = std::sqrt(2.0 / input_size_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    auto init = [&](Eigen::MatrixXd& mat) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = dist(gen);
    };
    
    init(kernel_i_);
    init(kernel_f_);
    init(kernel_c_);
    init(kernel_o_);
    
    double rec_scale = std::sqrt(1.0 / units_);
    std::normal_distribution<double> rec_dist(0.0, rec_scale);
    
    auto init_rec = [&](Eigen::MatrixXd& mat) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = rec_dist(gen);
    };
    
    init_rec(recurrent_i_);
    init_rec(recurrent_f_);
    init_rec(recurrent_c_);
    init_rec(recurrent_o_);
    
    if (use_bias_) {
        bias_i_.setZero();
        bias_f_.setOnes();
        bias_c_.setZero();
        bias_o_.setZero();
    }
}

// ============================================================================
// STATO
// ============================================================================

void LSTMLayer::reset_state() {
    hidden_state_.resize(0, 0);
    cell_state_.resize(0, 0);
    if (cache_) cache_->clear();
}

Eigen::MatrixXd LSTMLayer::get_hidden_state() const {
    return hidden_state_;
}

Eigen::MatrixXd LSTMLayer::get_cell_state() const {
    return cell_state_;
}

// ============================================================================
// FUNZIONI DI ATTIVAZIONE
// ============================================================================

Eigen::MatrixXd LSTMLayer::apply_activation(const Eigen::MatrixXd& z, const std::string& activation) const {
    if (activation == "tanh") {
        return z.array().tanh();
    } else if (activation == "sigmoid") {
        return 1.0 / (1.0 + (-z).array().exp());
    } else if (activation == "relu") {
        return z.cwiseMax(0.0);
    }
    return z;
}

Eigen::MatrixXd LSTMLayer::apply_activation_derivative(const Eigen::MatrixXd& z, const std::string& activation) const {
    if (activation == "tanh") {
        Eigen::MatrixXd tanh_z = z.array().tanh();
        return 1.0 - tanh_z.array().square();
    } else if (activation == "sigmoid") {
        Eigen::MatrixXd sig = 1.0 / (1.0 + (-z).array().exp());
        return sig.array() * (1.0 - sig.array());
    } else if (activation == "relu") {
        return (z.array() > 0.0).cast<double>();
    }
    return Eigen::MatrixXd::Ones(z.rows(), z.cols());
}

// ============================================================================
// FORWARD PASS
// ============================================================================

Eigen::MatrixXd LSTMLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd LSTMLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "LSTMLayer");
    ML_CHECK_PARAM(input_size_ > 0, "input_size", "layer not initialized", "LSTMLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input", input.rows(), input_size_,
                                    input.rows(), input.cols(), "LSTMLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<LSTMCache>();
    }
    
    int batch_size = input.rows();
    
    // Cache setup
    cache_->input_cache = input;
    cache_->output_cache.resize(batch_size, units_);
    cache_->batch_size = batch_size;
    cache_->timesteps = 1;
    cache_->training = training;
    
    if (training) {
        cache_->hidden_states.clear();
        cache_->cell_states.clear();
        cache_->input_gates.clear();
        cache_->forget_gates.clear();
        cache_->cell_candidates.clear();
        cache_->output_gates.clear();
    }
    
    // Inizializza stati nascosti se necessario
    if (hidden_state_.rows() != batch_size || hidden_state_.cols() != units_) {
        hidden_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
        cell_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
    }
    
    // Calcolo pre-attivazioni per i 4 gate
    Eigen::MatrixXd z_i = input * kernel_i_ + hidden_state_ * recurrent_i_;
    Eigen::MatrixXd z_f = input * kernel_f_ + hidden_state_ * recurrent_f_;
    Eigen::MatrixXd z_c = input * kernel_c_ + hidden_state_ * recurrent_c_;
    Eigen::MatrixXd z_o = input * kernel_o_ + hidden_state_ * recurrent_o_;
    
    // Aggiungi bias
    if (use_bias_) {
        z_i.rowwise() += bias_i_.transpose();
        z_f.rowwise() += bias_f_.transpose();
        z_c.rowwise() += bias_c_.transpose();
        z_o.rowwise() += bias_o_.transpose();
    }
    
    // Applica attivazioni
    Eigen::MatrixXd i_t = apply_activation(z_i, recurrent_activation_);
    Eigen::MatrixXd f_t = apply_activation(z_f, recurrent_activation_);
    Eigen::MatrixXd c_t = apply_activation(z_c, activation_);
    Eigen::MatrixXd o_t = apply_activation(z_o, recurrent_activation_);
    
    // Aggiorna cell state e hidden state
    Eigen::MatrixXd new_cell = f_t.array() * cell_state_.array() + i_t.array() * c_t.array();
    Eigen::MatrixXd new_hidden = o_t.array() * apply_activation(new_cell, activation_).array();
    
    // Salva in cache per il backward
    if (training) {
        cache_->input_gates.push_back(i_t);
        cache_->forget_gates.push_back(f_t);
        cache_->cell_candidates.push_back(c_t);
        cache_->output_gates.push_back(o_t);
        cache_->cell_states.push_back(cell_state_);
        cache_->hidden_states.push_back(new_hidden);
    }
    
    cell_state_ = new_cell;
    hidden_state_ = new_hidden;
    cache_->output_cache = hidden_state_;
    
    return hidden_state_;
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

Eigen::MatrixXd LSTMLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_ || !cache_->training) {
        return gradient;
    }
    
    auto lstm_cache = std::dynamic_pointer_cast<LSTMCache>(cache_);
    if (!lstm_cache) {
        return gradient;
    }
    
    int batch_size = lstm_cache->batch_size;
    
    if (gradient.rows() != batch_size || gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient", batch_size, units_,
                                    gradient.rows(), gradient.cols(), "LSTMLayer");
    }
    
    // Recupera valori dalla cache
    const Eigen::MatrixXd& i_t = lstm_cache->input_gates[0];
    const Eigen::MatrixXd& f_t = lstm_cache->forget_gates[0];
    const Eigen::MatrixXd& c_t = lstm_cache->cell_candidates[0];
    const Eigen::MatrixXd& o_t = lstm_cache->output_gates[0];
    const Eigen::MatrixXd& prev_cell = lstm_cache->cell_states[0];
    
    // Gradiente rispetto all'output
    Eigen::MatrixXd dH = gradient;
    
    // Calcolo gradiente per output gate
    Eigen::MatrixXd dO = dH.array() * apply_activation(cell_state_, activation_).array();
    dO = dO.array() * apply_activation_derivative(o_t, recurrent_activation_).array();
    
    // Gradiente per cell state
    Eigen::MatrixXd dC = dH.array() * o_t.array();
    dC = dC.array() * apply_activation_derivative(cell_state_, activation_).array();
    
    // Gradiente per forget gate
    Eigen::MatrixXd dF = dC.array() * prev_cell.array();
    dF = dF.array() * apply_activation_derivative(f_t, recurrent_activation_).array();
    
    // Gradiente per input gate
    Eigen::MatrixXd dI = dC.array() * c_t.array();
    dI = dI.array() * apply_activation_derivative(i_t, recurrent_activation_).array();
    
    // Gradiente per cell candidate
    Eigen::MatrixXd dCell = dC.array() * i_t.array();
    dCell = dCell.array() * apply_activation_derivative(c_t, activation_).array();
    
    // Aggiorna gradienti dei pesi (da accumulare)
    // Per semplicità, qui calcoliamo solo il gradiente rispetto all'input
    Eigen::MatrixXd dX = dI * kernel_i_.transpose() + 
                         dF * kernel_f_.transpose() + 
                         dCell * kernel_c_.transpose() + 
                         dO * kernel_o_.transpose();
    
    return dX;
}

// ============================================================================
// GETTER/SETTER PESI
// ============================================================================

Eigen::MatrixXd LSTMLayer::get_weights() const {
    int total_rows = (input_size_ + units_) * 4;
    int total_cols = units_ + (use_bias_ ? 1 : 0);
    Eigen::MatrixXd weights(total_rows, total_cols);
    weights.setZero();
    
    int row = 0;
    auto copy = [&](const Eigen::MatrixXd& mat) {
        weights.block(row, 0, mat.rows(), mat.cols()) = mat;
        row += mat.rows();
    };
    
    copy(kernel_i_);
    copy(kernel_f_);
    copy(kernel_c_);
    copy(kernel_o_);
    copy(recurrent_i_);
    copy(recurrent_f_);
    copy(recurrent_c_);
    copy(recurrent_o_);
    
    if (use_bias_) {
        weights.col(units_).head(units_) = bias_i_;
        weights.col(units_).segment(units_, units_) = bias_f_;
        weights.col(units_).segment(units_ * 2, units_) = bias_c_;
        weights.col(units_).segment(units_ * 3, units_) = bias_o_;
    }
    
    return weights;
}

void LSTMLayer::set_weights(const Eigen::MatrixXd& weights) {
    int row = 0;
    auto extract = [&](Eigen::MatrixXd& mat, int rows, int cols) {
        mat = weights.block(row, 0, rows, cols);
        row += rows;
    };
    
    extract(kernel_i_, input_size_, units_);
    extract(kernel_f_, input_size_, units_);
    extract(kernel_c_, input_size_, units_);
    extract(kernel_o_, input_size_, units_);
    extract(recurrent_i_, units_, units_);
    extract(recurrent_f_, units_, units_);
    extract(recurrent_c_, units_, units_);
    extract(recurrent_o_, units_, units_);
    
    if (use_bias_ && weights.cols() > units_) {
        bias_i_ = weights.col(units_).head(units_);
        bias_f_ = weights.col(units_).segment(units_, units_);
        bias_c_ = weights.col(units_).segment(units_ * 2, units_);
        bias_o_ = weights.col(units_).segment(units_ * 3, units_);
    }
}

Eigen::VectorXd LSTMLayer::get_biases() const {
    Eigen::VectorXd biases(units_ * 4);
    biases.head(units_) = bias_i_;
    biases.segment(units_, units_) = bias_f_;
    biases.segment(units_ * 2, units_) = bias_c_;
    biases.segment(units_ * 3, units_) = bias_o_;
    return biases;
}

void LSTMLayer::set_biases(const Eigen::VectorXd& biases) {
    if (biases.size() != units_ * 4) {
        ML_THROW_PARAMETER_ERROR("biases", "size must be 4 * units", "LSTMLayer");
    }
    bias_i_ = biases.head(units_);
    bias_f_ = biases.segment(units_, units_);
    bias_c_ = biases.segment(units_ * 2, units_);
    bias_o_ = biases.segment(units_ * 3, units_);
}

int LSTMLayer::get_parameter_count() const {
    return 4 * (input_size_ * units_ + units_ * units_) + (use_bias_ ? 4 * units_ : 0);
}

// ============================================================================
// SERIALIZZAZIONE
// ============================================================================

void LSTMLayer::serialize(std::ostream& out) const {
    // Scrivi dimensioni
    int32_t in_size = input_size_;
    int32_t u = units_;
    out.write(reinterpret_cast<const char*>(&in_size), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&u), sizeof(int32_t));
    
    // Scrivi activation
    int32_t act_len = static_cast<int32_t>(activation_.size());
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(int32_t));
    out.write(activation_.c_str(), act_len);
    
    // Scrivi recurrent_activation
    int32_t rec_act_len = static_cast<int32_t>(recurrent_activation_.size());
    out.write(reinterpret_cast<const char*>(&rec_act_len), sizeof(int32_t));
    out.write(recurrent_activation_.c_str(), rec_act_len);
    
    // Scrivi flags (use_bias, return_sequences)
    int8_t use_bias_flag = use_bias_ ? 1 : 0;
    int8_t return_seq_flag = return_sequences_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&use_bias_flag), sizeof(int8_t));
    out.write(reinterpret_cast<const char*>(&return_seq_flag), sizeof(int8_t));
    
    // Helper per scrivere matrici
    auto write_matrix = [&](const Eigen::MatrixXd& m) {
        int32_t rows = static_cast<int32_t>(m.rows());
        int32_t cols = static_cast<int32_t>(m.cols());
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
    };
    
    // Scrivi kernel weights
    write_matrix(kernel_i_);
    write_matrix(kernel_f_);
    write_matrix(kernel_c_);
    write_matrix(kernel_o_);
    
    // Scrivi recurrent weights
    write_matrix(recurrent_i_);
    write_matrix(recurrent_f_);
    write_matrix(recurrent_c_);
    write_matrix(recurrent_o_);
    
    // Scrivi bias
    if (use_bias_) {
        write_matrix(bias_i_);
        write_matrix(bias_f_);
        write_matrix(bias_c_);
        write_matrix(bias_o_);
    }
    
    if (!out.good()) {
        throw ml_exception::SerializationException("Failed to write LSTMLayer", "LSTMLayer");
    }
}

void LSTMLayer::deserialize(std::istream& in) {
    // Leggi dimensioni
    int32_t in_size, u;
    in.read(reinterpret_cast<char*>(&in_size), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&u), sizeof(int32_t));
    
    if (in_size <= 0 || u <= 0) {
        throw ml_exception::DeserializationException(
            "Invalid dimensions: input_size=" + std::to_string(in_size) +
            ", units=" + std::to_string(u), "LSTMLayer");
    }
    
    input_size_ = in_size;
    units_ = u;
    output_size_ = u;
    
    // Leggi activation
    int32_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(int32_t));
    std::vector<char> act_buf(act_len + 1, '\0');
    in.read(act_buf.data(), act_len);
    activation_ = std::string(act_buf.data());
    
    // Leggi recurrent_activation
    int32_t rec_act_len;
    in.read(reinterpret_cast<char*>(&rec_act_len), sizeof(int32_t));
    std::vector<char> rec_buf(rec_act_len + 1, '\0');
    in.read(rec_buf.data(), rec_act_len);
    recurrent_activation_ = std::string(rec_buf.data());
    
    // Leggi flags
    int8_t use_bias_flag, return_seq_flag;
    in.read(reinterpret_cast<char*>(&use_bias_flag), sizeof(int8_t));
    in.read(reinterpret_cast<char*>(&return_seq_flag), sizeof(int8_t));
    use_bias_ = (use_bias_flag != 0);
    return_sequences_ = (return_seq_flag != 0);
    
    // Helper per leggere matrici
    auto read_matrix = [&]() -> Eigen::MatrixXd {
        int32_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        Eigen::MatrixXd m(rows, cols);
        in.read(reinterpret_cast<char*>(m.data()), rows * cols * sizeof(double));
        return m;
    };
    
    // Alloca e leggi kernel weights
    kernel_i_ = read_matrix();
    kernel_f_ = read_matrix();
    kernel_c_ = read_matrix();
    kernel_o_ = read_matrix();
    
    // Alloca e leggi recurrent weights
    recurrent_i_ = read_matrix();
    recurrent_f_ = read_matrix();
    recurrent_c_ = read_matrix();
    recurrent_o_ = read_matrix();
    
    // Alloca e leggi bias
    if (use_bias_) {
        bias_i_ = read_matrix();
        bias_f_ = read_matrix();
        bias_c_ = read_matrix();
        bias_o_ = read_matrix();
    }
    
    // Inizializza gradienti
    int total_rows = (input_size_ + units_) * 4;
    int total_cols = units_ + (use_bias_ ? 1 : 0);
    weights_gradient_.resize(total_rows, total_cols);
    weights_gradient_.setZero();
    
    if (use_bias_) {
        bias_gradient_.resize(units_ * 4);
        bias_gradient_.setZero();
    }
    
    // Inizializza stati
    hidden_state_.resize(1, units_);
    hidden_state_.setZero();
    cell_state_.resize(1, units_);
    cell_state_.setZero();
    
    // Ricrea cache
    cache_ = std::make_shared<LSTMCache>();
    
    if (!in.good() && !in.eof()) {
        throw ml_exception::DeserializationException("Stream error", "LSTMLayer");
    }
}

// ============================================================================
// CONFIG
// ============================================================================

std::string LSTMLayer::get_config() const {
    std::ostringstream oss;
    oss << "LSTMLayer(units=" << units_
        << ", input_size=" << input_size_
        << ", activation=" << activation_
        << ", recurrent_activation=" << recurrent_activation_
        << ", use_bias=" << (use_bias_ ? "true" : "false") << ")";
    return oss.str();
}

} // namespace layers