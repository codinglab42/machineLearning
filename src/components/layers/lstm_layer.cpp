// src/components/layers/lstm_layer.cpp
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/lstm_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

LSTMLayer::LSTMLayer(int units, int input_size, 
                     const std::string& activation,
                     const std::string& recurrent_activation,
                     bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation),
      recurrent_activation_(recurrent_activation), use_bias_(use_bias) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "LSTMLayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "LSTMLayer");
    
    double scale = std::sqrt(2.0 / (input_size + units));

    weights_gradient_.resize(0, 0);
    bias_gradient_.resize(0);
    
    // Inizializza pesi per i 4 gate
    kernel_i.resize(input_size, units);
    kernel_f.resize(input_size, units);
    kernel_c.resize(input_size, units);
    kernel_o.resize(input_size, units);
    
    recurrent_i.resize(units, units);
    recurrent_f.resize(units, units);
    recurrent_c.resize(units, units);
    recurrent_o.resize(units, units);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, scale);
    
    auto initialize_matrix = [&](Eigen::MatrixXd& mat) {
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                mat(i, j) = dist(gen);
            }
        }
    };
    
    initialize_matrix(kernel_i);
    initialize_matrix(kernel_f);
    initialize_matrix(kernel_c);
    initialize_matrix(kernel_o);
    
    initialize_matrix(recurrent_i);
    initialize_matrix(recurrent_f);
    initialize_matrix(recurrent_c);
    initialize_matrix(recurrent_o);
    
    if (use_bias_) {
        bias_i.setZero(units);
        bias_f = Eigen::VectorXd::Ones(units);  // Forget gate bias inizializzato a 1
        bias_c.setZero(units);
        bias_o.setZero(units);
    }
    
    hidden_state_.resize(0, 0);
    cell_state_.resize(0, 0);
    cache_ = nullptr;
}

void LSTMLayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "LSTMLayer");
    
    if (input_size_ != input_size) {
        input_size_ = input_size;
        
        double scale = std::sqrt(2.0 / (input_size + units_));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, scale);
        
        auto initialize_matrix = [&](Eigen::MatrixXd& mat, int rows, int cols) {
            mat.resize(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    mat(i, j) = dist(gen);
                }
            }
        };
        
        // Ridimensiona tutti i kernel
        initialize_matrix(kernel_i, input_size, units_);
        initialize_matrix(kernel_f, input_size, units_);
        initialize_matrix(kernel_c, input_size, units_);
        initialize_matrix(kernel_o, input_size, units_);
        
        // I pesi ricorrenti rimangono [units, units]
        // I bias rimangono [units]
    }
}

void LSTMLayer::reset_state() {
    hidden_state_.resize(0, 0);
    cell_state_.resize(0, 0);
}

Eigen::MatrixXd LSTMLayer::get_hidden_state() const {
    return hidden_state_;
}

Eigen::MatrixXd LSTMLayer::sigmoid(const Eigen::MatrixXd& x) const {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::MatrixXd LSTMLayer::sigmoid_derivative(const Eigen::MatrixXd& x) const {
    Eigen::MatrixXd sig = sigmoid(x);
    return sig.array() * (1.0 - sig.array());
}

Eigen::MatrixXd LSTMLayer::tanh(const Eigen::MatrixXd& x) const {
    return x.array().tanh();
}

Eigen::MatrixXd LSTMLayer::tanh_derivative(const Eigen::MatrixXd& x) const {
    Eigen::MatrixXd t = tanh(x);
    return 1.0 - t.array().square();
}

Eigen::MatrixXd LSTMLayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd LSTMLayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "LSTMLayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "LSTMLayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<LSTMCache>();
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
        cache_->cell_states.clear();
        cache_->input_gates.clear();
        cache_->forget_gates.clear();
        cache_->output_gates.clear();
        cache_->cell_candidates.clear();
        cache_->z_i.clear();
        cache_->z_f.clear();
        cache_->z_c.clear();
        cache_->z_o.clear();
    }
    
    if (hidden_state_.rows() != batch_size || hidden_state_.cols() != units_) {
        hidden_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
        cell_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
    }
    
    // Calcolo dei gate LSTM
    Eigen::MatrixXd z_i = input * kernel_i + hidden_state_ * recurrent_i;
    Eigen::MatrixXd z_f = input * kernel_f + hidden_state_ * recurrent_f;
    Eigen::MatrixXd z_c = input * kernel_c + hidden_state_ * recurrent_c;
    Eigen::MatrixXd z_o = input * kernel_o + hidden_state_ * recurrent_o;
    
    if (use_bias_) {
        z_i.rowwise() += bias_i.transpose();
        z_f.rowwise() += bias_f.transpose();
        z_c.rowwise() += bias_c.transpose();
        z_o.rowwise() += bias_o.transpose();
    }
    
    Eigen::MatrixXd i_t = sigmoid(z_i);  // Input gate
    Eigen::MatrixXd f_t = sigmoid(z_f);  // Forget gate
    Eigen::MatrixXd c_tilde = tanh(z_c); // Cell candidate
    Eigen::MatrixXd o_t = sigmoid(z_o);  // Output gate
    
    // Aggiornamento stato cella e stato nascosto
    Eigen::MatrixXd c_t = f_t.array() * cell_state_.array() + 
                          i_t.array() * c_tilde.array();
    Eigen::MatrixXd h_t = o_t.array() * tanh(c_t).array();
    
    if (training) {
        cache_->hidden_states.push_back(h_t);
        cache_->cell_states.push_back(c_t);
        cache_->input_gates.push_back(i_t);
        cache_->forget_gates.push_back(f_t);
        cache_->output_gates.push_back(o_t);
        cache_->cell_candidates.push_back(c_tilde);
        cache_->z_i.push_back(z_i);
        cache_->z_f.push_back(z_f);
        cache_->z_c.push_back(z_c);
        cache_->z_o.push_back(z_o);
    }
    
    hidden_state_ = h_t;
    cell_state_ = c_t;
    
    cache_->output_cache = h_t;
    return h_t;
}

// ============================================================================
// BACKWARD - SOLO CALCOLO GRADIENTI, NESSUN AGGIORNAMENTO PESI
// ============================================================================
Eigen::MatrixXd LSTMLayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("LSTMLayer", "cache not initialized. Call forward first.");
    }
    
    auto lstm_cache = get_specific_cache();
    
    if (!lstm_cache->training) {
        return gradient;
    }
    
    int batch_size = lstm_cache->batch_size;
    
    if (gradient.rows() != batch_size || gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient",
            batch_size, units_,
            gradient.rows(), gradient.cols(), "LSTMLayer");
    }
    
    const Eigen::MatrixXd& c_t = lstm_cache->cell_states[0];
    const Eigen::MatrixXd& i_t = lstm_cache->input_gates[0];
    const Eigen::MatrixXd& o_t = lstm_cache->output_gates[0];
    const Eigen::MatrixXd& c_tilde = lstm_cache->cell_candidates[0];
    const Eigen::MatrixXd& z_i = lstm_cache->z_i[0];
    const Eigen::MatrixXd& z_f = lstm_cache->z_f[0];
    const Eigen::MatrixXd& z_c = lstm_cache->z_c[0];
    const Eigen::MatrixXd& z_o = lstm_cache->z_o[0];
    
    const Eigen::MatrixXd& prev_c = (lstm_cache->cell_states.size() > 1) ? 
                                    lstm_cache->cell_states[0] : 
                                    Eigen::MatrixXd::Zero(batch_size, units_);
    
    Eigen::MatrixXd dH = gradient;
    Eigen::MatrixXd dC = dH.array() * o_t.array() * tanh_derivative(c_t).array();
    
    Eigen::MatrixXd dO = dH.array() * tanh(c_t).array() * sigmoid_derivative(z_o).array();
    Eigen::MatrixXd dI = dC.array() * c_tilde.array() * sigmoid_derivative(z_i).array();
    Eigen::MatrixXd dF = dC.array() * prev_c.array() * sigmoid_derivative(z_f).array();
    Eigen::MatrixXd dC_tilde = dC.array() * i_t.array() * tanh_derivative(z_c).array();
    
    const Eigen::MatrixXd& input = lstm_cache->input_cache;
    const Eigen::MatrixXd& prev_h = (lstm_cache->hidden_states.size() > 1) ? 
                                    lstm_cache->hidden_states[0] : 
                                    Eigen::MatrixXd::Zero(batch_size, units_);
    
    // Calcola gradienti
    Eigen::MatrixXd dKernel_i = input.transpose() * dI;
    Eigen::MatrixXd dKernel_f = input.transpose() * dF;
    Eigen::MatrixXd dKernel_c = input.transpose() * dC_tilde;
    Eigen::MatrixXd dKernel_o = input.transpose() * dO;
    
    Eigen::MatrixXd dRecurrent_i = prev_h.transpose() * dI;
    Eigen::MatrixXd dRecurrent_f = prev_h.transpose() * dF;
    Eigen::MatrixXd dRecurrent_c = prev_h.transpose() * dC_tilde;
    Eigen::MatrixXd dRecurrent_o = prev_h.transpose() * dO;
    
    int total_rows = kernel_i.rows() + recurrent_i.rows();
    
    if (use_bias_) {
        // UNIFICA: get_weights() restituisce [input_size + units, 4*units + 4]
        int total_cols = 4 * units_ + 4;
        
        Eigen::VectorXd dBias_i = dI.colwise().sum();
        Eigen::VectorXd dBias_f = dF.colwise().sum();
        Eigen::VectorXd dBias_c = dC_tilde.colwise().sum();
        Eigen::VectorXd dBias_o = dO.colwise().sum();
        
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        
        // Kernel gradients
        weights_gradient_.block(0, 0, dKernel_i.rows(), units_) = dKernel_i;
        weights_gradient_.block(0, units_, dKernel_f.rows(), units_) = dKernel_f;
        weights_gradient_.block(0, 2*units_, dKernel_c.rows(), units_) = dKernel_c;
        weights_gradient_.block(0, 3*units_, dKernel_o.rows(), units_) = dKernel_o;
        
        // Recurrent gradients
        weights_gradient_.block(kernel_i.rows(), 0, dRecurrent_i.rows(), units_) = dRecurrent_i;
        weights_gradient_.block(kernel_i.rows(), units_, dRecurrent_f.rows(), units_) = dRecurrent_f;
        weights_gradient_.block(kernel_i.rows(), 2*units_, dRecurrent_c.rows(), units_) = dRecurrent_c;
        weights_gradient_.block(kernel_i.rows(), 3*units_, dRecurrent_o.rows(), units_) = dRecurrent_o;
        
        // Bias gradients (ultime 4 colonne)
        weights_gradient_.col(4*units_).head(dBias_i.size()) = dBias_i;
        weights_gradient_.col(4*units_ + 1).head(dBias_f.size()) = dBias_f;
        weights_gradient_.col(4*units_ + 2).head(dBias_c.size()) = dBias_c;
        weights_gradient_.col(4*units_ + 3).head(dBias_o.size()) = dBias_o;
        
        bias_gradient_.resize(0);
    } else {
        int total_cols = 4 * units_;
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        
        weights_gradient_.block(0, 0, dKernel_i.rows(), units_) = dKernel_i;
        weights_gradient_.block(0, units_, dKernel_f.rows(), units_) = dKernel_f;
        weights_gradient_.block(0, 2*units_, dKernel_c.rows(), units_) = dKernel_c;
        weights_gradient_.block(0, 3*units_, dKernel_o.rows(), units_) = dKernel_o;
        
        weights_gradient_.block(kernel_i.rows(), 0, dRecurrent_i.rows(), units_) = dRecurrent_i;
        weights_gradient_.block(kernel_i.rows(), units_, dRecurrent_f.rows(), units_) = dRecurrent_f;
        weights_gradient_.block(kernel_i.rows(), 2*units_, dRecurrent_c.rows(), units_) = dRecurrent_c;
        weights_gradient_.block(kernel_i.rows(), 3*units_, dRecurrent_o.rows(), units_) = dRecurrent_o;
    }
    
    // dX deve avere dimensioni [batch_size, input_size]
    Eigen::MatrixXd dX = dI * kernel_i.transpose() + 
                        dF * kernel_f.transpose() + 
                        dC_tilde * kernel_c.transpose() + 
                        dO * kernel_o.transpose();
    
    return dX;
}

// include/components/layers/lstm_layer.h
// Aggiungi/modifica questi metodi:

Eigen::MatrixXd LSTMLayer::get_weights() const {
    int total_rows = kernel_i.rows() + recurrent_i.rows();  // input_size + units
    
    if (use_bias_) {
        // 4 gates: input, forget, cell, output
        // Restituisce [input_size + units, 4*units + 4]
        int total_cols = 4 * units_ + 4;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.setZero();
        
        // Kernel weights per i 4 gate
        weights.block(0, 0, kernel_i.rows(), units_) = kernel_i;
        weights.block(0, units_, kernel_f.rows(), units_) = kernel_f;
        weights.block(0, 2*units_, kernel_c.rows(), units_) = kernel_c;
        weights.block(0, 3*units_, kernel_o.rows(), units_) = kernel_o;
        
        // Recurrent weights per i 4 gate
        weights.block(kernel_i.rows(), 0, recurrent_i.rows(), units_) = recurrent_i;
        weights.block(kernel_i.rows(), units_, recurrent_f.rows(), units_) = recurrent_f;
        weights.block(kernel_i.rows(), 2*units_, recurrent_c.rows(), units_) = recurrent_c;
        weights.block(kernel_i.rows(), 3*units_, recurrent_o.rows(), units_) = recurrent_o;
        
        // Bias nell'ultime 4 colonne
        weights.col(4*units_).head(units_) = bias_i;
        weights.col(4*units_ + 1).head(units_) = bias_f;
        weights.col(4*units_ + 2).head(units_) = bias_c;
        weights.col(4*units_ + 3).head(units_) = bias_o;
        
        return weights;
    } else {
        // Senza bias: [input_size + units, 4*units]
        int total_cols = 4 * units_;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.setZero();
        
        weights.block(0, 0, kernel_i.rows(), units_) = kernel_i;
        weights.block(0, units_, kernel_f.rows(), units_) = kernel_f;
        weights.block(0, 2*units_, kernel_c.rows(), units_) = kernel_c;
        weights.block(0, 3*units_, kernel_o.rows(), units_) = kernel_o;
        
        weights.block(kernel_i.rows(), 0, recurrent_i.rows(), units_) = recurrent_i;
        weights.block(kernel_i.rows(), units_, recurrent_f.rows(), units_) = recurrent_f;
        weights.block(kernel_i.rows(), 2*units_, recurrent_c.rows(), units_) = recurrent_c;
        weights.block(kernel_i.rows(), 3*units_, recurrent_o.rows(), units_) = recurrent_o;
        
        return weights;
    }
}

void LSTMLayer::set_weights(const Eigen::MatrixXd& weights) {
    if (use_bias_) {
        int expected_cols = 4 * units_ + 4;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "LSTMLayer");
        }
        
        // Estrai kernel weights
        kernel_i = weights.block(0, 0, input_size_, units_);
        kernel_f = weights.block(0, units_, input_size_, units_);
        kernel_c = weights.block(0, 2*units_, input_size_, units_);
        kernel_o = weights.block(0, 3*units_, input_size_, units_);
        
        // Estrai recurrent weights
        recurrent_i = weights.block(input_size_, 0, units_, units_);
        recurrent_f = weights.block(input_size_, units_, units_, units_);
        recurrent_c = weights.block(input_size_, 2*units_, units_, units_);
        recurrent_o = weights.block(input_size_, 3*units_, units_, units_);
        
        // Estrai bias
        bias_i = weights.col(4*units_).head(units_);
        bias_f = weights.col(4*units_ + 1).head(units_);
        bias_c = weights.col(4*units_ + 2).head(units_);
        bias_o = weights.col(4*units_ + 3).head(units_);
    } else {
        int expected_cols = 4 * units_;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "LSTMLayer");
        }
        
        kernel_i = weights.block(0, 0, input_size_, units_);
        kernel_f = weights.block(0, units_, input_size_, units_);
        kernel_c = weights.block(0, 2*units_, input_size_, units_);
        kernel_o = weights.block(0, 3*units_, input_size_, units_);
        
        recurrent_i = weights.block(input_size_, 0, units_, units_);
        recurrent_f = weights.block(input_size_, units_, units_, units_);
        recurrent_c = weights.block(input_size_, 2*units_, units_, units_);
        recurrent_o = weights.block(input_size_, 3*units_, units_, units_);
    }
}

int LSTMLayer::get_parameter_count() const {
    return kernel_i.size() + kernel_f.size() + kernel_c.size() + kernel_o.size() +
           recurrent_i.size() + recurrent_f.size() + recurrent_c.size() + recurrent_o.size() +
           (use_bias_ ? bias_i.size() + bias_f.size() + bias_c.size() + bias_o.size() : 0);
}

Eigen::VectorXd LSTMLayer::get_biases() const {
    if (!use_bias_) return Eigen::VectorXd();
    
    Eigen::VectorXd all_biases(4 * units_);
    all_biases.segment(0, units_) = bias_i;
    all_biases.segment(units_, units_) = bias_f;
    all_biases.segment(2*units_, units_) = bias_c;
    all_biases.segment(3*units_, units_) = bias_o;
    return all_biases;
}

void LSTMLayer::set_biases(const Eigen::VectorXd& biases) {
    if (!use_bias_) return;
    
    if (biases.size() != 4 * units_) {
        ML_THROW_PARAMETER_ERROR("biases", "size must be 4*units", "LSTMLayer");
    }
    
    bias_i = biases.segment(0, units_);
    bias_f = biases.segment(units_, units_);
    bias_c = biases.segment(2*units_, units_);
    bias_o = biases.segment(3*units_, units_);
}

void LSTMLayer::serialize(std::ostream& out) const {
    // Versione
    uint32_t version = get_version();
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Scrivi configurazione
    out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
    
    size_t act_len = activation_.size();
    out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
    out.write(activation_.c_str(), act_len);
    
    size_t rec_act_len = recurrent_activation_.size();
    out.write(reinterpret_cast<const char*>(&rec_act_len), sizeof(size_t));
    out.write(recurrent_activation_.c_str(), rec_act_len);
    
    out.write(reinterpret_cast<const char*>(&use_bias_), sizeof(bool));
    
    // Serializza usando get_weights() che ora include i bias
    Eigen::MatrixXd weights_to_save = get_weights();
    int rows = weights_to_save.rows();
    int cols = weights_to_save.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(weights_to_save.data()), 
            rows * cols * sizeof(double));
}

void LSTMLayer::deserialize(std::istream& in) {
    // Leggi versione
    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Leggi configurazione
    in.read(reinterpret_cast<char*>(&units_), sizeof(int));
    in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
    
    size_t act_len;
    in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
    activation_.resize(act_len);
    in.read(&activation_[0], act_len);
    
    size_t rec_act_len;
    in.read(reinterpret_cast<char*>(&rec_act_len), sizeof(size_t));
    recurrent_activation_.resize(rec_act_len);
    in.read(&recurrent_activation_[0], rec_act_len);
    
    in.read(reinterpret_cast<char*>(&use_bias_), sizeof(bool));
    
    // Leggi la matrice dei pesi
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    Eigen::MatrixXd loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    
    // Ridimensiona matrici
    kernel_i.resize(input_size_, units_);
    kernel_f.resize(input_size_, units_);
    kernel_c.resize(input_size_, units_);
    kernel_o.resize(input_size_, units_);
    recurrent_i.resize(units_, units_);
    recurrent_f.resize(units_, units_);
    recurrent_c.resize(units_, units_);
    recurrent_o.resize(units_, units_);
    
    if (use_bias_) {
        bias_i.resize(units_);
        bias_f.resize(units_);
        bias_c.resize(units_);
        bias_o.resize(units_);
    }
    
    // Usa set_weights per decomporre
    set_weights(loaded_weights);
    
    // Resetta stato
    hidden_state_.resize(0, 0);
    cell_state_.resize(0, 0);
    cache_ = std::make_shared<LSTMCache>();
}

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