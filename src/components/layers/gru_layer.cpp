// src/components/layers/gru_layer.cpp
#include <random>
#include <memory>
#include <Eigen/Dense>
#include "components/layers/gru_layer.h"
#include "exceptions/exception_macros.h"
#include <sstream>
#include <cmath>

namespace layers {

GRULayer::GRULayer(int units, int input_size, 
                   const std::string& activation,
                   const std::string& recurrent_activation,
                   bool use_bias)
    : units_(units), input_size_(input_size), activation_(activation),
      recurrent_activation_(recurrent_activation), use_bias_(use_bias) {
    
    ML_CHECK_PARAM(units > 0, "units", "must be > 0", "GRULayer");
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
    
    double scale = std::sqrt(2.0 / (input_size + units));
    
    // Inizializza pesi per i 3 gate
    kernel_r.resize(input_size, units);
    kernel_z.resize(input_size, units);
    kernel_h.resize(input_size, units);
    
    recurrent_r.resize(units, units);
    recurrent_z.resize(units, units);
    recurrent_h.resize(units, units);
    
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
    
    initialize_matrix(kernel_r);
    initialize_matrix(kernel_z);
    initialize_matrix(kernel_h);
    
    initialize_matrix(recurrent_r);
    initialize_matrix(recurrent_z);
    initialize_matrix(recurrent_h);
    
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

void GRULayer::set_input_shape(int input_size) {
    ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
    
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
        
        // Ridimensiona i kernel
        initialize_matrix(kernel_r, input_size, units_);
        initialize_matrix(kernel_z, input_size, units_);
        initialize_matrix(kernel_h, input_size, units_);
        
        // I pesi ricorrenti rimangono [units, units]
        // I bias rimangono [units]
    }
}

void GRULayer::reset_state() {
    hidden_state_.resize(0, 0);
}

Eigen::MatrixXd GRULayer::get_hidden_state() const {
    return hidden_state_;
}

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

Eigen::MatrixXd GRULayer::forward(const Eigen::MatrixXd& input) {
    return forward(input, false);
}

Eigen::MatrixXd GRULayer::forward(const Eigen::MatrixXd& input, bool training) {
    ML_CHECK_NOT_EMPTY(input, "input", "GRULayer");
    
    if (input.cols() != input_size_) {
        ML_THROW_DIMENSION_MISMATCH("forward input",
            input.rows(), input_size_,
            input.rows(), input.cols(), "GRULayer");
    }
    
    if (!cache_) {
        cache_ = std::make_shared<GRUCache>();
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
    
    // Calcolo dei gate GRU
    Eigen::MatrixXd z_r = input * kernel_r + hidden_state_ * recurrent_r;
    Eigen::MatrixXd z_z = input * kernel_z + hidden_state_ * recurrent_z;
    
    if (use_bias_) {
        z_r.rowwise() += bias_r.transpose();
        z_z.rowwise() += bias_z.transpose();
    }
    
    Eigen::MatrixXd r_t = sigmoid(z_r);  // Reset gate
    Eigen::MatrixXd z_t = sigmoid(z_z);  // Update gate
    
    // Calcolo candidato stato nascosto
    Eigen::MatrixXd h_prev_weighted = r_t.array() * hidden_state_.array();
    Eigen::MatrixXd z_h = input * kernel_h + h_prev_weighted * recurrent_h;
    
    if (use_bias_) {
        z_h.rowwise() += bias_h.transpose();
    }
    
    Eigen::MatrixXd h_tilde = tanh(z_h);  // Candidate hidden
    
    // Aggiornamento stato nascosto
    Eigen::MatrixXd h_t = (1.0 - z_t.array()) * hidden_state_.array() + 
                          z_t.array() * h_tilde.array();
    
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

Eigen::MatrixXd GRULayer::backward(const Eigen::MatrixXd& gradient) {
    if (!cache_) {
        ML_THROW_FITTING_ERROR("GRULayer", "cache not initialized. Call forward first.");
    }
    
    auto gru_cache = get_specific_cache();
    
    if (!gru_cache->training) {
        return gradient;
    }
    
    int batch_size = gru_cache->batch_size;
    
    if (gradient.rows() != batch_size || gradient.cols() != units_) {
        ML_THROW_DIMENSION_MISMATCH("backward gradient",
            batch_size, units_,
            gradient.rows(), gradient.cols(), "GRULayer");
    }
    
    const Eigen::MatrixXd& h_t = gru_cache->hidden_states[0];
    const Eigen::MatrixXd& r_t = gru_cache->reset_gates[0];
    const Eigen::MatrixXd& z_t = gru_cache->update_gates[0];
    const Eigen::MatrixXd& h_tilde = gru_cache->candidate_hidden[0];
    const Eigen::MatrixXd& z_r = gru_cache->z_r[0];
    const Eigen::MatrixXd& z_z = gru_cache->z_z[0];
    const Eigen::MatrixXd& z_h = gru_cache->z_h[0];
    
    const Eigen::MatrixXd& prev_h = (gru_cache->hidden_states.size() > 1) ? 
                                    gru_cache->hidden_states[0] : 
                                    Eigen::MatrixXd::Zero(batch_size, units_);
    
    Eigen::MatrixXd dH = gradient;
    
    Eigen::MatrixXd dZ_t = dH.array() * (h_tilde - prev_h).array() * sigmoid_derivative(z_z).array();
    Eigen::MatrixXd dH_tilde = dH.array() * z_t.array() * tanh_derivative(z_h).array();
    
    Eigen::MatrixXd dR_t = (dH_tilde * recurrent_h.transpose()).array() * 
                           prev_h.array() * sigmoid_derivative(z_r).array();
    
    const Eigen::MatrixXd& input = gru_cache->input_cache;
    
    Eigen::MatrixXd dKernel_r = input.transpose() * dR_t;
    Eigen::MatrixXd dKernel_z = input.transpose() * dZ_t;
    Eigen::MatrixXd dKernel_h = input.transpose() * dH_tilde;
    
    Eigen::MatrixXd h_weighted = r_t.array() * prev_h.array();
    Eigen::MatrixXd dRecurrent_r = prev_h.transpose() * dR_t;
    Eigen::MatrixXd dRecurrent_z = prev_h.transpose() * dZ_t;
    Eigen::MatrixXd dRecurrent_h = h_weighted.transpose() * dH_tilde;
    
    int total_rows = kernel_r.rows() + recurrent_r.rows();
    
    if (use_bias_) {
        // UNIFICA: get_weights() restituisce [input_size + units, 3*units + 3]
        int total_cols = 3 * units_ + 3;
        
        Eigen::VectorXd dBias_r = dR_t.colwise().sum();
        Eigen::VectorXd dBias_z = dZ_t.colwise().sum();
        Eigen::VectorXd dBias_h = dH_tilde.colwise().sum();
        
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        
        // Kernel gradients
        weights_gradient_.block(0, 0, dKernel_r.rows(), units_) = dKernel_r;
        weights_gradient_.block(0, units_, dKernel_z.rows(), units_) = dKernel_z;
        weights_gradient_.block(0, 2*units_, dKernel_h.rows(), units_) = dKernel_h;
        
        // Recurrent gradients
        weights_gradient_.block(kernel_r.rows(), 0, dRecurrent_r.rows(), units_) = dRecurrent_r;
        weights_gradient_.block(kernel_r.rows(), units_, dRecurrent_z.rows(), units_) = dRecurrent_z;
        weights_gradient_.block(kernel_r.rows(), 2*units_, dRecurrent_h.rows(), units_) = dRecurrent_h;
        
        // Bias gradients (ultime 3 colonne)
        weights_gradient_.col(3*units_).head(dBias_r.size()) = dBias_r;
        weights_gradient_.col(3*units_ + 1).head(dBias_z.size()) = dBias_z;
        weights_gradient_.col(3*units_ + 2).head(dBias_h.size()) = dBias_h;
        
        bias_gradient_.resize(0);
    } else {
        int total_cols = 3 * units_;
        weights_gradient_.resize(total_rows, total_cols);
        weights_gradient_.setZero();
        
        weights_gradient_.block(0, 0, dKernel_r.rows(), units_) = dKernel_r;
        weights_gradient_.block(0, units_, dKernel_z.rows(), units_) = dKernel_z;
        weights_gradient_.block(0, 2*units_, dKernel_h.rows(), units_) = dKernel_h;
        
        weights_gradient_.block(kernel_r.rows(), 0, dRecurrent_r.rows(), units_) = dRecurrent_r;
        weights_gradient_.block(kernel_r.rows(), units_, dRecurrent_z.rows(), units_) = dRecurrent_z;
        weights_gradient_.block(kernel_r.rows(), 2*units_, dRecurrent_h.rows(), units_) = dRecurrent_h;
    }
    
    // dX deve avere dimensioni [batch_size, input_size]
    Eigen::MatrixXd dX = dR_t * kernel_r.transpose() + 
                        dZ_t * kernel_z.transpose() + 
                        dH_tilde * kernel_h.transpose();
    
    return dX;
}

// include/components/layers/gru_layer.h
// Aggiungi/modifica questi metodi:

Eigen::MatrixXd GRULayer::get_weights() const {
    int total_rows = kernel_r.rows() + recurrent_r.rows();  // input_size + units
    
    if (use_bias_) {
        // 3 gates: reset, update, candidate
        // Restituisce [input_size + units, 3*units + 3]
        int total_cols = 3 * units_ + 3;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.setZero();
        
        // Kernel weights per i 3 gate
        weights.block(0, 0, kernel_r.rows(), units_) = kernel_r;
        weights.block(0, units_, kernel_z.rows(), units_) = kernel_z;
        weights.block(0, 2*units_, kernel_h.rows(), units_) = kernel_h;
        
        // Recurrent weights per i 3 gate
        weights.block(kernel_r.rows(), 0, recurrent_r.rows(), units_) = recurrent_r;
        weights.block(kernel_r.rows(), units_, recurrent_z.rows(), units_) = recurrent_z;
        weights.block(kernel_r.rows(), 2*units_, recurrent_h.rows(), units_) = recurrent_h;
        
        // Bias nell'ultime 3 colonne
        weights.col(3*units_).head(units_) = bias_r;
        weights.col(3*units_ + 1).head(units_) = bias_z;
        weights.col(3*units_ + 2).head(units_) = bias_h;
        
        return weights;
    } else {
        // Senza bias: [input_size + units, 3*units]
        int total_cols = 3 * units_;
        Eigen::MatrixXd weights(total_rows, total_cols);
        weights.setZero();
        
        weights.block(0, 0, kernel_r.rows(), units_) = kernel_r;
        weights.block(0, units_, kernel_z.rows(), units_) = kernel_z;
        weights.block(0, 2*units_, kernel_h.rows(), units_) = kernel_h;
        
        weights.block(kernel_r.rows(), 0, recurrent_r.rows(), units_) = recurrent_r;
        weights.block(kernel_r.rows(), units_, recurrent_z.rows(), units_) = recurrent_z;
        weights.block(kernel_r.rows(), 2*units_, recurrent_h.rows(), units_) = recurrent_h;
        
        return weights;
    }
}

void GRULayer::set_weights(const Eigen::MatrixXd& weights){
    if (use_bias_) {
        int expected_cols = 3 * units_ + 3;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "GRULayer");
        }
        
        // Estrai kernel weights
        kernel_r = weights.block(0, 0, input_size_, units_);
        kernel_z = weights.block(0, units_, input_size_, units_);
        kernel_h = weights.block(0, 2*units_, input_size_, units_);
        
        // Estrai recurrent weights
        recurrent_r = weights.block(input_size_, 0, units_, units_);
        recurrent_z = weights.block(input_size_, units_, units_, units_);
        recurrent_h = weights.block(input_size_, 2*units_, units_, units_);
        
        // Estrai bias
        bias_r = weights.col(3*units_).head(units_);
        bias_z = weights.col(3*units_ + 1).head(units_);
        bias_h = weights.col(3*units_ + 2).head(units_);
    } else {
        int expected_cols = 3 * units_;
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "GRULayer");
        }
        
        kernel_r = weights.block(0, 0, input_size_, units_);
        kernel_z = weights.block(0, units_, input_size_, units_);
        kernel_h = weights.block(0, 2*units_, input_size_, units_);
        
        recurrent_r = weights.block(input_size_, 0, units_, units_);
        recurrent_z = weights.block(input_size_, units_, units_, units_);
        recurrent_h = weights.block(input_size_, 2*units_, units_, units_);
    }
}

int GRULayer::get_parameter_count() const {
    return kernel_r.size() + kernel_z.size() + kernel_h.size() +
           recurrent_r.size() + recurrent_z.size() + recurrent_h.size() +
           (use_bias_ ? bias_r.size() + bias_z.size() + bias_h.size() : 0);
}

Eigen::VectorXd GRULayer::get_biases() const {
    if (!use_bias_) return Eigen::VectorXd();
    
    Eigen::VectorXd all_biases(3 * units_);
    all_biases.segment(0, units_) = bias_r;
    all_biases.segment(units_, units_) = bias_z;
    all_biases.segment(2*units_, units_) = bias_h;
    return all_biases;
}

void GRULayer::set_biases(const Eigen::VectorXd& biases) {
    if (!use_bias_) return;
    
    if (biases.size() != 3 * units_) {
        ML_THROW_PARAMETER_ERROR("biases", "size must be 3*units", "GRULayer");
    }
    
    bias_r = biases.segment(0, units_);
    bias_z = biases.segment(units_, units_);
    bias_h = biases.segment(2*units_, units_);
}

// src/components/layers/gru_layer.cpp

void GRULayer::serialize(std::ostream& out) const {
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
    
    size_t rec_act_len = recurrent_activation_.size();
    out.write(reinterpret_cast<const char*>(&rec_act_len), sizeof(size_t));
    out.write(recurrent_activation_.c_str(), rec_act_len);
    
    // Serializza usando get_weights() che ora include i bias
    Eigen::MatrixXd weights_to_save = get_weights();
    int rows = weights_to_save.rows();
    int cols = weights_to_save.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(weights_to_save.data()), 
            rows * cols * sizeof(double));
}

void GRULayer::deserialize(std::istream& in) {
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
    
    size_t rec_act_len;
    in.read(reinterpret_cast<char*>(&rec_act_len), sizeof(size_t));
    recurrent_activation_.resize(rec_act_len);
    in.read(&recurrent_activation_[0], rec_act_len);
    
    // Leggi la matrice dei pesi
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    
    Eigen::MatrixXd loaded_weights(rows, cols);
    in.read(reinterpret_cast<char*>(loaded_weights.data()), rows * cols * sizeof(double));
    
    // Ridimensiona matrici
    kernel_r.resize(input_size_, units_);
    kernel_z.resize(input_size_, units_);
    kernel_h.resize(input_size_, units_);
    recurrent_r.resize(units_, units_);
    recurrent_z.resize(units_, units_);
    recurrent_h.resize(units_, units_);
    
    if (use_bias_) {
        bias_r.resize(units_);
        bias_z.resize(units_);
        bias_h.resize(units_);
    }
    
    // Usa set_weights per decomporre
    set_weights(loaded_weights);
    
    // Resetta stato
    hidden_state_.resize(0, 0);
    cache_ = std::make_shared<GRUCache>();
}

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