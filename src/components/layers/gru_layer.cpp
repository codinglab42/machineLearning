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
    }

    void GRULayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
        input_size_ = input_size;
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

    Eigen::MatrixXd GRULayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("GRULayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "GRULayer");
        
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
        
        // Recupera dati dalla cache
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
        
        // Gradiente rispetto a h_t (dato)
        Eigen::MatrixXd dH = gradient;
        
        // Gradienti intermedi
        Eigen::MatrixXd dZ_t = dH.array() * (h_tilde - prev_h).array() * sigmoid_derivative(z_z).array();
        Eigen::MatrixXd dH_tilde = dH.array() * z_t.array() * tanh_derivative(z_h).array();
        
        // Gradiente per reset gate
        Eigen::MatrixXd dR_t = (dH_tilde * recurrent_h.transpose()).array() * 
                               prev_h.array() * sigmoid_derivative(z_r).array();
        
        // Gradienti per i pesi
        const Eigen::MatrixXd& input = gru_cache->input_cache;
        
        Eigen::MatrixXd dKernel_r = input.transpose() * dR_t;
        Eigen::MatrixXd dKernel_z = input.transpose() * dZ_t;
        Eigen::MatrixXd dKernel_h = input.transpose() * dH_tilde;
        
        Eigen::MatrixXd h_weighted = r_t.array() * prev_h.array();
        Eigen::MatrixXd dRecurrent_r = prev_h.transpose() * dR_t;
        Eigen::MatrixXd dRecurrent_z = prev_h.transpose() * dZ_t;
        Eigen::MatrixXd dRecurrent_h = h_weighted.transpose() * dH_tilde;
        
        Eigen::VectorXd dBias_r, dBias_z, dBias_h;
        if (use_bias_) {
            dBias_r = dR_t.colwise().sum();
            dBias_z = dZ_t.colwise().sum();
            dBias_h = dH_tilde.colwise().sum();
        }
        
        // Gradiente per l'input
        Eigen::MatrixXd dX = dR_t * kernel_r.transpose() + 
                            dZ_t * kernel_z.transpose() + 
                            dH_tilde * kernel_h.transpose();
        
        // Aggiorna pesi
        kernel_r -= learning_rate * dKernel_r / batch_size;
        kernel_z -= learning_rate * dKernel_z / batch_size;
        kernel_h -= learning_rate * dKernel_h / batch_size;
        
        recurrent_r -= learning_rate * dRecurrent_r / batch_size;
        recurrent_z -= learning_rate * dRecurrent_z / batch_size;
        recurrent_h -= learning_rate * dRecurrent_h / batch_size;
        
        if (use_bias_) {
            bias_r -= learning_rate * dBias_r / batch_size;
            bias_z -= learning_rate * dBias_z / batch_size;
            bias_h -= learning_rate * dBias_h / batch_size;
        }
        
        return dX;
    }

    Eigen::MatrixXd GRULayer::get_weights() const {
        int total_rows = kernel_r.rows() + recurrent_r.rows();
        int total_cols = 3 * units_ + (use_bias_ ? 3 : 0);
        
        Eigen::MatrixXd weights(total_rows, total_cols);
        
        // Kernel weights
        weights.block(0, 0, kernel_r.rows(), units_) = kernel_r;
        weights.block(0, units_, kernel_z.rows(), units_) = kernel_z;
        weights.block(0, 2*units_, kernel_h.rows(), units_) = kernel_h;
        
        // Recurrent weights
        weights.block(kernel_r.rows(), 0, recurrent_r.rows(), units_) = recurrent_r;
        weights.block(kernel_r.rows(), units_, recurrent_z.rows(), units_) = recurrent_z;
        weights.block(kernel_r.rows(), 2*units_, recurrent_h.rows(), units_) = recurrent_h;
        
        if (use_bias_) {
            weights.col(3*units_) = bias_r;
            weights.col(3*units_ + 1) = bias_z;
            weights.col(3*units_ + 2) = bias_h;
        }
        
        return weights;
    }

    void GRULayer::set_weights(const Eigen::MatrixXd& weights) {
        int expected_cols = 3 * units_ + (use_bias_ ? 3 : 0);
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "GRULayer");
        }
        
        kernel_r = weights.block(0, 0, input_size_, units_);
        kernel_z = weights.block(0, units_, input_size_, units_);
        kernel_h = weights.block(0, 2*units_, input_size_, units_);
        
        recurrent_r = weights.block(input_size_, 0, units_, units_);
        recurrent_z = weights.block(input_size_, units_, units_, units_);
        recurrent_h = weights.block(input_size_, 2*units_, units_, units_);
        
        if (use_bias_) {
            bias_r = weights.col(3*units_);
            bias_z = weights.col(3*units_ + 1);
            bias_h = weights.col(3*units_ + 2);
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

    void GRULayer::serialize(std::ostream& out) const {
        out << get_config() << std::endl;
        out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        
        bool has_bias = use_bias_;
        out.write(reinterpret_cast<const char*>(&has_bias), sizeof(bool));
        
        auto serialize_matrix = [&](const Eigen::MatrixXd& mat) {
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    out.write(reinterpret_cast<const char*>(&mat(i, j)), sizeof(double));
                }
            }
        };
        
        serialize_matrix(kernel_r);
        serialize_matrix(kernel_z);
        serialize_matrix(kernel_h);
        
        serialize_matrix(recurrent_r);
        serialize_matrix(recurrent_z);
        serialize_matrix(recurrent_h);
        
        if (use_bias_) {
            for (int i = 0; i < bias_r.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_r(i)), sizeof(double));
            for (int i = 0; i < bias_z.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_z(i)), sizeof(double));
            for (int i = 0; i < bias_h.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_h(i)), sizeof(double));
        }
    }

    void GRULayer::deserialize(std::istream& in) {
        std::string config;
        std::getline(in, config);
        
        in.read(reinterpret_cast<char*>(&units_), sizeof(int));
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        
        bool has_bias;
        in.read(reinterpret_cast<char*>(&has_bias), sizeof(bool));
        use_bias_ = has_bias;
        
        auto deserialize_matrix = [&](Eigen::MatrixXd& mat, int rows, int cols) {
            mat.resize(rows, cols);
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    in.read(reinterpret_cast<char*>(&mat(i, j)), sizeof(double));
                }
            }
        };
        
        deserialize_matrix(kernel_r, input_size_, units_);
        deserialize_matrix(kernel_z, input_size_, units_);
        deserialize_matrix(kernel_h, input_size_, units_);
        
        deserialize_matrix(recurrent_r, units_, units_);
        deserialize_matrix(recurrent_z, units_, units_);
        deserialize_matrix(recurrent_h, units_, units_);
        
        if (use_bias_) {
            bias_r.resize(units_);
            bias_z.resize(units_);
            bias_h.resize(units_);
            
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_r(i)), sizeof(double));
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_z(i)), sizeof(double));
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_h(i)), sizeof(double));
        }
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

