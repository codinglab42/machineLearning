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
            // Bias per forget gate inizializzato a 1 (meglio per LSTM)
            bias_i.setZero(units);
            bias_f = Eigen::VectorXd::Ones(units);
            bias_c.setZero(units);
            bias_o.setZero(units);
        }
        
        hidden_state_.resize(0, 0);
        cell_state_.resize(0, 0);
        cache_ = nullptr;
    }

    void LSTMLayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "LSTMLayer");
        input_size_ = input_size;
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
            cache_->z_o.clear();
            cache_->z_c.clear();
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

    Eigen::MatrixXd LSTMLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("LSTMLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "LSTMLayer");
        
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
        
        // Recupera dati dalla cache
        const Eigen::MatrixXd& h_t = lstm_cache->hidden_states[0];
        const Eigen::MatrixXd& c_t = lstm_cache->cell_states[0];
        const Eigen::MatrixXd& i_t = lstm_cache->input_gates[0];
        const Eigen::MatrixXd& f_t = lstm_cache->forget_gates[0];
        const Eigen::MatrixXd& o_t = lstm_cache->output_gates[0];
        const Eigen::MatrixXd& c_tilde = lstm_cache->cell_candidates[0];
        const Eigen::MatrixXd& z_i = lstm_cache->z_i[0];
        const Eigen::MatrixXd& z_f = lstm_cache->z_f[0];
        const Eigen::MatrixXd& z_c = lstm_cache->z_c[0];
        const Eigen::MatrixXd& z_o = lstm_cache->z_o[0];
        
        const Eigen::MatrixXd& prev_c = (lstm_cache->cell_states.size() > 1) ? 
                                        lstm_cache->cell_states[0] : 
                                        Eigen::MatrixXd::Zero(batch_size, units_);
        
        // Gradiente rispetto a h_t (dato)
        Eigen::MatrixXd dH = gradient;
        
        // Gradiente rispetto a c_t
        Eigen::MatrixXd dC = dH.array() * o_t.array() * tanh_derivative(c_t).array();
        
        // Gradienti per i gate
        Eigen::MatrixXd dO = dH.array() * tanh(c_t).array() * sigmoid_derivative(z_o).array();
        Eigen::MatrixXd dI = dC.array() * c_tilde.array() * sigmoid_derivative(z_i).array();
        Eigen::MatrixXd dF = dC.array() * prev_c.array() * sigmoid_derivative(z_f).array();
        Eigen::MatrixXd dC_tilde = dC.array() * i_t.array() * tanh_derivative(z_c).array();
        
        // Gradienti per i pesi
        const Eigen::MatrixXd& input = lstm_cache->input_cache;
        const Eigen::MatrixXd& prev_h = (lstm_cache->hidden_states.size() > 1) ? 
                                        lstm_cache->hidden_states[0] : 
                                        Eigen::MatrixXd::Zero(batch_size, units_);
        
        Eigen::MatrixXd dKernel_i = input.transpose() * dI;
        Eigen::MatrixXd dKernel_f = input.transpose() * dF;
        Eigen::MatrixXd dKernel_c = input.transpose() * dC_tilde;
        Eigen::MatrixXd dKernel_o = input.transpose() * dO;
        
        Eigen::MatrixXd dRecurrent_i = prev_h.transpose() * dI;
        Eigen::MatrixXd dRecurrent_f = prev_h.transpose() * dF;
        Eigen::MatrixXd dRecurrent_c = prev_h.transpose() * dC_tilde;
        Eigen::MatrixXd dRecurrent_o = prev_h.transpose() * dO;
        
        Eigen::VectorXd dBias_i, dBias_f, dBias_c, dBias_o;
        if (use_bias_) {
            dBias_i = dI.colwise().sum();
            dBias_f = dF.colwise().sum();
            dBias_c = dC_tilde.colwise().sum();
            dBias_o = dO.colwise().sum();
        }
        
        // Gradiente per l'input
        Eigen::MatrixXd dX = dI * kernel_i.transpose() + 
                            dF * kernel_f.transpose() + 
                            dC_tilde * kernel_c.transpose() + 
                            dO * kernel_o.transpose();
        
        // Aggiorna pesi
        kernel_i -= learning_rate * dKernel_i / batch_size;
        kernel_f -= learning_rate * dKernel_f / batch_size;
        kernel_c -= learning_rate * dKernel_c / batch_size;
        kernel_o -= learning_rate * dKernel_o / batch_size;
        
        recurrent_i -= learning_rate * dRecurrent_i / batch_size;
        recurrent_f -= learning_rate * dRecurrent_f / batch_size;
        recurrent_c -= learning_rate * dRecurrent_c / batch_size;
        recurrent_o -= learning_rate * dRecurrent_o / batch_size;
        
        if (use_bias_) {
            bias_i -= learning_rate * dBias_i / batch_size;
            bias_f -= learning_rate * dBias_f / batch_size;
            bias_c -= learning_rate * dBias_c / batch_size;
            bias_o -= learning_rate * dBias_o / batch_size;
        }
        
        return dX;
    }

    Eigen::MatrixXd LSTMLayer::get_weights() const {
        int total_rows = kernel_i.rows() + recurrent_i.rows();
        int total_cols = 4 * units_ + (use_bias_ ? 4 : 0);
        
        Eigen::MatrixXd weights(total_rows, total_cols);
        
        // Kernel weights
        weights.block(0, 0, kernel_i.rows(), units_) = kernel_i;
        weights.block(0, units_, kernel_f.rows(), units_) = kernel_f;
        weights.block(0, 2*units_, kernel_c.rows(), units_) = kernel_c;
        weights.block(0, 3*units_, kernel_o.rows(), units_) = kernel_o;
        
        // Recurrent weights
        weights.block(kernel_i.rows(), 0, recurrent_i.rows(), units_) = recurrent_i;
        weights.block(kernel_i.rows(), units_, recurrent_f.rows(), units_) = recurrent_f;
        weights.block(kernel_i.rows(), 2*units_, recurrent_c.rows(), units_) = recurrent_c;
        weights.block(kernel_i.rows(), 3*units_, recurrent_o.rows(), units_) = recurrent_o;
        
        if (use_bias_) {
            weights.col(4*units_) = bias_i;
            weights.col(4*units_ + 1) = bias_f;
            weights.col(4*units_ + 2) = bias_c;
            weights.col(4*units_ + 3) = bias_o;
        }
        
        return weights;
    }

    void LSTMLayer::set_weights(const Eigen::MatrixXd& weights) {
        int expected_cols = 4 * units_ + (use_bias_ ? 4 : 0);
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
        
        if (use_bias_) {
            bias_i = weights.col(4*units_);
            bias_f = weights.col(4*units_ + 1);
            bias_c = weights.col(4*units_ + 2);
            bias_o = weights.col(4*units_ + 3);
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
        
        serialize_matrix(kernel_i);
        serialize_matrix(kernel_f);
        serialize_matrix(kernel_c);
        serialize_matrix(kernel_o);
        
        serialize_matrix(recurrent_i);
        serialize_matrix(recurrent_f);
        serialize_matrix(recurrent_c);
        serialize_matrix(recurrent_o);
        
        if (use_bias_) {
            for (int i = 0; i < bias_i.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_i(i)), sizeof(double));
            for (int i = 0; i < bias_f.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_f(i)), sizeof(double));
            for (int i = 0; i < bias_c.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_c(i)), sizeof(double));
            for (int i = 0; i < bias_o.size(); ++i) out.write(reinterpret_cast<const char*>(&bias_o(i)), sizeof(double));
        }
    }

    void LSTMLayer::deserialize(std::istream& in) {
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
        
        deserialize_matrix(kernel_i, input_size_, units_);
        deserialize_matrix(kernel_f, input_size_, units_);
        deserialize_matrix(kernel_c, input_size_, units_);
        deserialize_matrix(kernel_o, input_size_, units_);
        
        deserialize_matrix(recurrent_i, units_, units_);
        deserialize_matrix(recurrent_f, units_, units_);
        deserialize_matrix(recurrent_c, units_, units_);
        deserialize_matrix(recurrent_o, units_, units_);
        
        if (use_bias_) {
            bias_i.resize(units_);
            bias_f.resize(units_);
            bias_c.resize(units_);
            bias_o.resize(units_);
            
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_i(i)), sizeof(double));
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_f(i)), sizeof(double));
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_c(i)), sizeof(double));
            for (int i = 0; i < units_; ++i) in.read(reinterpret_cast<char*>(&bias_o(i)), sizeof(double));
        }
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
