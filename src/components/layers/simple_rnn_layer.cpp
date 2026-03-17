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
            bias_.setZero();  // <-- DEVE ESSERE QUI
        }
        
        hidden_state_.resize(0, 0);
        cache_ = nullptr;
    }
    void SimpleRNNLayer::set_input_shape(int input_size) {
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "SimpleRNNLayer");
        input_size_ = input_size;
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
        
        if (input.cols() != input_size_) {
            ML_THROW_DIMENSION_MISMATCH("forward input",
                input.rows(), input_size_,
                input.rows(), input.cols(), "SimpleRNNLayer");
        }
        
        if (!cache_) {
            cache_ = std::make_shared<SimpleRNNCache>();
        }
        
        int batch_size = input.rows();
        int timesteps = 1; // Assumiamo input [batch, features]
        
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
        
        // Inizializza stato nascosto se necessario
        if (hidden_state_.rows() != batch_size || hidden_state_.cols() != units_) {
            hidden_state_ = Eigen::MatrixXd::Zero(batch_size, units_);
        }
        
        Eigen::MatrixXd output(batch_size, units_);
        
        // Calcolo: h_t = activation(x_t * kernel + h_{t-1} * recurrent + bias)
        Eigen::MatrixXd z = input * kernel_ + hidden_state_ * recurrent_;
        
        if (use_bias_) {
            z.rowwise() += bias_.transpose();
        }
          
        if (training) {
            cache_->z_values.push_back(z);
        }
        
        hidden_state_ = apply_activation(z);
        output = hidden_state_;
        
        if (training) {
            cache_->hidden_states.push_back(hidden_state_);
            cache_->pre_activations.push_back(z);
        }
        
        cache_->output_cache = output;
        return output;
    }

    Eigen::MatrixXd SimpleRNNLayer::backward(const Eigen::MatrixXd& gradient, double learning_rate) {
        if (!cache_) {
            ML_THROW_FITTING_ERROR("SimpleRNNLayer", "cache not initialized. Call forward first.");
        }
        
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "SimpleRNNLayer");
        
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
        
        // Backpropagation through time (semplificata per 1 timestep)
        const Eigen::MatrixXd& z = rnn_cache->z_values[0];
        const Eigen::MatrixXd& prev_h = (rnn_cache->hidden_states.size() > 1) ? 
                                         rnn_cache->hidden_states[0] : 
                                         Eigen::MatrixXd::Zero(batch_size, units_);
        
        Eigen::MatrixXd dZ = gradient.array() * apply_activation_derivative(z).array();
        
        // Gradienti
        Eigen::MatrixXd dKernel = rnn_cache->input_cache.transpose() * dZ;
        Eigen::MatrixXd dRecurrent = prev_h.transpose() * dZ;
        Eigen::VectorXd dBias;
        if (use_bias_) {
            dBias = dZ.colwise().sum();
        }
        
        // Gradiente per l'input (da propagare indietro)
        Eigen::MatrixXd dX = dZ * kernel_.transpose();
        
        // Gradiente per lo stato nascosto precedente
        Eigen::MatrixXd dPrevHidden = dZ * recurrent_.transpose();
        
        // Aggiorna pesi
        kernel_ -= learning_rate * dKernel / batch_size;
        recurrent_ -= learning_rate * dRecurrent / batch_size;
        if (use_bias_) {
            bias_ -= learning_rate * dBias / batch_size;
        }
        
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
        return z.array().tanh(); // default tanh per RNN
    }

    Eigen::MatrixXd SimpleRNNLayer::apply_activation_derivative(const Eigen::MatrixXd& z) const {
        if (activation_ == "tanh") {
            Eigen::MatrixXd tanh = z.array().tanh();
            return 1.0 - tanh.array().square();
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

    Eigen::MatrixXd SimpleRNNLayer::get_weights() const {
        int total_rows = kernel_.rows() + recurrent_.rows();
        int total_cols = kernel_.cols() + recurrent_.cols();
        if (use_bias_) total_cols += 1;
        
        // Inizializza a ZERO tutta la matrice
        Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(total_rows, total_cols);
        
        // Ora assegna i blocchi
        weights.block(0, 0, kernel_.rows(), kernel_.cols()) = kernel_;
        weights.block(0, kernel_.cols(), recurrent_.rows(), recurrent_.cols()) = recurrent_;
        
        if (use_bias_) {
            weights.col(kernel_.cols() + recurrent_.cols()).head(bias_.size()) = bias_;
            // Le righe rimanenti restano zero
        }
        
        return weights;
    }
    void SimpleRNNLayer::set_weights(const Eigen::MatrixXd& weights) {
        int expected_cols = kernel_.cols() + recurrent_.cols() + (use_bias_ ? 1 : 0);
        if (weights.cols() != expected_cols) {
            ML_THROW_PARAMETER_ERROR("weights", "invalid dimensions", "SimpleRNNLayer");
        }
        
        kernel_ = weights.block(0, 0, kernel_.rows(), kernel_.cols());
        recurrent_ = weights.block(0, kernel_.cols(), recurrent_.rows(), recurrent_.cols());
        
        if (use_bias_) {
            bias_ = weights.col(kernel_.cols() + recurrent_.cols());
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

    void SimpleRNNLayer::serialize(std::ostream& out) const {
        // Scrivi configurazione con lunghezza
        std::string config = get_config();
        size_t config_len = config.size() + 1;
        out.write(reinterpret_cast<const char*>(&config_len), sizeof(size_t));
        out.write(config.c_str(), config_len);
        
        // Scrivi parametri
        out.write(reinterpret_cast<const char*>(&units_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&input_size_), sizeof(int));
        
        bool has_bias = use_bias_;
        out.write(reinterpret_cast<const char*>(&has_bias), sizeof(bool));
        
        // Scrivi attivazione
        size_t act_len = activation_.size() + 1;
        out.write(reinterpret_cast<const char*>(&act_len), sizeof(size_t));
        out.write(activation_.c_str(), act_len);
        
        // Scrivi kernel
        for (int i = 0; i < kernel_.rows(); ++i) {
            for (int j = 0; j < kernel_.cols(); ++j) {
                out.write(reinterpret_cast<const char*>(&kernel_(i, j)), sizeof(double));
            }
        }
        
        // Scrivi recurrent
        for (int i = 0; i < recurrent_.rows(); ++i) {
            for (int j = 0; j < recurrent_.cols(); ++j) {
                out.write(reinterpret_cast<const char*>(&recurrent_(i, j)), sizeof(double));
            }
        }
        
        // Scrivi bias se presenti
        if (use_bias_) {
            for (int i = 0; i < bias_.size(); ++i) {
                out.write(reinterpret_cast<const char*>(&bias_(i)), sizeof(double));
            }
        }
    }

    void SimpleRNNLayer::deserialize(std::istream& in) {
        // Leggi configurazione
        size_t config_len;
        in.read(reinterpret_cast<char*>(&config_len), sizeof(size_t));
        std::vector<char> config_buf(config_len);
        in.read(config_buf.data(), config_len);
        
        // Leggi parametri
        in.read(reinterpret_cast<char*>(&units_), sizeof(int));
        in.read(reinterpret_cast<char*>(&input_size_), sizeof(int));
        
        bool has_bias;
        in.read(reinterpret_cast<char*>(&has_bias), sizeof(bool));
        use_bias_ = has_bias;
        
        // Leggi attivazione
        size_t act_len;
        in.read(reinterpret_cast<char*>(&act_len), sizeof(size_t));
        std::vector<char> act_buf(act_len);
        in.read(act_buf.data(), act_len);
        activation_ = std::string(act_buf.data());
        
        // Ridimensiona matrici
        kernel_.resize(input_size_, units_);
        recurrent_.resize(units_, units_);
        if (use_bias_) {
            bias_.resize(units_);
        }
        
        // Leggi kernel
        for (int i = 0; i < kernel_.rows(); ++i) {
            for (int j = 0; j < kernel_.cols(); ++j) {
                in.read(reinterpret_cast<char*>(&kernel_(i, j)), sizeof(double));
            }
        }
        
        // Leggi recurrent
        for (int i = 0; i < recurrent_.rows(); ++i) {
            for (int j = 0; j < recurrent_.cols(); ++j) {
                in.read(reinterpret_cast<char*>(&recurrent_(i, j)), sizeof(double));
            }
        }
        
        // Leggi bias se presenti
        if (use_bias_) {
            for (int i = 0; i < bias_.size(); ++i) {
                in.read(reinterpret_cast<char*>(&bias_(i)), sizeof(double));
            }
        }
        
        // Resetta stato
        hidden_state_.resize(0, 0);
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

