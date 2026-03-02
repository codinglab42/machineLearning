#include "components/layers/simple_rnn_layer.h"
#include <cmath>

namespace layers {

    SimpleRNNLayer::SimpleRNNLayer(int hidden_size, int input_size)
        : RecurrentLayer(hidden_size, input_size) {
        
        // Inizializza pesi (Xavier initialization)
        double scale = std::sqrt(2.0 / (input_size + hidden_size));
        Wx_ = Eigen::MatrixXd::Random(input_size, hidden_size) * scale;
        
        scale = std::sqrt(2.0 / (hidden_size + hidden_size));
        Wh_ = Eigen::MatrixXd::Random(hidden_size, hidden_size) * scale;
        
        b_ = Eigen::VectorXd::Zero(hidden_size);
        
        // Crea cache specifica
        cache_ = std::make_unique<SimpleRNNCache>();
    }

    Eigen::MatrixXd SimpleRNNLayer::forward(const Eigen::MatrixXd& input) {
        int batch_size = input.rows();
        int total_steps = input.cols() / input_size_;
        
        set_sequence_length(total_steps);
        
        // Inizializza cache
        auto* rnn_cache = static_cast<SimpleRNNCache*>(cache_.get());
        rnn_cache->init(total_steps, batch_size, hidden_size_);
        
        // Stato iniziale
        Eigen::MatrixXd h_t;
        if (h0_.size() > 0) {
            h_t = h0_;
        } else {
            h_t = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
        }
        rnn_cache->add_state(0, h_t, Eigen::MatrixXd());
        
        // Forward attraverso i timestep
        for (int t = 0; t < total_steps; ++t) {
            Eigen::MatrixXd x_t = extract_timestep(input, t);
            
            // h_t = tanh(x_t * Wx + h_{t-1} * Wh + b)
            Eigen::MatrixXd pre_act = x_t * Wx_ + h_t * Wh_;
            pre_act.rowwise() += b_.transpose();
            
            h_t = pre_act.array().tanh();  // attivazione tanh
            
            rnn_cache->add_state(t + 1, h_t, x_t);
        }
        
        // Output finale (puoi anche restituire tutti gli stati)
        cache_->set_output(h_t);
        return h_t;
    }

    Eigen::MatrixXd SimpleRNNLayer::backward(const Eigen::MatrixXd& gradient, 
                                            double learning_rate) {
        auto* rnn_cache = static_cast<SimpleRNNCache*>(cache_.get());
        
        int seq_len = rnn_cache->get_sequence_length();
        int batch_size = gradient.rows();
        
        // Gradienti
        Eigen::MatrixXd dWx = Eigen::MatrixXd::Zero(input_size_, hidden_size_);
        Eigen::MatrixXd dWh = Eigen::MatrixXd::Zero(hidden_size_, hidden_size_);
        Eigen::VectorXd db = Eigen::VectorXd::Zero(hidden_size_);
        
        // Gradiente rispetto all'output (già fornito)
        Eigen::MatrixXd dh_next = gradient;
        
        // Backprop through time
        for (int t = seq_len - 1; t >= 0; --t) {
            Eigen::MatrixXd h_t = rnn_cache->get_hidden_state(t + 1);
            Eigen::MatrixXd h_prev = rnn_cache->get_hidden_state(t);
            Eigen::MatrixXd x_t = rnn_cache->get_all_inputs()[t];
            
            // Gradiente attraverso tanh
            Eigen::MatrixXd dpre = dh_next.array() * (1 - h_t.array().square());
            
            // Gradienti pesi
            dWx += x_t.transpose() * dpre;
            dWh += h_prev.transpose() * dpre;
            db += dpre.colwise().sum().transpose();
            
            // Gradiente per il prossimo timestep
            dh_next = dpre * Wh_.transpose();
        }
        
        // Aggiorna pesi con gradient descent
        Wx_ -= learning_rate * dWx;
        Wh_ -= learning_rate * dWh;
        b_ -= learning_rate * db;
        
        // Gradiente rispetto all'input (per layer precedenti)
        return dh_next;  // oppure calcola gradiente per input
    }

    std::string SimpleRNNLayer::get_config() const {
        std::ostringstream oss;
        oss << "SimpleRNN(hidden=" << hidden_size_ 
            << ", input=" << input_size_
            << ", seq_len=" << sequence_length_ << ")";
        return oss.str();
    }

    void SimpleRNNLayer::set_weights(const Eigen::MatrixXd& Wx, 
                                     const Eigen::MatrixXd& Wh, 
                                     const Eigen::VectorXd& b) {
        ML_CHECK_DIMENSIONS(Wx.rows(), input_size_, Wx.cols(), hidden_size_,
                           "Wx", "SimpleRNNLayer");
        ML_CHECK_DIMENSIONS(Wh.rows(), hidden_size_, Wh.cols(), hidden_size_,
                           "Wh", "SimpleRNNLayer");
        ML_CHECK_DIMENSIONS(b.rows(), hidden_size_, b.cols(), 1,
                           "b", "SimpleRNNLayer");
        
        Wx_ = Wx;
        Wh_ = Wh;
        b_ = b;
    }

    Eigen::MatrixXd SimpleRNNLayer::get_weights() const {
        // Flatten tutti i pesi in un'unica matrice
        Eigen::MatrixXd weights(input_size_ * hidden_size_ + 
                                hidden_size_ * hidden_size_ + 
                                hidden_size_, 1);
        
        int idx = 0;
        // Wx
        for (int i = 0; i < Wx_.size(); ++i) {
            weights(idx++, 0) = Wx_(i);
        }
        // Wh
        for (int i = 0; i < Wh_.size(); ++i) {
            weights(idx++, 0) = Wh_(i);
        }
        // b
        for (int i = 0; i < b_.size(); ++i) {
            weights(idx++, 0) = b_(i);
        }
        
        return weights;
    }

    void SimpleRNNLayer::set_weights(const Eigen::MatrixXd& weights) {
        int expected_size = input_size_ * hidden_size_ + 
                           hidden_size_ * hidden_size_ + 
                           hidden_size_;
        
        ML_CHECK_DIMENSIONS(weights.rows(), expected_size, weights.cols(), 1,
                           "weights", "SimpleRNNLayer");
        
        int idx = 0;
        // Wx
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                Wx_(i, j) = weights(idx++, 0);
            }
        }
        // Wh
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                Wh_(i, j) = weights(idx++, 0);
            }
        }
        // b
        for (int i = 0; i < hidden_size_; ++i) {
            b_(i) = weights(idx++, 0);
        }
    }

} // namespace layers
