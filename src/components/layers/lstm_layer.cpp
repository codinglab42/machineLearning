#include "components/layers/lstm_layer.h"
#include "utils/serializable.h"
#include <cmath>
#include <stdexcept>

namespace layers {

    // Costruttore
    LSTMLayer::LSTMLayer(int hidden_size, int input_size)
        : RecurrentLayer(hidden_size, input_size) {
        
        // Xavier initialization per i pesi
        double scale_wx = std::sqrt(2.0 / (input_size + hidden_size));
        double scale_wh = std::sqrt(2.0 / (hidden_size + hidden_size));
        
        // 4 gates: input (i), forget (f), cell (g), output (o)
        // Wx: [input_size, 4*hidden_size]
        Wx_ = Eigen::MatrixXd::Random(input_size, 4 * hidden_size) * scale_wx;
        
        // Wh: [hidden_size, 4*hidden_size]
        Wh_ = Eigen::MatrixXd::Random(hidden_size, 4 * hidden_size) * scale_wh;
        
        // Bias: [4*hidden_size]
        b_ = Eigen::VectorXd::Zero(4 * hidden_size);
        
        // Inizializza bias per forget gate a 1 (per non dimenticare all'inizio)
        for (int i = hidden_size; i < 2 * hidden_size; ++i) {
            b_(i) = 1.0;
        }
        
        // Crea cache specifica LSTM
        cache_ = std::make_unique<LSTMCache>();
    }

    // Forward pass
    Eigen::MatrixXd LSTMLayer::forward(const Eigen::MatrixXd& input) {
        // Verifica dimensioni input
        int batch_size = input.rows();
        int total_elements = input.cols();
        
        // Calcola numero di timestep
        if (total_elements % input_size_ != 0) {
            throw ml_exception::DimensionMismatchException(
                "input columns", input_size_, total_elements / batch_size,
                total_elements, batch_size, "LSTMLayer");
        }
        
        int total_steps = total_elements / input_size_;
        set_sequence_length(total_steps);
        
        // Ottieni cache specifica
        auto* lstm_cache = static_cast<LSTMCache*>(cache_.get());
        lstm_cache->init(total_steps, batch_size, hidden_size_);
        
        // Stati iniziali
        Eigen::MatrixXd h_t, c_t;
        if (h0_.size() > 0) {
            // Usa stato iniziale fornito
            h_t = h0_;
            c_t = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
        } else {
            // Stato iniziale zero
            h_t = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
            c_t = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
        }
        
        // Salva stato iniziale nella cache
        lstm_cache->add_lstm_state(0, h_t, c_t, Eigen::MatrixXd(),
                                   Eigen::MatrixXd(), Eigen::MatrixXd(),
                                   Eigen::MatrixXd(), Eigen::MatrixXd());
        
        // Forward attraverso i timestep
        for (int t = 0; t < total_steps; ++t) {
            // Estrai input per questo timestep
            Eigen::MatrixXd x_t = input.block(0, t * input_size_, 
                                              batch_size, input_size_);
            
            // Calcolo pre-attivazione gates
            // gates = x_t * Wx + h_t * Wh + b
            Eigen::MatrixXd gates = x_t * Wx_ + h_t * Wh_;
            gates.rowwise() += b_.transpose();
            
            // Split gates nei 4 componenti
            Eigen::MatrixXd i_gate = gates.block(0, 0, batch_size, hidden_size_);
            Eigen::MatrixXd f_gate = gates.block(0, hidden_size_, batch_size, hidden_size_);
            Eigen::MatrixXd g_gate = gates.block(0, 2 * hidden_size_, batch_size, hidden_size_);
            Eigen::MatrixXd o_gate = gates.block(0, 3 * hidden_size_, batch_size, hidden_size_);
            
            // Attivazioni
            // i = sigmoid(i_gate)
            i_gate = (1.0 / (1.0 + (-i_gate).array().exp()));
            
            // f = sigmoid(f_gate)
            f_gate = (1.0 / (1.0 + (-f_gate).array().exp()));
            
            // g = tanh(g_gate)
            g_gate = g_gate.array().tanh();
            
            // o = sigmoid(o_gate)
            o_gate = (1.0 / (1.0 + (-o_gate).array().exp()));
            
            // Nuovo stato cella: c_t = f * c_{t-1} + i * g
            c_t = (f_gate.array() * c_t.array()) + (i_gate.array() * g_gate.array());
            
            // Nuovo stato hidden: h_t = o * tanh(c_t)
            h_t = o_gate.array() * c_t.array().tanh();
            
            // Salva nella cache
            lstm_cache->add_lstm_state(t + 1, h_t, c_t, x_t,
                                      i_gate, f_gate, o_gate, g_gate);
        }
        
        // Salva output finale nella cache base
        cache_->set_output(h_t);
        cache_->set_input(input);
        
        return h_t;
    }

    // Backward pass
    Eigen::MatrixXd LSTMLayer::backward(const Eigen::MatrixXd& gradient, 
                                        double learning_rate) {
        auto* lstm_cache = static_cast<LSTMCache*>(cache_.get());
        
        if (!lstm_cache->is_valid()) {
            throw std::runtime_error("LSTMLayer: cache not valid for backward");
        }
        
        int seq_len = lstm_cache->get_sequence_length();
        int batch_size = gradient.rows();
        
        // Verifica dimensioni gradiente
        if (gradient.cols() != hidden_size_) {
            throw ml_exception::DimensionMismatchException(
                "gradient cols", hidden_size_, 1,
                gradient.cols(), 1, "LSTMLayer");
        }
        
        // Inizializza gradienti pesi
        Eigen::MatrixXd dWx = Eigen::MatrixXd::Zero(input_size_, 4 * hidden_size_);
        Eigen::MatrixXd dWh = Eigen::MatrixXd::Zero(hidden_size_, 4 * hidden_size_);
        Eigen::VectorXd db = Eigen::VectorXd::Zero(4 * hidden_size_);
        
        // Gradienti rispetto agli stati
        Eigen::MatrixXd dh_next = gradient;  // gradiente da loss
        Eigen::MatrixXd dc_next = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
        
        // Backpropagation through time
        for (int t = seq_len; t >= 1; --t) {
            // Recupera stati e gates dalla cache
            const auto& step = lstm_cache->get_step(t);
            const auto& step_prev = lstm_cache->get_step(t - 1);
            
            Eigen::MatrixXd h_t = step.h;
            Eigen::MatrixXd c_t = step.c;
            Eigen::MatrixXd c_prev = step_prev.c;
            Eigen::MatrixXd x_t = step.x;
            
            Eigen::MatrixXd i_t = step.i;
            Eigen::MatrixXd f_t = step.f;
            Eigen::MatrixXd o_t = step.o;
            Eigen::MatrixXd g_t = step.g;
            
            // Gradiente rispetto a output gate
            // dh_next = grad da loss + grad da timestep successivo
            
            // Gradiente rispetto a tanh(c_t)
            Eigen::MatrixXd dtanh_c = dh_next.array() * o_t.array();
            
            // Gradiente rispetto a c_t
            Eigen::MatrixXd dc_t = dtanh_c.array() * (1 - c_t.array().square());
            dc_t += dc_next;  // aggiungi gradiente da timestep successivo
            
            // Gradienti rispetto ai gates
            Eigen::MatrixXd di_t = dc_t.array() * g_t.array();
            Eigen::MatrixXd df_t = dc_t.array() * c_prev.array();
            Eigen::MatrixXd dg_t = dc_t.array() * i_t.array();
            Eigen::MatrixXd do_t = dh_next.array() * c_t.array().tanh();
            
            // Gradienti rispetto alle pre-attivazioni (considerando derivate delle attivazioni)
            // i = sigmoid(i_gate) -> di/da = i * (1 - i)
            Eigen::MatrixXd di_pre = di_t.array() * i_t.array() * (1 - i_t.array());
            
            // f = sigmoid(f_gate)
            Eigen::MatrixXd df_pre = df_t.array() * f_t.array() * (1 - f_t.array());
            
            // g = tanh(g_gate) -> dg/da = 1 - g^2
            Eigen::MatrixXd dg_pre = dg_t.array() * (1 - g_t.array().square());
            
            // o = sigmoid(o_gate)
            Eigen::MatrixXd do_pre = do_t.array() * o_t.array() * (1 - o_t.array());
            
            // Concatena gradienti dei gates
            Eigen::MatrixXd dgates(batch_size, 4 * hidden_size_);
            dgates.block(0, 0, batch_size, hidden_size_) = di_pre;
            dgates.block(0, hidden_size_, batch_size, hidden_size_) = df_pre;
            dgates.block(0, 2 * hidden_size_, batch_size, hidden_size_) = dg_pre;
            dgates.block(0, 3 * hidden_size_, batch_size, hidden_size_) = do_pre;
            
            // Gradienti pesi
            dWx += x_t.transpose() * dgates;
            dWh += h_t.transpose() * dgates;
            db += dgates.colwise().sum().transpose();
            
            // Gradiente per il prossimo timestep (h_prev)
            dh_next = dgates * Wh_.transpose();
            
            // Gradiente per c_next
            dc_next = dc_t.array() * f_t.array();
        }
        
        // Applica gradient clipping per evitare exploding gradients
        double clip_value = 5.0;
        dWx = dWx.array().min(clip_value).max(-clip_value);
        dWh = dWh.array().min(clip_value).max(-clip_value);
        db = db.array().min(clip_value).max(-clip_value);
        
        // Aggiorna pesi con gradient descent
        Wx_ -= learning_rate * dWx;
        Wh_ -= learning_rate * dWh;
        b_ -= learning_rate * db;
        
        // Calcola gradiente rispetto all'input (per layer precedenti)
        // Questo è approssimato - in realtà dovresti propagare attraverso tutti i timestep
        Eigen::MatrixXd dInput = Eigen::MatrixXd::Zero(batch_size, seq_len * input_size_);
        for (int t = 0; t < seq_len; ++t) {
            const auto& step = lstm_cache->get_step(t + 1);
            Eigen::MatrixXd x_t = step.x;
            
            // Calcola gradiente per questo timestep
            Eigen::MatrixXd dgates_t = ...;  // dovresti ricalcolare o salvare
            
            dInput.block(0, t * input_size_, batch_size, input_size_) = 
                dgates_t * Wx_.transpose();
        }
        
        return dInput;
    }

    // Get config
    std::string LSTMLayer::get_config() const {
        std::ostringstream oss;
        oss << "LSTM(hidden=" << hidden_size_ 
            << ", input=" << input_size_
            << ", seq_len=" << sequence_length_
            << ", params=" << get_parameter_count() << ")";
        return oss.str();
    }

    // Set pesi (versione separata)
    void LSTMLayer::set_weights(const Eigen::MatrixXd& Wx, 
                                const Eigen::MatrixXd& Wh, 
                                const Eigen::VectorXd& b) {
        // Verifica dimensioni
        if (Wx.rows() != input_size_ || Wx.cols() != 4 * hidden_size_) {
            throw ml_exception::DimensionMismatchException(
                "Wx dimensions", input_size_, 4 * hidden_size_,
                Wx.rows(), Wx.cols(), "LSTMLayer");
        }
        
        if (Wh.rows() != hidden_size_ || Wh.cols() != 4 * hidden_size_) {
            throw ml_exception::DimensionMismatchException(
                "Wh dimensions", hidden_size_, 4 * hidden_size_,
                Wh.rows(), Wh.cols(), "LSTMLayer");
        }
        
        if (b.size() != 4 * hidden_size_) {
            throw ml_exception::DimensionMismatchException(
                "b size", 4 * hidden_size_, 1,
                b.size(), 1, "LSTMLayer");
        }
        
        Wx_ = Wx;
        Wh_ = Wh;
        b_ = b;
    }

    // Get pesi (formato flatten)
    Eigen::MatrixXd LSTMLayer::get_weights() const override {
        int total_size = input_size_ * 4 * hidden_size_ + 
                        hidden_size_ * 4 * hidden_size_ + 
                        4 * hidden_size_;
        
        Eigen::MatrixXd weights(total_size, 1);
        
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

    // Set pesi (formato flatten)
    void LSTMLayer::set_weights(const Eigen::MatrixXd& weights) override {
        int expected_size = input_size_ * 4 * hidden_size_ + 
                           hidden_size_ * 4 * hidden_size_ + 
                           4 * hidden_size_;
        
        if (weights.rows() != expected_size || weights.cols() != 1) {
            throw ml_exception::DimensionMismatchException(
                "weights", expected_size, 1,
                weights.rows(), weights.cols(), "LSTMLayer");
        }
        
        int idx = 0;
        
        // Wx
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < 4 * hidden_size_; ++j) {
                Wx_(i, j) = weights(idx++, 0);
            }
        }
        
        // Wh
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < 4 * hidden_size_; ++j) {
                Wh_(i, j) = weights(idx++, 0);
            }
        }
        
        // b
        for (int i = 0; i < 4 * hidden_size_; ++i) {
            b_(i) = weights(idx++, 0);
        }
    }

    // Serializzazione
    void LSTMLayer::serialize(std::ostream& out) const override {
        // Prima serializza i parametri della classe base
        RecurrentLayer::serialize(out);
        
        // Poi serializza i pesi specifici LSTM
        utils::serialize_matrix(out, Wx_);
        utils::serialize_matrix(out, Wh_);
        utils::serialize_vector(out, b_);
    }

    void LSTMLayer::deserialize(std::istream& in) override {
        // Deserializza parametri base
        RecurrentLayer::deserialize(in);
        
        // Deserializza pesi
        Wx_ = utils::deserialize_matrix(in);
        Wh_ = utils::deserialize_matrix(in);
        b_ = utils::deserialize_vector(in);
        
        clear_cache();
    }

    // Inizializzazione pesi (utile per restart)
    void LSTMLayer::initialize_weights() {
        double scale_wx = std::sqrt(2.0 / (input_size_ + hidden_size_));
        double scale_wh = std::sqrt(2.0 / (hidden_size_ + hidden_size_));
        
        Wx_ = Eigen::MatrixXd::Random(input_size_, 4 * hidden_size_) * scale_wx;
        Wh_ = Eigen::MatrixXd::Random(hidden_size_, 4 * hidden_size_) * scale_wh;
        b_ = Eigen::VectorXd::Zero(4 * hidden_size_);
        
        // Re-inizializza bias forget gate
        for (int i = hidden_size_; i < 2 * hidden_size_; ++i) {
            b_(i) = 1.0;
        }
    }

    // Reset stati (per nuovo batch)
    void LSTMLayer::reset_states() {
        h0_ = Eigen::MatrixXd();
        clear_cache();
    }

    // Getter per i gate (utile per debug/analisi)
    LSTMLayer::GateValues LSTMLayer::get_gates(int timestep) const {
        auto* lstm_cache = static_cast<LSTMCache*>(cache_.get());
        
        if (!lstm_cache || timestep < 1 || timestep > lstm_cache->get_sequence_length()) {
            throw std::out_of_range("LSTMLayer: timestep out of range");
        }
        
        const auto& step = lstm_cache->get_step(timestep);
        
        GateValues gates;
        gates.input = step.i;
        gates.forget = step.f;
        gates.output = step.o;
        gates.cell = step.g;
        gates.hidden = step.h;
        gates.cell_state = step.c;
        
        return gates;
    }

    // Calcolo numero parametri
    int LSTMLayer::get_parameter_count() const override {
        // Wx: input_size * 4 * hidden_size
        // Wh: hidden_size * 4 * hidden_size
        // b: 4 * hidden_size
        return (input_size_ * 4 * hidden_size_) + 
               (hidden_size_ * 4 * hidden_size_) + 
               (4 * hidden_size_);
    }

} // namespace layers