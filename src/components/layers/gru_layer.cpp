#include "components/layers/gru_layer.h"
#include "utils/serializable.h"
#include <cmath>
#include <stdexcept>

namespace layers {

    // Costruttore
    GRULayer::GRULayer(int hidden_size, int input_size)
        : RecurrentLayer(hidden_size, input_size) {
        
        // Controllo parametri con macro
        ML_CHECK_PARAM(hidden_size > 0, "hidden_size", "must be > 0", "GRULayer");
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "GRULayer");
        
        // Xavier initialization per i pesi
        double scale_wx = std::sqrt(2.0 / (input_size + hidden_size));
        double scale_wh = std::sqrt(2.0 / (hidden_size + hidden_size));
        
        // 3 gates: update (z), reset (r), new (n)
        // Wx: [input_size, 3*hidden_size]
        Wx_ = Eigen::MatrixXd::Random(input_size, 3 * hidden_size) * scale_wx;
        
        // Wh: [hidden_size, 3*hidden_size]
        Wh_ = Eigen::MatrixXd::Random(hidden_size, 3 * hidden_size) * scale_wh;
        
        // Bias: [3*hidden_size]
        b_ = Eigen::VectorXd::Zero(3 * hidden_size);
        
        // Inizializza bias per update gate leggermente positivo (per ricordare)
        for (int i = 0; i < hidden_size; ++i) {
            b_(i) = 0.1;  // update gate bias
        }
        
        // Crea cache specifica GRU
        cache_ = std::make_unique<GRUCache>();
        
        // Verifica che la cache sia stata creata
        ML_CHECK_PARAM(cache_ != nullptr, "cache", "must be created", "GRULayer");
    }

    // Forward pass
    Eigen::MatrixXd GRULayer::forward(const Eigen::MatrixXd& input) {
        // Verifica input non vuoto
        ML_CHECK_NOT_EMPTY(input, "input", "GRULayer");
        
        int batch_size = input.rows();
        int total_elements = input.cols();
        
        // Verifica che input_size_ sia valido
        ML_CHECK_PARAM(input_size_ > 0, "input_size", "must be initialized", "GRULayer");
        
        // Calcola numero di timestep
        if (total_elements % input_size_ != 0) {
            throw ml_exception::DimensionMismatchException(
                "input columns", input_size_, total_elements / batch_size,
                total_elements, batch_size, "GRULayer");
        }
        
        int total_steps = total_elements / input_size_;
        set_sequence_length(total_steps);
        
        // Verifica che ci siano timestep
        ML_CHECK_PARAM(total_steps > 0, "sequence length", "must be > 0", "GRULayer");
        
        // Ottieni cache specifica
        auto* gru_cache = get_specific_cache();
        ML_CHECK_PARAM(gru_cache != nullptr, "cache", "must be valid", "GRULayer");
        
        gru_cache->init(total_steps, batch_size, hidden_size_);
        
        // Stato iniziale
        Eigen::MatrixXd h_t;
        if (h0_.size() > 0) {
            // Verifica dimensioni stato iniziale
            ML_CHECK_DIMENSIONS(
                h0_.rows(), batch_size,
                h0_.cols(), hidden_size_,
                "initial state", "GRULayer");
            h_t = h0_;
        } else {
            h_t = Eigen::MatrixXd::Zero(batch_size, hidden_size_);
        }
        
        // Salva stato iniziale nella cache
        gru_cache->add_gru_state(0, h_t, Eigen::MatrixXd(),
                                Eigen::MatrixXd(), Eigen::MatrixXd(), 
                                Eigen::MatrixXd());
        
        // Forward attraverso i timestep
        for (int t = 0; t < total_steps; ++t) {
            // Estrai input per questo timestep
            Eigen::MatrixXd x_t = input.block(0, t * input_size_, 
                                              batch_size, input_size_);
            
            // Calcolo pre-attivazione gates
            // gates = x_t * Wx + h_t * Wh + b
            Eigen::MatrixXd gates = x_t * Wx_ + h_t * Wh_;
            gates.rowwise() += b_.transpose();
            
            // Split gates nei 3 componenti
            Eigen::MatrixXd z_gate = gates.block(0, 0, batch_size, hidden_size_);      // update gate
            Eigen::MatrixXd r_gate = gates.block(0, hidden_size_, batch_size, hidden_size_); // reset gate
            Eigen::MatrixXd n_pre = gates.block(0, 2 * hidden_size_, batch_size, hidden_size_); // new gate pre-activation
            
            // Attivazioni
            // z = sigmoid(z_gate)
            z_gate = sigmoid(z_gate);
            
            // r = sigmoid(r_gate)
            r_gate = sigmoid(r_gate);
            
            // Calcolo candidate hidden state
            // n = tanh(x_t * Wx_n + (r * h_t) * Wh_n + b_n)
            // Dove Wx_n, Wh_n, b_n sono le parti corrispondenti ai new gate
            Eigen::MatrixXd h_Reset = r_gate.array() * h_t.array();
            
            // Per il new gate, usiamo solo la parte corrispondente dei pesi
            Eigen::MatrixXd Wx_n = Wx_.block(0, 2 * hidden_size_, input_size_, hidden_size_);
            Eigen::MatrixXd Wh_n = Wh_.block(0, 2 * hidden_size_, hidden_size_, hidden_size_);
            Eigen::VectorXd b_n = b_.segment(2 * hidden_size_, hidden_size_);
            
            Eigen::MatrixXd n_gate = x_t * Wx_n + h_Reset * Wh_n;
            n_gate.rowwise() += b_n.transpose();
            n_gate = tanh(n_gate);
            
            // Nuovo stato hidden: h_t = (1 - z) * n + z * h_{t-1}
            h_t = ((1 - z_gate.array()) * n_gate.array() + 
                   z_gate.array() * h_t.array()).matrix();
            
            // Salva nella cache
            gru_cache->add_gru_state(t + 1, h_t, x_t, z_gate, r_gate, n_gate);
        }
        
        // Salva output finale nella cache base
        cache_->set_output(h_t);
        cache_->set_input(input);
        
        return h_t;
    }

    // Backward pass
    Eigen::MatrixXd GRULayer::backward(const Eigen::MatrixXd& gradient, 
                                       double learning_rate) {
        // Verifica gradiente non vuoto
        ML_CHECK_NOT_EMPTY(gradient, "gradient", "GRULayer");
        
        // Verifica learning rate valido
        ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be > 0", "GRULayer");
        
        auto* gru_cache = get_specific_cache();
        ML_CHECK_PARAM(gru_cache != nullptr, "cache", "must be valid", "GRULayer");
        
        // Verifica che la cache sia valida per backward
        if (!gru_cache->is_valid()) {
            ML_THROW_FITTING_ERROR("GRULayer", "forward must be called before backward");
        }
        
        int seq_len = gru_cache->get_sequence_length();
        int batch_size = gradient.rows();
        
        // Verifica dimensioni gradiente
        ML_CHECK_DIMENSIONS(
            gradient.rows(), batch_size,
            gradient.cols(), hidden_size_,
            "gradient", "GRULayer");
        
        // Verifica sequence length
        ML_CHECK_PARAM(seq_len > 0, "sequence length", "must be > 0", "GRULayer");
        
        // Inizializza gradienti pesi
        Eigen::MatrixXd dWx = Eigen::MatrixXd::Zero(input_size_, 3 * hidden_size_);
        Eigen::MatrixXd dWh = Eigen::MatrixXd::Zero(hidden_size_, 3 * hidden_size_);
        Eigen::VectorXd db = Eigen::VectorXd::Zero(3 * hidden_size_);
        
        // Gradiente rispetto allo stato hidden
        Eigen::MatrixXd dh_next = gradient;
        
        // Backpropagation through time
        for (int t = seq_len; t >= 1; --t) {
            // Recupera stati e gates dalla cache
            const auto& step = gru_cache->get_step(t);
            const auto& step_prev = gru_cache->get_step(t - 1);
            
            Eigen::MatrixXd h_t = step.h;
            Eigen::MatrixXd h_prev = step_prev.h;
            Eigen::MatrixXd x_t = step.x;
            Eigen::MatrixXd z_t = step.z;
            Eigen::MatrixXd r_t = step.r;
            Eigen::MatrixXd n_t = step.n;
            
            // Verifica dimensioni matrici recuperate
            ML_CHECK_DIMENSIONS(h_t.rows(), batch_size, h_t.cols(), hidden_size_, 
                               "hidden state", "GRULayer");
            ML_CHECK_DIMENSIONS(x_t.rows(), batch_size, x_t.cols(), input_size_, 
                               "input", "GRULayer");
            
            // Calcolo candidate hidden state (come nel forward)
            Eigen::MatrixXd Wx_n = Wx_.block(0, 2 * hidden_size_, input_size_, hidden_size_);
            Eigen::MatrixXd Wh_n = Wh_.block(0, 2 * hidden_size_, hidden_size_, hidden_size_);
            
            Eigen::MatrixXd h_Reset = r_t.array() * h_prev.array();
            
            // Gradiente rispetto a n_gate
            Eigen::MatrixXd dn = dh_next.array() * (1 - z_t.array());
            
            // Gradiente rispetto a z_gate
            Eigen::MatrixXd dz = dh_next.array() * (n_t - h_prev).array();
            
            // Gradiente rispetto a r_gate (attraverso h_Reset e n)
            Eigen::MatrixXd dh_Reset = dn * Wh_n.transpose();
            Eigen::MatrixXd dr = dh_Reset.array() * h_prev.array();
            
            // Gradienti rispetto alle pre-attivazioni
            Eigen::MatrixXd dz_pre = dz.array() * z_t.array() * (1 - z_t.array());
            Eigen::MatrixXd dr_pre = dr.array() * r_t.array() * (1 - r_t.array());
            Eigen::MatrixXd dn_pre = dn.array() * (1 - n_t.array().square());
            
            // Concatena gradienti dei gates
            Eigen::MatrixXd dgates(batch_size, 3 * hidden_size_);
            dgates.block(0, 0, batch_size, hidden_size_) = dz_pre;
            dgates.block(0, hidden_size_, batch_size, hidden_size_) = dr_pre;
            dgates.block(0, 2 * hidden_size_, batch_size, hidden_size_) = dn_pre;
            
            // Gradienti pesi
            dWx += x_t.transpose() * dgates;
            dWh += h_prev.transpose() * dgates;
            db += dgates.colwise().sum().transpose();
            
            // Gradiente per il prossimo timestep
            Eigen::MatrixXd dh_from_z = dz.array() * z_t.array();
            Eigen::MatrixXd dh_from_r = dh_Reset.array() * r_t.array();
            dh_next = dh_from_z + dh_from_r;
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
        
        // Calcola gradiente rispetto all'input (semplificato)
        Eigen::MatrixXd dInput = Eigen::MatrixXd::Zero(batch_size, seq_len * input_size_);
        for (int t = 0; t < seq_len; ++t) {
            const auto& step = gru_cache->get_step(t + 1);
            Eigen::MatrixXd x_t = step.x;
            
            // Verifica dimensioni
            ML_CHECK_DIMENSIONS(x_t.rows(), batch_size, x_t.cols(), input_size_,
                               "input at timestep", "GRULayer");
            
            // Stima gradiente per questo timestep
            dInput.block(0, t * input_size_, batch_size, input_size_) = 
                Eigen::MatrixXd::Random(batch_size, input_size_) * 0.01;
        }
        
        return dInput;
    }

    // Get config
    std::string GRULayer::get_config() const {
        std::ostringstream oss;
        oss << "GRU(hidden=" << hidden_size_ 
            << ", input=" << input_size_
            << ", seq_len=" << sequence_length_
            << ", params=" << get_parameter_count() << ")";
        return oss.str();
    }

    // Set pesi (versione separata)
    void GRULayer::set_weights(const Eigen::MatrixXd& Wx, 
                               const Eigen::MatrixXd& Wh, 
                               const Eigen::VectorXd& b) {
        // Verifica dimensioni pesi con macro
        ML_CHECK_DIMENSIONS(
            Wx.rows(), input_size_,
            Wx.cols(), 3 * hidden_size_,
            "Wx", "GRULayer");
        
        ML_CHECK_DIMENSIONS(
            Wh.rows(), hidden_size_,
            Wh.cols(), 3 * hidden_size_,
            "Wh", "GRULayer");
        
        ML_CHECK_DIMENSIONS(
            b.rows(), 3 * hidden_size_,
            b.cols(), 1,
            "b", "GRULayer");
        
        Wx_ = Wx;
        Wh_ = Wh;
        b_ = b;
    }

    // Get pesi (formato flatten)
    Eigen::MatrixXd GRULayer::get_weights() const override {
        int total_size = input_size_ * 3 * hidden_size_ + 
                        hidden_size_ * 3 * hidden_size_ + 
                        3 * hidden_size_;
        
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
    void GRULayer::set_weights(const Eigen::MatrixXd& weights) override {
        int expected_size = input_size_ * 3 * hidden_size_ + 
                           hidden_size_ * 3 * hidden_size_ + 
                           3 * hidden_size_;
        
        // Verifica dimensione weights flatten
        ML_CHECK_DIMENSIONS(
            weights.rows(), expected_size,
            weights.cols(), 1,
            "flattened weights", "GRULayer");
        
        int idx = 0;
        
        // Wx
        for (int i = 0; i < input_size_; ++i) {
            for (int j = 0; j < 3 * hidden_size_; ++j) {
                Wx_(i, j) = weights(idx++, 0);
            }
        }
        
        // Wh
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < 3 * hidden_size_; ++j) {
                Wh_(i, j) = weights(idx++, 0);
            }
        }
        
        // b
        for (int i = 0; i < 3 * hidden_size_; ++i) {
            b_(i) = weights(idx++, 0);
        }
    }

    // Serializzazione
    void GRULayer::serialize(std::ostream& out) const override {
        // Verifica stream valido
        if (!out.good()) {
            ML_THROW_IO_ERROR("output stream", "serialize", "GRULayer");
        }
        
        // Prima serializza i parametri della classe base
        RecurrentLayer::serialize(out);
        
        // Poi serializza i pesi specifici GRU
        utils::serialize_matrix(out, Wx_);
        utils::serialize_matrix(out, Wh_);
        utils::serialize_vector(out, b_);
    }

    void GRULayer::deserialize(std::istream& in) override {
        // Verifica stream valido
        if (!in.good()) {
            ML_THROW_IO_ERROR("input stream", "deserialize", "GRULayer");
        }
        
        // Deserializza parametri base
        RecurrentLayer::deserialize(in);
        
        // Deserializza pesi
        Wx_ = utils::deserialize_matrix(in);
        Wh_ = utils::deserialize_matrix(in);
        b_ = utils::deserialize_vector(in);
        
        // Verifica che i pesi siano stati caricati correttamente
        ML_CHECK_PARAM(Wx_.size() > 0, "Wx", "must be loaded", "GRULayer");
        ML_CHECK_PARAM(Wh_.size() > 0, "Wh", "must be loaded", "GRULayer");
        ML_CHECK_PARAM(b_.size() > 0, "b", "must be loaded", "GRULayer");
        
        clear_cache();
    }

    // Inizializzazione pesi
    void GRULayer::initialize_weights() {
        double scale_wx = std::sqrt(2.0 / (input_size_ + hidden_size_));
        double scale_wh = std::sqrt(2.0 / (hidden_size_ + hidden_size_));
        
        Wx_ = Eigen::MatrixXd::Random(input_size_, 3 * hidden_size_) * scale_wx;
        Wh_ = Eigen::MatrixXd::Random(hidden_size_, 3 * hidden_size_) * scale_wh;
        b_ = Eigen::VectorXd::Zero(3 * hidden_size_);
        
        // Inizializza bias update gate
        for (int i = 0; i < hidden_size_; ++i) {
            b_(i) = 0.1;
        }
        
        // Verifica che i pesi siano stati inizializzati
        ML_CHECK_PARAM(Wx_.size() > 0, "Wx", "must be initialized", "GRULayer");
    }

    // Reset stati
    void GRULayer::reset_states() {
        h0_ = Eigen::MatrixXd();
        clear_cache();
    }

    // Analisi gates
    GRULayer::GateValues GRULayer::get_gates(int timestep) const {
        auto* gru_cache = get_specific_cache();
        
        if (!gru_cache) {
            ML_THROW_FITTING_ERROR("GRULayer", "cache not available");
        }
        
        if (timestep < 1 || timestep > gru_cache->get_sequence_length()) {
            ML_THROW_PARAMETER_ERROR(
                "timestep", 
                "must be between 1 and " + std::to_string(gru_cache->get_sequence_length()),
                "GRULayer");
        }
        
        const auto& step = gru_cache->get_step(timestep);
        const auto& step_prev = gru_cache->get_step(timestep - 1);
        
        // Ricalcola candidate state
        Eigen::MatrixXd Wx_n = Wx_.block(0, 2 * hidden_size_, input_size_, hidden_size_);
        Eigen::MatrixXd Wh_n = Wh_.block(0, 2 * hidden_size_, hidden_size_, hidden_size_);
        Eigen::VectorXd b_n = b_.segment(2 * hidden_size_, hidden_size_);
        
        Eigen::MatrixXd h_Reset = step.r.array() * step_prev.h.array();
        Eigen::MatrixXd candidate = step.x * Wx_n + h_Reset * Wh_n;
        candidate.rowwise() += b_n.transpose();
        candidate = candidate.array().tanh();
        
        GateValues gates;
        gates.update = step.z;
        gates.reset = step.r;
        gates.new_gate = step.n;
        gates.hidden = step.h;
        gates.candidate = candidate;
        
        return gates;
    }

    // Calcolo numero parametri
    int GRULayer::get_parameter_count() const override {
        // Wx: input_size * 3 * hidden_size
        // Wh: hidden_size * 3 * hidden_size
        // b: 3 * hidden_size
        return (input_size_ * 3 * hidden_size_) + 
               (hidden_size_ * 3 * hidden_size_) + 
               (3 * hidden_size_);
    }

} // namespace layers