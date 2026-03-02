#include "components/cache/lstm_cache.h"
#include <stdexcept>

namespace layers {

    // Costruttore
    LSTMCache::LSTMCache() 
        : RecurrentCache(), 
          seq_len_(0), 
          batch_size_(0), 
          hidden_size_(0) {}

    // Inizializzazione
    void LSTMCache::init(int sequence_length, int batch_size, int hidden_size) {
        seq_len_ = sequence_length;
        batch_size_ = batch_size;
        hidden_size_ = hidden_size;
        
        steps_.clear();
        steps_.reserve(sequence_length + 1);  // +1 per lo stato iniziale
    }

    // Metodo della classe base - non usato per LSTM
    void LSTMCache::add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                              const Eigen::MatrixXd& input) {
        throw std::runtime_error("LSTMCache: use add_lstm_state() instead of add_state()");
    }

    // Metodo specifico LSTM per aggiungere uno stato
    void LSTMCache::add_lstm_state(int timestep,
                                   const Eigen::MatrixXd& hidden_state,
                                   const Eigen::MatrixXd& cell_state,
                                   const Eigen::MatrixXd& input,
                                   const Eigen::MatrixXd& input_gate,
                                   const Eigen::MatrixXd& forget_gate,
                                   const Eigen::MatrixXd& output_gate,
                                   const Eigen::MatrixXd& candidate_gate) {
        
        LSTMStep step;
        step.h = hidden_state;
        step.c = cell_state;
        step.x = input;
        step.i = input_gate;
        step.f = forget_gate;
        step.o = output_gate;
        step.g = candidate_gate;
        step.timestep = timestep;
        
        steps_.push_back(step);
        
        // Aggiorna anche input/output della BasicCache
        if (timestep == 0) {
            // Il primo step potrebbe essere lo stato iniziale
        } else {
            // Per i timestep successivi, accumula input/output
            // Questo dipende da come vuoi gestire la cache base
        }
    }

    // Getter per stato hidden
    Eigen::MatrixXd LSTMCache::get_hidden_state(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep " + 
                                   std::to_string(timestep) + " out of range");
        }
        return steps_[timestep].h;
    }

    // Getter per stato cella
    Eigen::MatrixXd LSTMCache::get_cell_state(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].c;
    }

    // Getter per input a un timestep
    Eigen::MatrixXd LSTMCache::get_input(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].x;
    }

    // Getter per i gate
    Eigen::MatrixXd LSTMCache::get_input_gate(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].i;
    }

    Eigen::MatrixXd LSTMCache::get_forget_gate(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].f;
    }

    Eigen::MatrixXd LSTMCache::get_output_gate(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].o;
    }

    Eigen::MatrixXd LSTMCache::get_candidate_gate(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep].g;
    }

    // Getter per tutti gli stati hidden (utile per backward)
    std::vector<Eigen::MatrixXd> LSTMCache::get_all_hidden_states() const {
        std::vector<Eigen::MatrixXd> hidden_states;
        hidden_states.reserve(steps_.size());
        for (const auto& step : steps_) {
            hidden_states.push_back(step.h);
        }
        return hidden_states;
    }

    // Getter per tutti gli stati cella
    std::vector<Eigen::MatrixXd> LSTMCache::get_all_cell_states() const {
        std::vector<Eigen::MatrixXd> cell_states;
        cell_states.reserve(steps_.size());
        for (const auto& step : steps_) {
            cell_states.push_back(step.c);
        }
        return cell_states;
    }

    // Getter per uno step completo
    const LSTMCache::LSTMStep& LSTMCache::get_step(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("LSTMCache: timestep out of range");
        }
        return steps_[timestep];
    }

    // Getter per tutti gli step
    const std::vector<LSTMCache::LSTMStep>& LSTMCache::get_all_steps() const {
        return steps_;
    }

    // Numero di timestep
    int LSTMCache::get_sequence_length() const override {
        return seq_len_;
    }

    // Dimensione hidden
    int LSTMCache::get_hidden_size() const override {
        return hidden_size_;
    }

    // Batch size
    int LSTMCache::get_batch_size() const {
        return batch_size_;
    }

    // Clear
    void LSTMCache::clear() override {
        BasicCache::clear();  // Pulisci input/output della BasicCache
        steps_.clear();
        seq_len_ = 0;
        batch_size_ = 0;
        hidden_size_ = 0;
    }

    // Validità
    bool LSTMCache::is_valid() const override {
        // La cache è valida se:
        // 1. BasicCache è valida (ha input/output)
        // 2. Abbiamo il numero corretto di step (seq_len + 1)
        // 3. Le dimensioni sono consistenti
        return BasicCache::is_valid() && 
               seq_len_ > 0 && 
               steps_.size() == static_cast<size_t>(seq_len_ + 1) &&
               batch_size_ > 0 && 
               hidden_size_ > 0;
    }

    // Stampa debug
    void LSTMCache::print_debug() const {
        std::cout << "LSTMCache: seq_len=" << seq_len_
                  << ", batch=" << batch_size_
                  << ", hidden=" << hidden_size_
                  << ", steps=" << steps_.size() << std::endl;
        
        if (!steps_.empty()) {
            std::cout << "  Step 0 - h: " << steps_[0].h.rows() << "x" << steps_[0].h.cols()
                      << ", c: " << steps_[0].c.rows() << "x" << steps_[0].c.cols() << std::endl;
        }
    }

} // namespace layers
