#include "components/cache/simple_rnn_cache.h"
#include <stdexcept>
#include <iostream>

namespace layers {

    // Costruttore
    SimpleRNNCache::SimpleRNNCache() 
        : RecurrentCache(), 
          seq_len_(0), 
          batch_size_(0), 
          hidden_size_(0) {}

    // Inizializzazione
    void SimpleRNNCache::init(int sequence_length, int batch_size, int hidden_size) {
        seq_len_ = sequence_length;
        batch_size_ = batch_size;
        hidden_size_ = hidden_size;
        
        steps_.clear();
        steps_.reserve(sequence_length + 1);  // +1 per lo stato iniziale
    }

    // Metodo della classe base - non usato per SimpleRNN
    void SimpleRNNCache::add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                                   const Eigen::MatrixXd& input) {
        throw std::runtime_error("SimpleRNNCache: use add_rnn_state() instead of add_state()");
    }

    // Metodo specifico SimpleRNN per aggiungere uno stato
    void SimpleRNNCache::add_rnn_state(int timestep,
                                       const Eigen::MatrixXd& hidden_state,
                                       const Eigen::MatrixXd& input,
                                       const Eigen::MatrixXd& pre_act) {
        
        RNNStep step;
        step.h = hidden_state;
        step.x = input;
        step.pre_activation = pre_act;
        step.timestep = timestep;
        
        steps_.push_back(step);
    }

    // Getter per stato hidden
    Eigen::MatrixXd SimpleRNNCache::get_hidden_state(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("SimpleRNNCache: timestep " + 
                                   std::to_string(timestep) + " out of range");
        }
        return steps_[timestep].h;
    }

    // Getter per input
    Eigen::MatrixXd SimpleRNNCache::get_input(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("SimpleRNNCache: timestep out of range");
        }
        return steps_[timestep].x;
    }

    // Getter per pre-attivazione
    Eigen::MatrixXd SimpleRNNCache::get_pre_activation(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("SimpleRNNCache: timestep out of range");
        }
        return steps_[timestep].pre_activation;
    }

    // Getter per tutti gli stati hidden
    std::vector<Eigen::MatrixXd> SimpleRNNCache::get_all_hidden_states() const {
        std::vector<Eigen::MatrixXd> hidden_states;
        hidden_states.reserve(steps_.size());
        for (const auto& step : steps_) {
            hidden_states.push_back(step.h);
        }
        return hidden_states;
    }

    // Getter per tutti gli input
    std::vector<Eigen::MatrixXd> SimpleRNNCache::get_all_inputs() const {
        std::vector<Eigen::MatrixXd> inputs;
        inputs.reserve(steps_.size());
        for (const auto& step : steps_) {
            inputs.push_back(step.x);
        }
        return inputs;
    }

    // Getter per tutte le pre-attivazioni
    std::vector<Eigen::MatrixXd> SimpleRNNCache::get_all_pre_activations() const {
        std::vector<Eigen::MatrixXd> pre_acts;
        pre_acts.reserve(steps_.size());
        for (const auto& step : steps_) {
            pre_acts.push_back(step.pre_activation);
        }
        return pre_acts;
    }

    // Getter per uno step completo
    const SimpleRNNCache::RNNStep& SimpleRNNCache::get_step(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("SimpleRNNCache: timestep " + 
                                   std::to_string(timestep) + " out of range");
        }
        return steps_[timestep];
    }

    // Getter per tutti gli step
    const std::vector<SimpleRNNCache::RNNStep>& SimpleRNNCache::get_all_steps() const {
        return steps_;
    }

    // Getter per sequence length
    int SimpleRNNCache::get_sequence_length() const override {
        return seq_len_;
    }

    // Getter per hidden size
    int SimpleRNNCache::get_hidden_size() const override {
        return hidden_size_;
    }

    // Getter per batch size
    int SimpleRNNCache::get_batch_size() const {
        return batch_size_;
    }

    // Clear
    void SimpleRNNCache::clear() override {
        BasicCache::clear();
        steps_.clear();
        seq_len_ = 0;
        batch_size_ = 0;
        hidden_size_ = 0;
    }

    // Validità
    bool SimpleRNNCache::is_valid() const override {
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
    void SimpleRNNCache::print_debug() const {
        std::cout << "SimpleRNNCache: seq_len=" << seq_len_
                  << ", batch=" << batch_size_
                  << ", hidden=" << hidden_size_
                  << ", steps=" << steps_.size() << std::endl;
        
        if (!steps_.empty()) {
            std::cout << "  Step 0 - h: " << steps_[0].h.rows() << "x" << steps_[0].h.cols()
                      << ", x: " << steps_[0].x.rows() << "x" << steps_[0].x.cols() << std::endl;
        }
    }

} // namespace layers
