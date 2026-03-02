#include "components/cache/gru_cache.h"

namespace layers {

    GRUCache::GRUCache() 
        : RecurrentCache(), 
          seq_len_(0), 
          batch_size_(0), 
          hidden_size_(0) {}

    void GRUCache::init(int sequence_length, int batch_size, int hidden_size) {
        seq_len_ = sequence_length;
        batch_size_ = batch_size;
        hidden_size_ = hidden_size;
        
        steps_.clear();
        steps_.reserve(sequence_length + 1);
    }

    void GRUCache::add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                             const Eigen::MatrixXd& input) {
        // Questo metodo non dovrebbe essere chiamato per GRU
        throw std::runtime_error("GRUCache: use add_gru_state() instead of add_state()");
    }

    void GRUCache::add_gru_state(int timestep,
                                 const Eigen::MatrixXd& hidden_state,
                                 const Eigen::MatrixXd& input,
                                 const Eigen::MatrixXd& update_gate,
                                 const Eigen::MatrixXd& reset_gate,
                                 const Eigen::MatrixXd& new_gate) {
        
        GRUStep step;
        step.h = hidden_state;
        step.x = input;
        step.z = update_gate;
        step.r = reset_gate;
        step.n = new_gate;
        
        steps_.push_back(step);
    }

    Eigen::MatrixXd GRUCache::get_hidden_state(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            return Eigen::MatrixXd();
        }
        return steps_[timestep].h;
    }

    const GRUCache::GRUStep& GRUCache::get_step(int timestep) const {
        if (timestep < 0 || timestep >= static_cast<int>(steps_.size())) {
            throw std::out_of_range("GRUCache: timestep out of range");
        }
        return steps_[timestep];
    }

    void GRUCache::clear() {
        BasicCache::clear();
        steps_.clear();
        seq_len_ = 0;
        batch_size_ = 0;
        hidden_size_ = 0;
    }

    bool GRUCache::is_valid() const {
        return BasicCache::is_valid() && 
               seq_len_ > 0 && 
               steps_.size() == static_cast<size_t>(seq_len_ + 1);
    }

} // namespace layers
