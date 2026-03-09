#include "components/cache/rnn_cache.h"

namespace layers {

    RNNCache::RNNCache() 
        : timesteps(0), batch_size(0), input_size(0), hidden_size(0), training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        hidden_states.clear();
        pre_activations.clear();
    }

    void RNNCache::clear() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        hidden_states.clear();
        pre_activations.clear();
        
        timesteps = 0;
        batch_size = 0;
        input_size = 0;
        hidden_size = 0;
        training = false;
    }

    bool RNNCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        return validate_dimensions();
    }

    bool RNNCache::validate_dimensions() const {
        if (input_cache.rows() != batch_size * timesteps || 
            input_cache.cols() != input_size) {
            return false;
        }
        
        if (output_cache.rows() != batch_size * timesteps || 
            output_cache.cols() != hidden_size) {
            return false;
        }
        
        if (training) {
            if (hidden_states.size() != static_cast<size_t>(timesteps + 1)) {
                return false;
            }
            if (pre_activations.size() != static_cast<size_t>(timesteps)) {
                return false;
            }
        }
        
        return true;
    }

} // namespace layers

