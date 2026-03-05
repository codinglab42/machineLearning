#include "components/cache/dropout_cache.h"
#include <stdexcept>

namespace layers {

    DropoutCache::DropoutCache() : training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        mask.resize(0, 0);
    }

    void DropoutCache::clear() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        mask.resize(0, 0);
        training = false;
    }

    bool DropoutCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        
        return validate_dimensions();
    }

    bool DropoutCache::validate_dimensions() const {
        // In training mode, deve esistere la maschera
        if (training) {
            if (mask.size() == 0) {
                return false;
            }
            
            // La maschera deve avere le stesse dimensioni dell'input
            if (mask.rows() != input_cache.rows() || mask.cols() != input_cache.cols()) {
                return false;
            }
        }
        
        // Input e output devono avere stesse dimensioni
        if (input_cache.rows() != output_cache.rows() || 
            input_cache.cols() != output_cache.cols()) {
            return false;
        }
        
        return true;
    }

} // namespace layers
