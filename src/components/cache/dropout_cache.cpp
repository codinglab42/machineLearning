#include "components/cache/dropout_cache.h"

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
        if (input_cache.rows() != output_cache.rows() || 
            input_cache.cols() != output_cache.cols()) {
            return false;
        }
        
        if (training && mask.size() > 0) {
            if (mask.rows() != input_cache.rows() || mask.cols() != input_cache.cols()) {
                return false;
            }
        }
        
        return true;
    }

} // namespace layers

