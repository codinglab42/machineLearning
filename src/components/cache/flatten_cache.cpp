#include "components/cache/flatten_cache.h"
#include <stdexcept>

namespace layers {

    FlattenCache::FlattenCache() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        original_shape.clear();
    }

    void FlattenCache::clear() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        original_shape.clear();
    }

    bool FlattenCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        
        if (original_shape.size() != 2) {
            return false;
        }
        
        return validate_dimensions();
    }

    bool FlattenCache::validate_dimensions() const {
        // Verifica che input e output abbiano le stesse dimensioni totali
        if (input_cache.rows() != output_cache.rows() || 
            input_cache.cols() != output_cache.cols()) {
            return false;
        }
        
        // Verifica che lo shape originale sia coerente con l'input
        int expected_elements = original_shape[0] * original_shape[1];
        if (input_cache.rows() * input_cache.cols() != expected_elements) {
            return false;
        }
        
        return true;
    }

} // namespace layers

