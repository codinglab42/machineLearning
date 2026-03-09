#include "components/cache/pooling_cache.h"

namespace layers {

    PoolingCache::PoolingCache() 
        : training_(false), input_height(0), input_width(0), channels(0) {
        input.resize(0, 0);
        output.resize(0, 0);
        max_indices_.clear();
    }

    void PoolingCache::clear() {
        input.resize(0, 0);
        output.resize(0, 0);
        max_indices_.clear();
        training_ = false;
        input_height = 0;
        input_width = 0;
        channels = 0;
    }

    bool PoolingCache::is_valid() const {
        if (input.size() == 0 || output.size() == 0) {
            return false;
        }
        return validate_dimensions();
    }

    bool PoolingCache::validate_dimensions() const {
        if (input.rows() <= 0 || input.cols() <= 0) return false;
        if (output.rows() <= 0 || output.cols() <= 0) return false;
        if (input.rows() != output.rows()) return false;
        return true;
    }

} // namespace layers

