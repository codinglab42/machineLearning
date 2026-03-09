#include "components/cache/dense_cache.h"
#include "exceptions/exception_macros.h"

namespace layers {

    DenseCache::DenseCache() {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
    }

    void DenseCache::clear() {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
    }

    bool DenseCache::is_valid() const {
        if (input_cache.size() == 0 || output_cache.size() == 0) {
            return false;
        }
        return validate_dimensions();
    }

    bool DenseCache::validate_dimensions() const {
        if (input_cache.rows() != output_cache.rows()) {
            return false;
        }
        if (z_cache.rows() != output_cache.rows() || z_cache.cols() != output_cache.cols()) {
            return false;
        }
        return true;
    }

} // namespace layers

