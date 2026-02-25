#include "components/cache/basic_cache.h"

namespace layers {

    BasicCache::BasicCache() : has_activation_(false) {}

    void BasicCache::clear() {
        input_.resize(0, 0);
        output_.resize(0, 0);
        has_activation_ = false;
    }

    bool BasicCache::is_valid() const {
        return input_.size() > 0 && output_.size() > 0;
    }

} // namespace layers
