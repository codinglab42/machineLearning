#include "components/cache/weighted_cache.h"

namespace layers {

    WeightedCache::WeightedCache() : BasicCache() {
        z_.resize(0, 0);
    }

    void WeightedCache::clear() {
        BasicCache::clear();
        z_.resize(0, 0);
    }

    bool WeightedCache::is_valid() const {
        // Per layer con pesi, deve avere anche z (pre-attivazione)
        return BasicCache::is_valid() && z_.size() > 0;
    }

} // namespace layers
