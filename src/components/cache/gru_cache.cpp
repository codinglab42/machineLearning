#include "components/cache/gru_cache.h"

namespace layers {

    GRUCache::GRUCache() : RNNCache() {
        reset_gates.clear();
        update_gates.clear();
        candidate_hidden.clear();
        z_r.clear();
        z_z.clear();
        z_h.clear();
    }

} // namespace layers
