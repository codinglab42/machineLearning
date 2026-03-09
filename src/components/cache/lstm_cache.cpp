#include "components/cache/lstm_cache.h"

namespace layers {

    LSTMCache::LSTMCache() : RNNCache() {
        cell_states.clear();
        input_gates.clear();
        forget_gates.clear();
        output_gates.clear();
        cell_candidates.clear();
        z_i.clear();
        z_f.clear();
        z_o.clear();
        z_c.clear();
    }

} // namespace layers

