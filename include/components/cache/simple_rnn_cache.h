#ifndef SIMPLE_RNN_CACHE_H
#define SIMPLE_RNN_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>

namespace layers {

    class SimpleRNNCache : public RNNCache {
    public:
        SimpleRNNCache();
        ~SimpleRNNCache() override = default;
        
        std::string get_type() const override { return "SimpleRNNCache"; }
        
        // Dati specifici SimpleRNN
        std::vector<Eigen::MatrixXd> z_values;  // Valori pre-attivazione per gate
    };

} // namespace layers

#endif
