// simple_rnn_cache.h
#ifndef SIMPLE_RNN_CACHE_H
#define SIMPLE_RNN_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>

namespace layers {

class SimpleRNNCache : public RNNCache {
public:
    SimpleRNNCache() : RNNCache() {
        z_values.clear();
    }
    
    ~SimpleRNNCache() override = default;
    
    std::string get_type() const override { return "SimpleRNNCache"; }
    
    void clear() override {
        RNNCache::clear();
        z_values.clear();
    }
    
    // Dati specifici SimpleRNN
    std::vector<Eigen::MatrixXd> z_values;
};

} // namespace layers

#endif