// lstm_cache.h
#ifndef LSTM_CACHE_H
#define LSTM_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

class LSTMCache : public RNNCache {
public:
    LSTMCache() : RNNCache() {
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
    
    ~LSTMCache() override = default;
    
    std::string get_type() const override { return "LSTMCache"; }
    
    void clear() override {
        RNNCache::clear();
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
    
    // Dati specifici LSTM
    std::vector<Eigen::MatrixXd> cell_states;
    std::vector<Eigen::MatrixXd> input_gates;
    std::vector<Eigen::MatrixXd> forget_gates;
    std::vector<Eigen::MatrixXd> output_gates;
    std::vector<Eigen::MatrixXd> cell_candidates;
    
    std::vector<Eigen::MatrixXd> z_i;
    std::vector<Eigen::MatrixXd> z_f;
    std::vector<Eigen::MatrixXd> z_o;
    std::vector<Eigen::MatrixXd> z_c;
};

} // namespace layers

#endif