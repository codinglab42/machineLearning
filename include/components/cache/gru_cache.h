// gru_cache.h
#ifndef GRU_CACHE_H
#define GRU_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

class GRUCache : public RNNCache {
public:
    GRUCache() : RNNCache() {
        reset_gates.clear();
        update_gates.clear();
        candidate_hidden.clear();
        z_r.clear();
        z_z.clear();
        z_h.clear();
    }
    
    ~GRUCache() override = default;
    
    std::string get_type() const override { return "GRUCache"; }
    
    void clear() override {
        RNNCache::clear();
        reset_gates.clear();
        update_gates.clear();
        candidate_hidden.clear();
        z_r.clear();
        z_z.clear();
        z_h.clear();
    }
    
    // Dati specifici GRU
    std::vector<Eigen::MatrixXd> reset_gates;
    std::vector<Eigen::MatrixXd> update_gates;
    std::vector<Eigen::MatrixXd> candidate_hidden;
    
    std::vector<Eigen::MatrixXd> z_r;
    std::vector<Eigen::MatrixXd> z_z;
    std::vector<Eigen::MatrixXd> z_h;
};

} // namespace layers

#endif