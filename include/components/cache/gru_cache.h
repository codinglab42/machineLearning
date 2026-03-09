#ifndef GRU_CACHE_H
#define GRU_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class GRUCache : public RNNCache {
    public:
        GRUCache();
        ~GRUCache() override = default;
        
        std::string get_type() const override { return "GRUCache"; }
        
        // Dati specifici GRU
        std::vector<Eigen::MatrixXd> reset_gates;   // Gate di reset
        std::vector<Eigen::MatrixXd> update_gates;  // Gate di update
        std::vector<Eigen::MatrixXd> candidate_hidden; // Candidato stato nascosto
        
        std::vector<Eigen::MatrixXd> z_r;  // Pre-attivazione reset gate
        std::vector<Eigen::MatrixXd> z_z;  // Pre-attivazione update gate
        std::vector<Eigen::MatrixXd> z_h;  // Pre-attivazione candidate
    };

} // namespace layers

#endif

