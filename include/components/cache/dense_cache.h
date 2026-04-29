// include/components/cache/dense_cache.h
#ifndef DENSE_CACHE_H
#define DENSE_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

class DenseCache : public LayerCache {
public:
    DenseCache() {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
    }
    
    ~DenseCache() override = default;
    
    void clear() override {
        input_cache.resize(0, 0);
        z_cache.resize(0, 0);
        output_cache.resize(0, 0);
    }
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "DenseCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei del forward
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd z_cache;
    Eigen::MatrixXd output_cache;
    
    // Accesso modificabile per i layer
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_z() { return z_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
};

} // namespace layers

#endif