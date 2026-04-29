// include/components/cache/flatten_cache.h
#ifndef FLATTEN_CACHE_H
#define FLATTEN_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

class FlattenCache : public LayerCache {
public:
    FlattenCache() {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        original_shape.clear();
    }
    
    ~FlattenCache() override = default;
    
    void clear() override {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        original_shape.clear();
    }
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "FlattenCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
    std::vector<int> original_shape;
    
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
    std::vector<int>& mutable_shape() { return original_shape; }
};

} // namespace layers

#endif