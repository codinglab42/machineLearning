// include/components/cache/dropout_cache.h
#ifndef DROPOUT_CACHE_H
#define DROPOUT_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

class DropoutCache : public LayerCache {
public:
    DropoutCache() : training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        mask.resize(0, 0);
    }
    
    ~DropoutCache() override = default;
    
    void clear() override {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        mask.resize(0, 0);
        training = false;
    }
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "DropoutCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
    Eigen::MatrixXd mask;
    bool training;
    
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
    Eigen::MatrixXd& mutable_mask() { return mask; }
};

} // namespace layers

#endif