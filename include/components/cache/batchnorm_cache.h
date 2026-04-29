// include/components/cache/batchnorm_cache.h
#ifndef BATCH_NORM_CACHE_H
#define BATCH_NORM_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

class BatchNormCache : public LayerCache {
public:
    BatchNormCache() : training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        x_centered.resize(0, 0);
        x_norm.resize(0, 0);
        batch_mean.resize(0);
        batch_var.resize(0);
        inv_std.resize(0);
    }
    
    ~BatchNormCache() override = default;
    
    void clear() override {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        x_centered.resize(0, 0);
        x_norm.resize(0, 0);
        batch_mean.resize(0);
        batch_var.resize(0);
        inv_std.resize(0);
        training = false;
    }
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "BatchNormCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
    Eigen::MatrixXd x_centered;
    Eigen::MatrixXd x_norm;
    Eigen::VectorXd batch_mean;
    Eigen::VectorXd batch_var;
    Eigen::VectorXd inv_std;
    bool training;
    
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
    Eigen::MatrixXd& mutable_x_centered() { return x_centered; }
    Eigen::MatrixXd& mutable_x_norm() { return x_norm; }
    Eigen::VectorXd& mutable_batch_mean() { return batch_mean; }
    Eigen::VectorXd& mutable_batch_var() { return batch_var; }
    Eigen::VectorXd& mutable_inv_std() { return inv_std; }
};

} // namespace layers

#endif