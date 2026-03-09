#ifndef BATCH_NORM_CACHE_H
#define BATCH_NORM_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

    class BatchNormCache : public LayerCache {
    public:
        BatchNormCache();
        ~BatchNormCache() override = default;
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "BatchNormCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return false; }
        
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

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif

