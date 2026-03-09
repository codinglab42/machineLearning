#ifndef DROPOUT_CACHE_H
#define DROPOUT_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

    class DropoutCache : public LayerCache {
    public:
        DropoutCache();
        ~DropoutCache() override = default;
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "DropoutCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return false; }
        
        Eigen::MatrixXd input_cache;
        Eigen::MatrixXd output_cache;
        Eigen::MatrixXd mask;
        
        bool training;
        
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        Eigen::MatrixXd& mutable_mask() { return mask; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif

