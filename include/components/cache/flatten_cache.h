#ifndef FLATTEN_CACHE_H
#define FLATTEN_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class FlattenCache : public LayerCache {
    public:
        FlattenCache();
        ~FlattenCache() override = default;
        
        // Implementazione interfaccia
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "FlattenCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return false; }
        
        // Dati specifici del flatten
        Eigen::MatrixXd input_cache;       // Input originale
        Eigen::MatrixXd output_cache;       // Output flattened
        std::vector<int> original_shape;    // Shape originale [batch, features]
        
        // Accesso modificabile
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        std::vector<int>& mutable_shape() { return original_shape; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif
