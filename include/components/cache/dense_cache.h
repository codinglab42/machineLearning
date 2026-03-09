#ifndef DENSE_CACHE_H
#define DENSE_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

    class DenseCache : public LayerCache {
    public:
        DenseCache();
        ~DenseCache() override = default;
        
        // Implementazione interfaccia LayerCache
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "DenseCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return true; }
        
        // Dati specifici del Dense layer
        Eigen::MatrixXd input_cache;  // Input originale
        Eigen::MatrixXd z_cache;       // Output pre-attivazione
        Eigen::MatrixXd output_cache;   // Output post-attivazione
        
        // Accesso modificabile
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_z() { return z_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif
