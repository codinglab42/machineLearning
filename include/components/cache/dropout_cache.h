#ifndef DROPOUT_CACHE_H
#define DROPOUT_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>

namespace layers {

    class DropoutCache : public LayerCache {
    public:
        DropoutCache();
        ~DropoutCache() override = default;
        
        // Implementazione interfaccia
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "DropoutCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return false; }
        
        // Dati specifici del dropout
        Eigen::MatrixXd input_cache;       // Input originale
        Eigen::MatrixXd output_cache;       // Output dopo dropout
        Eigen::MatrixXd mask;                // Maschera di dropout
        
        bool training;                       // Modalità training/inference
        
        // Accesso modificabile
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        Eigen::MatrixXd& mutable_mask() { return mask; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif
