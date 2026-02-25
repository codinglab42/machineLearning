#ifndef WEIGHTED_CACHE_H
#define WEIGHTED_CACHE_H

#include "basic_cache.h"

namespace layers {

    class WeightedCache : public BasicCache {
    public:
        WeightedCache();
        ~WeightedCache() override = default;
        
        // Override metodi virtuali
        void clear() override;
        std::string get_type() const override { return "WeightedCache"; }
        bool is_valid() const override;
        
        // Getter/Setter per z (pre-attivazione)
        const Eigen::MatrixXd& get_z() const { return z_; }
        void set_z(const Eigen::MatrixXd& z) { z_ = z; }
        
        // Accesso modificabile (per i layer che devono popolare la cache)
        Eigen::MatrixXd& mutable_z() { return z_; }

    private:
        Eigen::MatrixXd z_;  // pre-attivazione (prima della funzione di attivazione)
    };

} // namespace layers

#endif
