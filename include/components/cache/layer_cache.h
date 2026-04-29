// include/components/cache/layer_cache.h
#ifndef LAYER_CACHE_H
#define LAYER_CACHE_H

#include <Eigen/Dense>
#include <string>
#include <memory>

namespace layers {

    // Classe base per tutte le cache - SOLO dati temporanei!
    class LayerCache {
    public:
        virtual ~LayerCache() = default;
        
        virtual void clear() = 0;
        virtual bool is_valid() const = 0;
        virtual std::string get_type() const = 0;
        
        // Getters opzionali per dati comuni
        virtual const Eigen::MatrixXd& get_input() const { 
            static Eigen::MatrixXd empty;
            return empty; 
        }
        virtual const Eigen::MatrixXd& get_output() const { 
            static Eigen::MatrixXd empty;
            return empty; 
        }
    };

} // namespace layers

#endif