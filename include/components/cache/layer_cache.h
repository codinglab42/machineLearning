#ifndef LAYER_CACHE_H
#define LAYER_CACHE_H

#include <Eigen/Dense>
#include <string>

namespace layers {

    class LayerCache {
    public:
        virtual ~LayerCache() = default;
        
        // Interfaccia pubblica
        virtual void clear() = 0;
        virtual bool is_valid() const = 0;
        virtual std::string get_type() const = 0;
        
        // Getters comuni (possono essere override nelle derivate)
        virtual const Eigen::MatrixXd& get_input() const = 0;
        virtual const Eigen::MatrixXd& get_output() const = 0;
        virtual bool has_activation() const = 0;
    };

} // namespace layers

#endif

