#ifndef RECURRENT_CACHE_H
#define RECURRENT_CACHE_H

#include "layer_cache.h"
#include <vector>
#include <Eigen/Dense>

namespace layers {

    class RecurrentCache : public LayerCache {
    public:
        ~RecurrentCache() override = default;
        
        // Metodi comuni a tutti i ricorrenti
        virtual void init(int sequence_length, int batch_size, int hidden_size) = 0;
        virtual void add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                              const Eigen::MatrixXd& input) = 0;
        virtual Eigen::MatrixXd get_hidden_state(int timestep) const = 0;
        virtual int get_sequence_length() const = 0;
        virtual int get_hidden_size() const = 0;
        
        // override di BasicCache
        void clear() override = 0;
        bool is_valid() const override = 0;
    };

} // namespace layers

#endif
