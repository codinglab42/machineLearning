#ifndef RNN_CACHE_H
#define RNN_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class RNNCache : public LayerCache {
    public:
        RNNCache();
        ~RNNCache() override = default;
        
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "RNNCache"; }
        
        const Eigen::MatrixXd& get_input() const override { return input_cache; }
        const Eigen::MatrixXd& get_output() const override { return output_cache; }
        bool has_activation() const override { return true; }
        
        // Dati specifici RNN
        Eigen::MatrixXd input_cache;           // Input originale
        Eigen::MatrixXd output_cache;           // Output finale
        std::vector<Eigen::MatrixXd> hidden_states;  // Stati nascosti per ogni timestep
        std::vector<Eigen::MatrixXd> pre_activations; // Output pre-attivazione per ogni timestep
        
        int timesteps;
        int batch_size;
        int input_size;
        int hidden_size;
        bool training;
        
        Eigen::MatrixXd& mutable_input() { return input_cache; }
        Eigen::MatrixXd& mutable_output() { return output_cache; }
        std::vector<Eigen::MatrixXd>& mutable_hidden_states() { return hidden_states; }
        std::vector<Eigen::MatrixXd>& mutable_pre_activations() { return pre_activations; }

    private:
        bool validate_dimensions() const;
    };

} // namespace layers

#endif

