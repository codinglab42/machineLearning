// include/components/cache/rnn_cache.h
#ifndef RNN_CACHE_H
#define RNN_CACHE_H

#include "layer_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

class RNNCache : public LayerCache {
public:
    RNNCache() 
        : timesteps(0), batch_size(0), input_size(0), hidden_size(0), training(false) {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        hidden_states.clear();
        pre_activations.clear();
    }
    
    ~RNNCache() override = default;
    
    void clear() override {
        input_cache.resize(0, 0);
        output_cache.resize(0, 0);
        hidden_states.clear();
        pre_activations.clear();
        
        timesteps = 0;
        batch_size = 0;
        input_size = 0;
        hidden_size = 0;
        training = false;
    }
    
    bool is_valid() const override {
        return input_cache.size() > 0 && output_cache.size() > 0;
    }
    
    std::string get_type() const override { return "RNNCache"; }
    
    const Eigen::MatrixXd& get_input() const override { return input_cache; }
    const Eigen::MatrixXd& get_output() const override { return output_cache; }
    
    // Dati temporanei RNN
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
    std::vector<Eigen::MatrixXd> hidden_states;
    std::vector<Eigen::MatrixXd> pre_activations;
    
    int timesteps;
    int batch_size;
    int input_size;
    int hidden_size;
    bool training;
    
    Eigen::MatrixXd& mutable_input() { return input_cache; }
    Eigen::MatrixXd& mutable_output() { return output_cache; }
    std::vector<Eigen::MatrixXd>& mutable_hidden_states() { return hidden_states; }
    std::vector<Eigen::MatrixXd>& mutable_pre_activations() { return pre_activations; }
};

} // namespace layers

#endif