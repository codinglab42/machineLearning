#ifndef SIMPLE_RNN_CACHE_H
#define SIMPLE_RNN_CACHE_H

#include "recurrent_cache.h"
#include <vector>
#include <Eigen/Dense>

namespace layers {

    class SimpleRNNCache : public RecurrentCache {
    public:
        struct RNNStep {
            Eigen::MatrixXd h;  // hidden state
            Eigen::MatrixXd x;  // input
            Eigen::MatrixXd pre_activation;  // pre-attivazione (prima di tanh)
            int timestep;
            
            RNNStep() : timestep(-1) {}
        };
        
        SimpleRNNCache();
        ~SimpleRNNCache() override = default;
        
        // Implementazione RecurrentCache
        void init(int sequence_length, int batch_size, int hidden_size) override;
        void add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                      const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd get_hidden_state(int timestep) const override;
        int get_sequence_length() const override;
        int get_hidden_size() const override;
        
        // Metodi specifici SimpleRNN
        void add_rnn_state(int timestep,
                          const Eigen::MatrixXd& hidden_state,
                          const Eigen::MatrixXd& input,
                          const Eigen::MatrixXd& pre_act);
        
        Eigen::MatrixXd get_input(int timestep) const;
        Eigen::MatrixXd get_pre_activation(int timestep) const;
        
        std::vector<Eigen::MatrixXd> get_all_hidden_states() const;
        std::vector<Eigen::MatrixXd> get_all_inputs() const;
        std::vector<Eigen::MatrixXd> get_all_pre_activations() const;
        
        const RNNStep& get_step(int timestep) const;
        const std::vector<RNNStep>& get_all_steps() const;
        
        int get_batch_size() const;
        
        // override BasicCache
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "SimpleRNNCache"; }
        
        // Debug
        void print_debug() const;

    private:
        std::vector<RNNStep> steps_;
        int seq_len_;
        int batch_size_;
        int hidden_size_;
    };

} // namespace layers

#endif
