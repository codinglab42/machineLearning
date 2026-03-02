#ifndef LSTM_CACHE_H
#define LSTM_CACHE_H

#include "recurrent_cache.h"
#include <vector>
#include <Eigen/Dense>

namespace layers {

    class LSTMCache : public RecurrentCache {
    public:
        struct LSTMStep {
            Eigen::MatrixXd h;  // hidden state
            Eigen::MatrixXd c;  // cell state
            Eigen::MatrixXd x;  // input
            Eigen::MatrixXd i;  // input gate
            Eigen::MatrixXd f;  // forget gate
            Eigen::MatrixXd o;  // output gate
            Eigen::MatrixXd g;  // candidate gate
            int timestep;
            
            LSTMStep() : timestep(-1) {}
        };
        
        LSTMCache();
        ~LSTMCache() override = default;
        
        // Implementazione RecurrentCache
        void init(int sequence_length, int batch_size, int hidden_size) override;
        void add_state(int timestep, const Eigen::MatrixXd& hidden_state, 
                      const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd get_hidden_state(int timestep) const override;
        int get_sequence_length() const override;
        int get_hidden_size() const override;
        
        // Metodi specifici LSTM
        void add_lstm_state(int timestep,
                           const Eigen::MatrixXd& hidden_state,
                           const Eigen::MatrixXd& cell_state,
                           const Eigen::MatrixXd& input,
                           const Eigen::MatrixXd& input_gate,
                           const Eigen::MatrixXd& forget_gate,
                           const Eigen::MatrixXd& output_gate,
                           const Eigen::MatrixXd& candidate_gate);
        
        Eigen::MatrixXd get_cell_state(int timestep) const;
        Eigen::MatrixXd get_input(int timestep) const;
        Eigen::MatrixXd get_input_gate(int timestep) const;
        Eigen::MatrixXd get_forget_gate(int timestep) const;
        Eigen::MatrixXd get_output_gate(int timestep) const;
        Eigen::MatrixXd get_candidate_gate(int timestep) const;
        
        std::vector<Eigen::MatrixXd> get_all_hidden_states() const;
        std::vector<Eigen::MatrixXd> get_all_cell_states() const;
        
        const LSTMStep& get_step(int timestep) const;
        const std::vector<LSTMStep>& get_all_steps() const;
        
        int get_batch_size() const;
        
        // override BasicCache
        void clear() override;
        bool is_valid() const override;
        std::string get_type() const override { return "LSTMCache"; }
        
        // Debug
        void print_debug() const;

    private:
        std::vector<LSTMStep> steps_;
        int seq_len_;
        int batch_size_;
        int hidden_size_;
    };

} // namespace layers

#endif