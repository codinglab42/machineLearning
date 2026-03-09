#ifndef LSTM_CACHE_H
#define LSTM_CACHE_H

#include "rnn_cache.h"
#include <Eigen/Dense>
#include <vector>

namespace layers {

    class LSTMCache : public RNNCache {
    public:
        LSTMCache();
        ~LSTMCache() override = default;
        
        std::string get_type() const override { return "LSTMCache"; }
        
        // Dati specifici LSTM
        std::vector<Eigen::MatrixXd> cell_states;      // Stato cella per ogni timestep
        std::vector<Eigen::MatrixXd> input_gates;      // Gate di input
        std::vector<Eigen::MatrixXd> forget_gates;     // Gate di forget
        std::vector<Eigen::MatrixXd> output_gates;     // Gate di output
        std::vector<Eigen::MatrixXd> cell_candidates;  // Candidati cella
        
        std::vector<Eigen::MatrixXd> z_i;  // Pre-attivazione input gate
        std::vector<Eigen::MatrixXd> z_f;  // Pre-attivazione forget gate
        std::vector<Eigen::MatrixXd> z_o;  // Pre-attivazione output gate
        std::vector<Eigen::MatrixXd> z_c;  // Pre-attivazione cell candidate
    };

} // namespace layers

#endif
