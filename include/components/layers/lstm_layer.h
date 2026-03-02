#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/lstm_cache.h"

namespace layers {

    class LSTMLayer : public RecurrentLayer {
    public:
        // Struttura per analisi gates
        struct GateValues {
            Eigen::MatrixXd input;      // input gate
            Eigen::MatrixXd forget;     // forget gate
            Eigen::MatrixXd output;     // output gate
            Eigen::MatrixXd cell;       // candidate gate
            Eigen::MatrixXd hidden;     // hidden state
            Eigen::MatrixXd cell_state; // cell state
        };
        
        LSTMLayer(int hidden_size, int input_size);
        ~LSTMLayer() override = default;
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        std::string get_type() const override { return "LSTMLayer"; }
        std::string get_config() const override;
        int get_parameter_count() const override;
        
        // Gestione pesi
        void set_weights(const Eigen::MatrixXd& Wx, 
                        const Eigen::MatrixXd& Wh, 
                        const Eigen::VectorXd& b);
        
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;
        void set_weights(const Eigen::MatrixXd& weights) override;
        
        // Serializzazione
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        // Utility
        void initialize_weights();
        void reset_states();
        
        // Analisi gates
        GateValues get_gates(int timestep) const;
        
        // Getter pesi
        const Eigen::MatrixXd& get_kernel() const { return Wx_; }
        const Eigen::MatrixXd& get_recurrent_kernel() const { return Wh_; }
        const Eigen::VectorXd& get_bias() const { return b_; }
        
    private:
        Eigen::MatrixXd Wx_;  // [input_size, 4*hidden_size]
        Eigen::MatrixXd Wh_;  // [hidden_size, 4*hidden_size]
        Eigen::VectorXd b_;   // [4*hidden_size]
        
        // Accesso cache specifica
        LSTMCache* get_specific_cache() {
            return static_cast<LSTMCache*>(cache_.get());
        }
        const LSTMCache* get_specific_cache() const {
            return static_cast<const LSTMCache*>(cache_.get());
        }
    };

} // namespace layers

#endif
