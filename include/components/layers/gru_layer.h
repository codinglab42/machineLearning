#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/gru_cache.h"

namespace layers {

    class GRULayer : public RecurrentLayer {
    public:
        // Struttura per analisi gates
        struct GateValues {
            Eigen::MatrixXd update;     // update gate (z)
            Eigen::MatrixXd reset;      // reset gate (r)
            Eigen::MatrixXd new_gate;   // new gate (n)
            Eigen::MatrixXd hidden;     // hidden state
            Eigen::MatrixXd candidate;  // candidate hidden state
        };
        
        GRULayer(int hidden_size, int input_size);
        ~GRULayer() override = default;
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        std::string get_type() const override { return "GRULayer"; }
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
        Eigen::MatrixXd Wx_;  // [input_size, 3*hidden_size] - update, reset, new
        Eigen::MatrixXd Wh_;  // [hidden_size, 3*hidden_size] - update, reset, new
        Eigen::VectorXd b_;   // [3*hidden_size]
        
        // Accesso cache specifica
        GRUCache* get_specific_cache() {
            return static_cast<GRUCache*>(cache_.get());
        }
        const GRUCache* get_specific_cache() const {
            return static_cast<const GRUCache*>(cache_.get());
        }
        
        // Funzioni di attivazione
        inline Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) const {
            return (1.0 / (1.0 + (-x).array().exp()));
        }
        
        inline Eigen::MatrixXd tanh(const Eigen::MatrixXd& x) const {
            return x.array().tanh();
        }
    };

} // namespace layers

#endif
