#ifndef SIMPLE_RNN_LAYER_H
#define SIMPLE_RNN_LAYER_H

#include "recurrent_layer.h"
#include "components/cache/simple_rnn_cache.h"
#include <Eigen/Dense>

namespace layers {

    class SimpleRNNLayer : public RecurrentLayer {
    public:
        SimpleRNNLayer(int hidden_size, int input_size);
        ~SimpleRNNLayer() override = default;
        
        // Layer interface
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                               double learning_rate) override;
        
        std::string get_type() const override { return "SimpleRNNLayer"; }
        std::string get_config() const override;
        
        // Getter/Setter per pesi (specifici RNN)
        const Eigen::MatrixXd& get_kernel() const { return Wx_; }
        const Eigen::MatrixXd& get_recurrent_kernel() const { return Wh_; }
        const Eigen::VectorXd& get_bias() const { return b_; }
        
        void set_weights(const Eigen::MatrixXd& Wx, const Eigen::MatrixXd& Wh, 
                        const Eigen::VectorXd& b);
        
        // override metodi pesi della classe Layer
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;  // flatten di tutti i pesi
        void set_weights(const Eigen::MatrixXd& weights) override;

        int get_input_size() const  override{ return input_size_; }
        int get_output_size() const override { return hidden_size_; }
        void clear_cache() { cache_.clear(); }
        Eigen::MatrixXd get_weights() const { 
            // Flatten weights if needed
            return Wx_; 
        }
        Eigen::VectorXd get_biases() const { return b_; }
        void set_weights(const Eigen::MatrixXd& weights) { 
            // Set weights if needed
        }
        void set_biases(const Eigen::VectorXd& biases) { b_ = biases; }
        
    private:
        Eigen::MatrixXd Wx_;  // kernel (input -> hidden)
        Eigen::MatrixXd Wh_;  // recurrent kernel (hidden -> hidden)
        Eigen::VectorXd b_;   // bias
        
        // Accesso alla cache col tipo specifico
        SimpleRNNCache* get_specific_cache() {
            return static_cast<SimpleRNNCache*>(cache_.get());
        }
    };

} // namespace layers

#endif
