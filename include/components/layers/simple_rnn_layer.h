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
        
        // Layer interface - IMPLEMENTA TUTTI I METODI VIRTUALI PURI
        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
        Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient,
                                double learning_rate) override;
        
        std::string get_type() const override { return "SimpleRNNLayer"; }
        std::string get_config() const override;
        
        // Gestione pesi - UN VERSIONE SOLA, non overload
        bool has_weights() const override { return true; }
        Eigen::MatrixXd get_weights() const override;  // flatten di tutti i pesi
        void set_weights(const Eigen::MatrixXd& weights) override;
        
        // Metodi Layer obbligatori
        int get_input_size() const override { return input_size_; }
        int get_output_size() const override { return hidden_size_; }
        void clear_cache() override { 
            if (cache_) cache_->clear(); 
        }
        
        Eigen::VectorXd get_biases() const override { return b_; }
        void set_biases(const Eigen::VectorXd& biases) override { b_ = biases; }
        
        // Getter/Setter specifici RNN
        const Eigen::MatrixXd& get_kernel() const { return Wx_; }
        const Eigen::MatrixXd& get_recurrent_kernel() const { return Wh_; }
        const Eigen::VectorXd& get_bias() const { return b_; }
        
        void set_specific_weights(const Eigen::MatrixXd& Wx, const Eigen::MatrixXd& Wh, 
                                  const Eigen::VectorXd& b);
        
        // Metodo RecurrentLayer
        void set_input_shape(int input_size) override { input_size_ = input_size; }
        
    private:
        Eigen::MatrixXd Wx_;  // kernel (input -> hidden)
        Eigen::MatrixXd Wh_;  // recurrent kernel (hidden -> hidden)
        Eigen::VectorXd b_;   // bias
        int input_size_;
        int hidden_size_;
        
        // Accesso alla cache col tipo specifico
        SimpleRNNCache* get_specific_cache() {
            return static_cast<SimpleRNNCache*>(cache_.get());
        }
    };

} // namespace layers

#endif