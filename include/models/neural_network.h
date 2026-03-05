#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <Eigen/Dense>
#include "estimator.h"
#include "components/optimizers/optimizer.h"
#include "components/optimizers/optimizer_factory.h"
#include "components/regularizers/regularizer.h"
#include "components/regularizers/regularizer_factory.h"
#include "components/layers/layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/recurrent_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include "components/layers/lstm_layer.h"
#include "components/layers/gru_layer.h"
#include "components/optimizers/optimizer.h"
//#include "components/regularizers/regularizer.h"
#include "exceptions/exception_macros.h"

namespace models {

    // Enum per i tipi di layer ricorrenti
    enum class RecurrentType {
        SIMPLE_RNN,
        LSTM,
        GRU
    };

    // Enum per i tipi di layer
    enum class LayerType {
        DENSE,
        CONV2D,
        MAX_POOLING,
        AVERAGE_POOLING,
        FLATTEN,
        DROPOUT,
        BATCH_NORM,
        SIMPLE_RNN,
        LSTM,
        GRU
    };

    class NeuralNetwork : public Estimator {
    public:
        // Costruttori
        NeuralNetwork();
        NeuralNetwork(const std::vector<int>& layer_sizes,
                     const std::string& activation = "relu",
                     const std::string& output_activation = "softmax",
                     OptimizerType optimizer_type = OptimizerType::SGD,
                     double learning_rate = 0.01,
                     RegularizerType regularizer_type = RegularizerType::NONE,
                     double regularizer_strength = 0.01);
        
        virtual ~NeuralNetwork() = default;

        // Set regularizer
        void set_regularizer(RegularizerType type, double strength = 0.01,
                            const std::unordered_map<std::string, double>& params = {});
        
        // Layer management generico
        void add_layer(std::unique_ptr<layers::Layer> layer);
        
        // Factory methods con enum
        void add_layer(LayerType type, const std::unordered_map<std::string, double>& params);
        
        // Metodi specifici per layer comuni
        void add_dense_layer(int units, const std::string& activation = "relu",
                            bool use_bias = true);
        
        void add_conv2d_layer(int filters, int kernel_size, int strides = 1,
                             const std::string& padding = "valid",
                             const std::string& activation = "relu");
        
        void add_pooling_layer(int pool_size = 2, int strides = 2,
                              const std::string& pool_type = "max");
        
        void add_flatten_layer();
        void add_dropout_layer(double rate);
        void add_batch_norm_layer();
        
        // Factory per layer ricorrenti
        void add_recurrent_layer(RecurrentType type, int units, 
                                bool return_sequences = false,
                                const std::string& activation = "tanh");
        
        // Training - OVERRIDE CORRETTI (senza parametri di default)
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override {
            fit(X, y, 100, 32, true);
        }
        
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) override {
            fit(X, y, 100, 32, true);
        }
        
        // Metodi con parametri aggiuntivi (NON override)
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                 int epochs, int batch_size, bool verbose);
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                 int epochs, int batch_size, bool verbose);
    
        // PREDICT - const corretto
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
        
        // predict_proba NON è virtuale in Estimator, quindi NON mettere override
        Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const;
    
        // SCORE
        double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const override;
    
        // Serialization
        void save(const std::string& filename) const;
        void load(const std::string& filename);
        
        // Getters
        int get_input_size() const { return n_features_; }
        int get_output_size() const { return n_classes_; }
        const std::vector<std::unique_ptr<layers::Layer>>& get_layers() const { return layers_; }
        bool is_fitted() const { return fitted_; }
        
        // Settings
        void set_optimizer(OptimizerType type, double learning_rate = 0.01);
        void set_loss_function(const std::string& loss) { loss_function_ = loss; }
        void set_verbose(bool verbose) { verbose_ = verbose; }
        
        // Reset
        void reset();

    protected:
        // Forward/backward passes - RIMOSSO const da forward_pass
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd& X, bool training = false);
        Eigen::VectorXd backward_pass(const Eigen::VectorXd& y_true, const Eigen::MatrixXd& y_pred);
        
        // Loss functions
        double compute_loss(const Eigen::VectorXd& y_true, const Eigen::MatrixXd& y_pred) const;
        double compute_loss(const Eigen::MatrixXd& y_true, const Eigen::MatrixXd& y_pred) const;
        Eigen::MatrixXd compute_loss_gradient(const Eigen::VectorXd& y_true, 
                                             const Eigen::MatrixXd& y_pred) const;
        
        // Initialization
        void initialize_layers(int input_size);

    private:
        std::vector<std::unique_ptr<layers::Layer>> layers_;
        std::unique_ptr<Optimizer> optimizer_;
        std::unique_ptr<Regularizer> regularizer_;
        
        // Cache per forward/backward
        std::vector<std::unique_ptr<layers::BasicCache>> forward_cache_;
            
        std::string loss_function_;
        double learning_rate_;
        bool verbose_;
        int n_features_;
        int n_classes_;
        bool fitted_;
        
        // Factory methods privati
        std::unique_ptr<layers::Layer> create_dense_layer(
            int units, const std::string& activation, bool use_bias);
        
        std::unique_ptr<layers::Layer> create_conv2d_layer(
            int filters, int kernel_size, int strides,
            const std::string& padding, const std::string& activation);
        
        std::unique_ptr<layers::Layer> create_pooling_layer(
            int pool_size, int strides, const std::string& pool_type);
        
        std::unique_ptr<layers::Layer> create_recurrent_layer(
            RecurrentType type, int units, bool return_sequences, 
            const std::string& activation);
    };

} // namespace models

#endif
