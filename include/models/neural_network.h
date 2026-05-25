#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <unordered_map>
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
#include "components/cache/layer_cache.h"
#include "components/loss/loss_factory.h"
#include "components/loss/loss.h"
#include "exceptions/exception_macros.h"

namespace models {

    // Enum per i tipi di layer ricorrenti
    enum class RecurrentType {
        SIMPLE_RNN,
        LSTM,
        GRU
    };

    using LayerType = layers::LayerType;

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
        
        virtual ~NeuralNetwork() override = default;

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
        
        // Training - NON usare override per metodi con parametri diversi
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y);
        
        // Metodi con parametri aggiuntivi
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                 int epochs, int batch_size, bool verbose);
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                 int epochs, int batch_size, bool verbose);

        void build(int n_features, int n_classes);
    
        // PREDICT - const corretto
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
        Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const;
    
        // SCORE
        double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const override;
    
        
        // Training parameters
        void set_batch_size(int batch_size) { batch_size_ = batch_size; }
        void set_epochs(int epochs) { epochs_ = epochs; }
        void set_validation_split(double split) { 
            ML_CHECK_PARAM(split >= 0.0 && split < 1.0, "validation_split", 
                        "must be in [0, 1)", "NeuralNetwork");
            validation_split_ = split; 
        }

        // History getters
        const std::vector<double>& get_loss_history() const { return loss_history_; }
        const std::vector<double>& get_val_loss_history() const { return val_loss_history_; }
        const std::vector<double>& get_accuracy_history() const { return accuracy_history_; }
        std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> get_training_history() const;


        // Network info
        void summary() const;
        int get_num_layers() const { return layers_.size(); }
        int get_num_parameters() const;



        // Serialization
        void save(const std::string& filename) const override;
        void load(const std::string& filename) override;
        
        // Metodi ereditati da SerializableModel
        std::string to_string() const override;
        void serialize_binary(std::ostream& out) const override;
        void deserialize_binary(std::istream& in) override;
        std::string get_model_type() const override;
        
        // Getters
        int get_input_size() const { return n_features_; }
        int get_output_size() const { return n_classes_; }
        const std::vector<std::unique_ptr<layers::Layer>>& get_layers() const { return layers_; }
        bool is_fitted() const { return fitted_; }
        
        // Settings
        void set_optimizer(OptimizerType type, double learning_rate = 0.01);
        void set_loss_function(const std::string& loss);
        void set_verbose(bool verbose) { verbose_ = verbose; }
        
        // Reset
        void reset();
        void reset_history() ;

    protected:

        void determine_problem_type(const Eigen::VectorXd& y);
        void clip_gradient(Eigen::MatrixXd& gradient, double max_norm = 1.0);

        // Forward/backward passes - forward_pass DICHIARATA const
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd& X, bool training = false) const;
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

        std::vector<double> loss_history_;
        std::vector<double> val_loss_history_;
        std::vector<double> accuracy_history_;
        int batch_size_;
        int epochs_;
        double validation_split_;

        // Cache per forward/backward (opzionale)
        mutable std::vector<std::shared_ptr<layers::LayerCache>> forward_cache_;
            
        std::unique_ptr<loss::Loss> loss_function_;  // Usa la classe astratta
        std::string loss_function_name_;              // Mantieni il nome per serializzazione
        double learning_rate_;
        bool verbose_;
        int n_features_;
        int n_classes_;
        bool fitted_;

        // random generator for shuffling
        mutable std::mt19937 rng_;
        
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
