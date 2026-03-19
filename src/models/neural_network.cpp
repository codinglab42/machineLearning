#include "models/neural_network.h"
#include "exceptions/dimension_exception.h"
#include "exceptions/fitting_exception.h" 
#include "exceptions/io_exception.h"
#include "exceptions/validation_exception.h"
#include "exceptions/exception_macros.h"
#include "utils/serializable.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace models {

    //===========================================================================
    // COSTRUTTORI
    //===========================================================================
    
    // Costruttore default
    NeuralNetwork::NeuralNetwork() 
        : loss_function_("mse"),
          learning_rate_(0.01),
          verbose_(false),
          n_features_(0),
          n_classes_(0),
          fitted_(false),
          batch_size_(32),
          epochs_(100),
          validation_split_(0.0) {
        optimizer_ = OptimizerFactory::create(OptimizerType::SGD, learning_rate_);
    }

    // Costruttore con parametri
    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::string& activation,
                                 const std::string& output_activation,
                                 OptimizerType optimizer_type,
                                 double learning_rate,
                                 RegularizerType regularizer_type,
                                 double regularizer_strength)
        : loss_function_("categorical_crossentropy"),
          learning_rate_(learning_rate),
          verbose_(false),
          n_features_(layer_sizes.front()),
          n_classes_(layer_sizes.back()),
          fitted_(false) {
        
        ML_CHECK_PARAM(layer_sizes.size() >= 2, "layer_sizes", 
                      "must have at least input and output layer", "NeuralNetwork");
        
        optimizer_ = OptimizerFactory::create(optimizer_type, learning_rate);
        regularizer_ = RegularizerFactory::create(regularizer_type, regularizer_strength);
        
        // Crea layer nascosti
        for (size_t i = 1; i < layer_sizes.size() - 1; ++i) {
            add_dense_layer(layer_sizes[i], activation);
        }
        
        // Crea layer di output
        add_dense_layer(layer_sizes.back(), output_activation);
    }

    //===========================================================================
    // LAYER MANAGEMENT
    //===========================================================================
    
    void NeuralNetwork::add_layer(std::unique_ptr<layers::Layer> layer) {
        if (!layers_.empty()) {
            int prev_output = layers_.back()->get_output_size();
            
            // Imposta input shape per il nuovo layer
            layer->set_input_shape(prev_output);
        }
        
        layers_.push_back(std::move(layer));
        forward_cache_.resize(layers_.size());
    }

    void NeuralNetwork::add_layer(LayerType type, 
                                   const std::unordered_map<std::string, double>& params) {
        std::unique_ptr<layers::Layer> layer;
        
        switch (type) {
            case LayerType::DENSE: {
                int units = static_cast<int>(params.at("units"));
                std::string activation = params.count("activation") ? 
                    std::to_string(static_cast<int>(params.at("activation"))) : "relu";
                bool use_bias = params.count("use_bias") ? 
                    static_cast<bool>(params.at("use_bias")) : true;
                layer = create_dense_layer(units, activation, use_bias);
                break;
            }
            case LayerType::CONV2D: {
                int filters = static_cast<int>(params.at("filters"));
                int kernel_size = static_cast<int>(params.at("kernel_size"));
                int strides = params.count("strides") ? 
                    static_cast<int>(params.at("strides")) : 1;
                std::string padding = params.count("padding") ? 
                    std::to_string(static_cast<int>(params.at("padding"))) : "valid";
                std::string activation = params.count("activation") ? 
                    std::to_string(static_cast<int>(params.at("activation"))) : "relu";
                layer = create_conv2d_layer(filters, kernel_size, strides, padding, activation);
                break;
            }
            case LayerType::MAX_POOLING:
            case LayerType::AVERAGE_POOLING: {
                int pool_size = params.count("pool_size") ? 
                    static_cast<int>(params.at("pool_size")) : 2;
                int strides = params.count("strides") ? 
                    static_cast<int>(params.at("strides")) : 2;
                std::string pool_type = (type == LayerType::MAX_POOLING) ? "max" : "average";
                layer = create_pooling_layer(pool_size, strides, pool_type);
                break;
            }
            case LayerType::FLATTEN:
                layer = std::make_unique<layers::FlattenLayer>();
                break;
            case LayerType::DROPOUT: {
                double rate = params.at("rate");
                layer = std::make_unique<layers::DropoutLayer>(rate);
                break;
            }
            case LayerType::BATCH_NORM:
                layer = std::make_unique<layers::BatchNormLayer>();
                break;
            case LayerType::SIMPLE_RNN:
            case LayerType::LSTM:
            case LayerType::GRU: {
                int units = static_cast<int>(params.at("units"));
                bool return_sequences = params.count("return_sequences") ? 
                    static_cast<bool>(params.at("return_sequences")) : false;
                std::string activation = params.count("activation") ? 
                    std::to_string(static_cast<int>(params.at("activation"))) : "tanh";
                
                RecurrentType rnn_type;
                switch (type) {
                    case LayerType::SIMPLE_RNN: rnn_type = RecurrentType::SIMPLE_RNN; break;
                    case LayerType::LSTM: rnn_type = RecurrentType::LSTM; break;
                    case LayerType::GRU: rnn_type = RecurrentType::GRU; break;
                    default: rnn_type = RecurrentType::SIMPLE_RNN;
                }
                
                layer = create_recurrent_layer(rnn_type, units, return_sequences, activation);
                break;
            }
        }
        
        if (layer) {
            add_layer(std::move(layer));
        }
    }

    //===========================================================================
    // LAYER SPECIFICI
    //===========================================================================
    
    void NeuralNetwork::add_dense_layer(int units, const std::string& activation, bool use_bias) {
        auto layer = create_dense_layer(units, activation, use_bias);
        add_layer(std::move(layer));
    }

    void NeuralNetwork::add_conv2d_layer(int filters, int kernel_size, int strides,
                                         const std::string& padding,
                                         const std::string& activation) {
        auto layer = create_conv2d_layer(filters, kernel_size, strides, padding, activation);
        add_layer(std::move(layer));
    }

    void NeuralNetwork::add_pooling_layer(int pool_size, int strides, const std::string& pool_type) {
        auto layer = create_pooling_layer(pool_size, strides, pool_type);
        add_layer(std::move(layer));
    }

    void NeuralNetwork::add_flatten_layer() {
        add_layer(std::make_unique<layers::FlattenLayer>());
    }

    void NeuralNetwork::add_dropout_layer(double rate) {
        ML_CHECK_PARAM(rate >= 0.0 && rate < 1.0, "rate", 
                      "must be in [0, 1)", "NeuralNetwork");
        add_layer(std::make_unique<layers::DropoutLayer>(rate));
    }

    void NeuralNetwork::add_batch_norm_layer() {
        add_layer(std::make_unique<layers::BatchNormLayer>());
    }

    //===========================================================================
    // LAYER RICORRENTI
    //===========================================================================
    
    void NeuralNetwork::add_recurrent_layer(RecurrentType type, int units,
                                            bool return_sequences,
                                            const std::string& activation) {
        auto layer = create_recurrent_layer(type, units, return_sequences, activation);
        add_layer(std::move(layer));
    }

    //===========================================================================
    // FACTORY METHODS PRIVATI
    //===========================================================================
    
    std::unique_ptr<layers::Layer> NeuralNetwork::create_dense_layer(
        int units, const std::string& activation, bool use_bias) {
        
        int input_size = layers_.empty() ? n_features_ : layers_.back()->get_output_size();
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "NeuralNetwork");
        
        auto layer = std::make_unique<layers::DenseLayer>(units, activation, use_bias);
        layer->set_input_shape(input_size);
        return layer;
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_conv2d_layer(
        int filters, int kernel_size, int strides,
        const std::string& padding, const std::string& activation) {
        
        auto layer = std::make_unique<layers::Conv2DLayer>(
            filters, kernel_size, strides, padding, activation);
        return layer;
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_pooling_layer(
        int pool_size, int strides, const std::string& pool_type) {
        
        layers::PoolingLayer::PoolType type = (pool_type == "max") ? 
            layers::PoolingLayer::MAX : layers::PoolingLayer::AVG;
        
        int channels = 1; // Da calcolare in base all'input
        return std::make_unique<layers::PoolingLayer>(pool_size, strides, type, channels);
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_recurrent_layer(
        RecurrentType type, int units, bool return_sequences,
        const std::string& activation) {
        
        int input_size = layers_.empty() ? n_features_ : layers_.back()->get_output_size();
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "NeuralNetwork");
        
        std::unique_ptr<layers::Layer> layer;
        
        switch (type) {
            case RecurrentType::SIMPLE_RNN:
                layer = std::make_unique<layers::SimpleRNNLayer>(units, input_size);
                break;
            case RecurrentType::LSTM:
                layer = std::make_unique<layers::LSTMLayer>(units, input_size);
                break;
            case RecurrentType::GRU:
                layer = std::make_unique<layers::GRULayer>(units, input_size);
                break;
        }
        
        return layer;
    }

    //===========================================================================
    // TRAINING - OVERRIDE METHODS
    //===========================================================================
    
    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit(X, y, 100, 32, true);
    }

    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
        fit(X, y, 100, 32, true);
    }

    //===========================================================================
    // TRAINING - METODI CON PARAMETRI
    //===========================================================================
    
    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                            int epochs, int batch_size, bool verbose) {
        
        ML_CHECK_NOT_EMPTY(X, "X", "NeuralNetwork");
        ML_CHECK_NOT_EMPTY(y, "y", "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");
        ML_CHECK_PARAM(epochs > 0, "epochs", "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", "NeuralNetwork");
        
        verbose_ = verbose;
        n_features_ = X.cols();
        n_classes_ = 1;
        
        if (layers_.empty()) {
            initialize_layers(n_features_);
        }
        
        int n_samples = X.rows();
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        forward_cache_.resize(layers_.size());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            
            std::shuffle(indices.begin(), indices.end(), 
                        std::mt19937(static_cast<unsigned int>(epoch)));
            
            for (int i = 0; i < n_samples; i += batch_size) {
                int end = std::min(i + batch_size, n_samples);
                int current_batch_size = end - i;
                
                Eigen::MatrixXd X_batch(current_batch_size, X.cols());
                Eigen::VectorXd y_batch(current_batch_size);
                
                for (int j = i; j < end; ++j) {
                    X_batch.row(j - i) = X.row(indices[j]);
                    y_batch(j - i) = y(indices[j]);
                }
                
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);
                
                double batch_loss = 0.0;
                for (int k = 0; k < current_batch_size; ++k) {
                    // Crea un vettore temporaneo per il singolo valore
                    Eigen::VectorXd y_true_single(1);
                    y_true_single(0) = y_batch(k);
                    batch_loss += compute_loss(y_true_single, y_pred.row(k));
                }
                batch_loss /= current_batch_size;
                epoch_loss += batch_loss * current_batch_size;
                
                backward_pass(y_batch, y_pred);
            }
            
            epoch_loss /= n_samples;
        
            // REGISTRA LA HISTORY
            loss_history_.push_back(epoch_loss);
            
            if (validation_split_ > 0.0) {
                // Calcola validation loss se necessario
                // val_loss_history_.push_back(val_loss);
            }
            
            if (verbose_ && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss;
                if (!val_loss_history_.empty() && epoch < val_loss_history_.size()) {
                    std::cout << ", Val Loss: " << val_loss_history_[epoch];
                }
                std::cout << std::endl;
            }
        }
        
        fitted_ = true;
    }

    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                            int epochs, int batch_size, bool verbose) {
        
        ML_CHECK_NOT_EMPTY(X, "X", "NeuralNetwork");
        ML_CHECK_NOT_EMPTY(y, "y", "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.rows(), "NeuralNetwork");
        ML_CHECK_PARAM(epochs > 0, "epochs", "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", "NeuralNetwork");
        
        verbose_ = verbose;
        n_features_ = X.cols();
        n_classes_ = y.cols();
        
        if (layers_.empty()) {
            initialize_layers(n_features_);
        }
        
        int n_samples = X.rows();
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        forward_cache_.resize(layers_.size());
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            
            std::shuffle(indices.begin(), indices.end(), 
                        std::mt19937(static_cast<unsigned int>(epoch)));
            
            for (int i = 0; i < n_samples; i += batch_size) {
                int end = std::min(i + batch_size, n_samples);
                int current_batch_size = end - i;
                
                Eigen::MatrixXd X_batch(current_batch_size, X.cols());
                Eigen::MatrixXd y_batch(current_batch_size, y.cols());
                
                for (int j = i; j < end; ++j) {
                    X_batch.row(j - i) = X.row(indices[j]);
                    y_batch.row(j - i) = y.row(indices[j]);
                }
                
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);
                
                double batch_loss = 0.0;
                for (int k = 0; k < current_batch_size; ++k) {
                    // Converti esplicitamente le righe in Eigen::VectorXd per disambiguare
                    Eigen::VectorXd y_true_row = y_batch.row(k);
                    Eigen::VectorXd y_pred_row = y_pred.row(k);
                    batch_loss += compute_loss(y_true_row, y_pred_row);
                }
                batch_loss /= current_batch_size;
                epoch_loss += batch_loss * current_batch_size;
                
                Eigen::MatrixXd gradient = (y_pred - y_batch) / current_batch_size;
                for (int j = static_cast<int>(layers_.size()) - 1; j >= 0; --j) {
                    gradient = layers_[j]->backward(gradient, learning_rate_);
                }
            }
            
            epoch_loss /= n_samples;
        
            // REGISTRA LA HISTORY
            loss_history_.push_back(epoch_loss);
            
            // Calcola accuracy se è un problema di classificazione
            if (n_classes_ > 1) {
                int correct = 0;
                Eigen::MatrixXd predictions = forward_pass(X, false);
                for (int i = 0; i < X.rows(); ++i) {
                    Eigen::Index pred_idx, true_idx;
                    predictions.row(i).maxCoeff(&pred_idx);
                    y.row(i).maxCoeff(&true_idx);
                    if (pred_idx == true_idx) correct++;
                }
                double accuracy = static_cast<double>(correct) / X.rows();
                accuracy_history_.push_back(accuracy);
            }
            
            if (validation_split_ > 0.0) {
                // Calcola validation loss se necessario
                // val_loss_history_.push_back(val_loss);
            }
            
            if (verbose_ && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss;
                if (!val_loss_history_.empty() && epoch < val_loss_history_.size()) {
                    std::cout << ", Val Loss: " << val_loss_history_[epoch];
                }
                if (!accuracy_history_.empty() && epoch < accuracy_history_.size()) {
                    std::cout << ", Acc: " << accuracy_history_[epoch];
                }
                std::cout << std::endl;
            }
        }
        
        fitted_ = true;
    }

    //===========================================================================
    // FORWARD/BACKWARD PASS
    //===========================================================================
    
    Eigen::MatrixXd NeuralNetwork::forward_pass(const Eigen::MatrixXd& X, bool training) const {
        Eigen::MatrixXd activation = X;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            activation = layers_[i]->forward(activation, training);
            
            // Salva riferimento alla cache (opzionale)
            if (i < forward_cache_.size()) {
                forward_cache_[i] = layers_[i]->get_cache();
            }
        }
        
        return activation;
    }

    Eigen::VectorXd NeuralNetwork::backward_pass(const Eigen::VectorXd& y_true, 
                                           const Eigen::MatrixXd& y_pred) {
        // Calcola gradiente iniziale
        Eigen::MatrixXd gradient;
        
        if (loss_function_ == "mse") {
            gradient = 2.0 * (y_pred - y_true) / y_true.size();  // Togli .transpose()
        } else if (loss_function_ == "binary_crossentropy") {
            gradient = (y_pred - y_true).array() / 
                    (y_pred.array() * (1.0 - y_pred.array()) + 1e-7);  // Togli .transpose()
        } else {
            gradient = y_pred - y_true;  // Togli .transpose()
        }
        
        // Backpropagation attraverso i layer
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            gradient = layers_[i]->backward(gradient, learning_rate_);
        }
        
        return gradient;
    }

    //===========================================================================
    // LOSS FUNCTIONS
    //===========================================================================
    
    double NeuralNetwork::compute_loss(const Eigen::VectorXd& y_true, 
                                        const Eigen::MatrixXd& y_pred) const {
        double data_loss = 0.0;

        if (loss_function_ == "mse") {
            data_loss = (y_pred - y_true.transpose()).array().square().mean();
        } else if (loss_function_ == "binary_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            // Aggiungi epsilon per evitare log(0)
            pred = pred.max(1e-7).min(1.0 - 1e-7);
            data_loss = -((true_val * pred.log() + (1 - true_val) * (1 - pred).log())).mean();
        } else if (loss_function_ == "categorical_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            // Aggiungi epsilon per evitare log(0)
            pred = pred.max(1e-7);
            data_loss = -(true_val * pred.log()).sum() / y_true.size();
        }

        // Aggiungi regolarizzazione
        double reg_loss = 0.0;
        if (regularizer_) {
            for (const auto& layer : layers_) {
                if (layer->has_weights()) {
                    reg_loss += regularizer_->compute_loss(layer->get_weights());
                }
            }
        }
        
        return data_loss + reg_loss;
    }

    double NeuralNetwork::compute_loss(const Eigen::MatrixXd& y_true, 
                                        const Eigen::MatrixXd& y_pred) const {
        double data_loss = 0.0;

        if (loss_function_ == "mse") {
            data_loss = (y_pred - y_true).array().square().mean();  // Togli .transpose()
        } else if (loss_function_ == "binary_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            pred = pred.max(1e-7).min(1.0 - 1e-7);
            data_loss = -((true_val * pred.log() + (1 - true_val) * (1 - pred).log())).mean();
        } else if (loss_function_ == "categorical_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            pred = pred.max(1e-7);
            data_loss = -(true_val * pred.log()).sum() / y_true.size();
        }

        // Aggiungi regolarizzazione
        double reg_loss = 0.0;
        if (regularizer_) {
            for (const auto& layer : layers_) {
                if (layer->has_weights()) {
                    reg_loss += regularizer_->compute_loss(layer->get_weights());
                }
            }
        }
        
        return data_loss + reg_loss;
    }

    Eigen::MatrixXd NeuralNetwork::compute_loss_gradient(const Eigen::VectorXd& y_true, 
                                                           const Eigen::MatrixXd& y_pred) const {
        if (loss_function_ == "mse") {
            return 2.0 * (y_pred - y_true.transpose()) / y_true.size();
        } else if (loss_function_ == "binary_crossentropy") {
            return (y_pred - y_true.transpose()).array() / 
                   (y_pred.array() * (1.0 - y_pred.array()) + 1e-7);
        } else {
            return y_pred - y_true.transpose();
        }
    }

    //===========================================================================
    // PREDIZIONE
    //===========================================================================
    
    Eigen::VectorXd NeuralNetwork::predict(const Eigen::MatrixXd& X) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        
        Eigen::MatrixXd output = forward_pass(X, false);
        
        if (n_classes_ == 1) {
            return output.col(0);
        } else {
            Eigen::VectorXd predictions(X.rows());
            for (int i = 0; i < X.rows(); ++i) {
                Eigen::Index max_idx;
                output.row(i).maxCoeff(&max_idx);
                predictions(i) = static_cast<double>(max_idx);
            }
            return predictions;
        }
    }

    Eigen::MatrixXd NeuralNetwork::predict_proba(const Eigen::MatrixXd& X) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        
        return forward_pass(X, false);
    }

    //===========================================================================
    // SCORE
    //===========================================================================
    
    double NeuralNetwork::score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");
        
        Eigen::VectorXd y_pred = predict(X);
        
        if (n_classes_ == 1) {
            // R^2 score per regressione
            double y_mean = y.mean();
            double ss_tot = (y.array() - y_mean).square().sum();
            double ss_res = (y.array() - y_pred.array()).square().sum();
            return 1.0 - (ss_res / (ss_tot + 1e-7));
        } else {
            // Accuratezza per classificazione
            int correct = 0;
            for (int i = 0; i < y.size(); ++i) {
                if (std::abs(y_pred(i) - y(i)) < 1e-6) correct++;
            }
            return static_cast<double>(correct) / y.size();
        }
    }

    //===========================================================================
    // SERIALIZZAZIONE
    //===========================================================================
    
    void NeuralNetwork::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            ML_THROW_IO_ERROR(filename, "save", "NeuralNetwork");
        }
        serialize_binary(file);
    }

    void NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            ML_THROW_IO_ERROR(filename, "load", "NeuralNetwork");
        }
        deserialize_binary(file);
    }

    std::string NeuralNetwork::to_string() const {
        std::string result = "NeuralNetwork(\n";
        for (size_t i = 0; i < layers_.size(); ++i) {
            result += "  Layer " + std::to_string(i) + ": " + 
                      layers_[i]->get_config() + "\n";
        }
        result += ")";
        return result;
    }

    void NeuralNetwork::serialize_binary(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&n_classes_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&fitted_), sizeof(bool));
        
        size_t num_layers = layers_.size();
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
        
        for (const auto& layer : layers_) {
            std::string type = layer->get_type();
            size_t type_len = type.size();
            out.write(reinterpret_cast<const char*>(&type_len), sizeof(size_t));
            out.write(type.c_str(), type_len);
            layer->serialize(out);
        }
    }

    void NeuralNetwork::deserialize_binary(std::istream& in) {
    in.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_classes_), sizeof(int));
    in.read(reinterpret_cast<char*>(&fitted_), sizeof(bool));
    
    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    layers_.clear();
    for (size_t i = 0; i < num_layers; ++i) {
        size_t type_len;
        in.read(reinterpret_cast<char*>(&type_len), sizeof(size_t));
        
        // Leggi la stringa del tipo
        std::vector<char> type_buf(type_len + 1, '\0');
        in.read(type_buf.data(), type_len);
        std::string type(type_buf.data());
        
        std::cout << "Deserializing layer type: " << type << std::endl;  // Debug
        
        std::unique_ptr<layers::Layer> layer;
        
        if (type == "DenseLayer") {
            layer = std::make_unique<layers::DenseLayer>(1, "relu", true);
        } else if (type == "Conv2DLayer") {
            layer = std::make_unique<layers::Conv2DLayer>(1, 3);
        } else if (type == "FlattenLayer") {
            layer = std::make_unique<layers::FlattenLayer>();
        } else if (type == "DropoutLayer") {
            layer = std::make_unique<layers::DropoutLayer>(0.5);
        } else if (type == "BatchNormLayer") {
            layer = std::make_unique<layers::BatchNormLayer>();
        } else if (type == "SimpleRNNLayer") {
            layer = std::make_unique<layers::SimpleRNNLayer>(1, 1);
        } else if (type == "LSTMLayer") {
            layer = std::make_unique<layers::LSTMLayer>(1, 1);
        } else if (type == "GRULayer") {
            layer = std::make_unique<layers::GRULayer>(1, 1);
        } else if (type == "PoolingLayer") {
            layer = std::make_unique<layers::PoolingLayer>(2, 2, layers::PoolingLayer::MAX, 1);
        } else {
            throw std::runtime_error("Unknown layer type: " + type);
        }
        
        layer->deserialize(in);
        layers_.push_back(std::move(layer));
    }
}
    std::string NeuralNetwork::get_model_type() const {
        return "NeuralNetwork";
    }

    //===========================================================================
    // UTILITY
    //===========================================================================
    
    void NeuralNetwork::set_regularizer(RegularizerType type, double strength,
                                        const std::unordered_map<std::string, double>& params) {
        regularizer_ = RegularizerFactory::create(type, strength, params);
    }

    void NeuralNetwork::set_optimizer(OptimizerType type, double learning_rate) {
        learning_rate_ = learning_rate;
        optimizer_ = OptimizerFactory::create(type, learning_rate);
    }

    void NeuralNetwork::initialize_layers(int input_size) {
        if (layers_.empty()) {
            add_dense_layer(n_classes_, "softmax");
        }
    }

    void NeuralNetwork::reset() {
        layers_.clear();
        forward_cache_.clear();
        fitted_ = false;
    }

    void NeuralNetwork::reset_history() {
        loss_history_.clear();
        val_loss_history_.clear();
        accuracy_history_.clear();
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
    NeuralNetwork::get_training_history() const {
        return std::make_tuple(loss_history_, val_loss_history_, accuracy_history_);
    }

    void NeuralNetwork::summary() const {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "              NEURAL NETWORK SUMMARY" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::cout << "Input size:          " << n_features_ << std::endl;
        std::cout << "Output size:         " << n_classes_ << std::endl;
        std::cout << "Number of layers:    " << layers_.size() << std::endl;
        std::cout << "Total parameters:    " << get_num_parameters() << std::endl;
        std::cout << "Loss function:       " << loss_function_ << std::endl;
        std::cout << "Optimizer:           " << (optimizer_ ? optimizer_->get_type_str() : "none") << std::endl;
        std::cout << "Learning rate:       " << learning_rate_ << std::endl;
        std::cout << "Fitted:              " << (fitted_ ? "yes" : "no") << std::endl;
        
        if (batch_size_ > 0) {
            std::cout << "Batch size:          " << batch_size_ << std::endl;
        }
        if (epochs_ > 0) {
            std::cout << "Epochs:              " << epochs_ << std::endl;
        }
        if (validation_split_ > 0) {
            std::cout << "Validation split:    " << validation_split_ << std::endl;
        }
        
        std::cout << std::string(50, '-') << std::endl;
        std::cout << "LAYER DETAILS:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            std::cout << "Layer " << i << ": " << layers_[i]->get_config() << std::endl;
            if (layers_[i]->has_weights()) {
                std::cout << "  Parameters: " << layers_[i]->get_parameter_count() << std::endl;
            }
        }
        
        std::cout << std::string(50, '=') << "\n" << std::endl;
    }

    int NeuralNetwork::get_num_parameters() const {
        int total = 0;
        for (const auto& layer : layers_) {
            total += layer->get_parameter_count();
        }
        return total;
    }

} // namespace models