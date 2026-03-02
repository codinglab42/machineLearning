#include "models/neural_network.h"
#include "exceptions/dimension_exception.h"
#include "exceptions/fitting_exception.h" 
#include "exceptions/io_exception.h"
#include "exceptions/validation_exception.h"
#include "exceptions/exception_macros.h"
#include "utils/serializable.h"
#include "components/optimizers/optimizer_factory.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>

namespace models {

    // Costruttori
    NeuralNetwork::NeuralNetwork() 
        : loss_function_("mse"),
          learning_rate_(0.01),
          verbose_(false),
          n_features_(0),
          n_classes_(0),
          fitted_(false) {
        optimizer_ = OptimizerFactory::create(OptimizerType::SGD, learning_rate_);
    }

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                                 const std::string& activation,
                                 const std::string& output_activation,
                                 OptimizerType optimizer_type,
                                 double learning_rate)
        : loss_function_("categorical_crossentropy"),
          learning_rate_(learning_rate),
          verbose_(false),
          n_features_(layer_sizes.front()),
          n_classes_(layer_sizes.back()),
          fitted_(false) {
        
        ML_CHECK_PARAM(layer_sizes.size() >= 2, "layer_sizes", 
                      "must have at least input and output layer", "NeuralNetwork");
        
        optimizer_ = OptimizerFactory::create(optimizer_type, learning_rate);
        
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
            
            if (auto* recurrent = dynamic_cast<layers::RecurrentLayer*>(layer.get())) {
                recurrent->set_input_shape(prev_output);
            }
        }
        
        layers_.push_back(std::move(layer));
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
        
        layers::Pooling::PoolType type = (pool_type == "max") ? 
            layers::Pooling::MAX : layers::Pooling::AVERAGE;
        
        int channels = 1;
        return std::make_unique<layers::Pooling>(pool_size, strides, type, channels);
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_recurrent_layer(
        RecurrentType type, int units, bool return_sequences,
        const std::string& activation) {
        
        int input_size = layers_.empty() ? n_features_ : layers_.back()->get_output_size();
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "NeuralNetwork");
        
        std::unique_ptr<layers::RecurrentLayer> layer;
        
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
    // TRAINING
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
                    batch_loss += compute_loss(y_batch(k), y_pred.row(k));
                }
                batch_loss /= current_batch_size;
                epoch_loss += batch_loss * current_batch_size;
                
                backward_pass(y_batch, y_pred);
            }
            
            epoch_loss /= n_samples;
            
            if (verbose_ && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
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
                    batch_loss += compute_loss(y_batch.row(k), y_pred.row(k));
                }
                batch_loss /= current_batch_size;
                epoch_loss += batch_loss * current_batch_size;
                
                Eigen::MatrixXd gradient = (y_pred - y_batch) / current_batch_size;
                for (int j = layers_.size() - 1; j >= 0; --j) {
                    gradient = layers_[j]->backward(gradient, learning_rate_);
                }
            }
            
            epoch_loss /= n_samples;
            
            if (verbose_ && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
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
            activation = layers_[i]->forward(activation);
        }
        
        return activation;
    }

    Eigen::VectorXd NeuralNetwork::backward_pass(const Eigen::VectorXd& y_true, 
                                                  const Eigen::MatrixXd& y_pred) {
        Eigen::MatrixXd gradient = (y_pred - y_true).transpose();
        
        for (int i = layers_.size() - 1; i >= 0; --i) {
            gradient = layers_[i]->backward(gradient, learning_rate_);
        }
        
        return gradient;
    }

    //===========================================================================
    // LOSS FUNCTIONS
    //===========================================================================
    
    double NeuralNetwork::compute_loss(const Eigen::VectorXd& y_true, 
                                        const Eigen::MatrixXd& y_pred) const {
        if (loss_function_ == "mse") {
            return (y_pred - y_true.transpose()).array().square().mean();
        } else if (loss_function_ == "binary_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            return -((true_val * pred.log() + (1 - true_val) * (1 - pred).log())).mean();
        } else if (loss_function_ == "categorical_crossentropy") {
            Eigen::ArrayXd pred = y_pred.array();
            Eigen::ArrayXd true_val = y_true.array();
            return -(true_val * pred.log()).sum() / y_true.size();
        }
        
        return 0.0;
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
                predictions(i) = max_idx;
            }
            return predictions;
        }
    }

    Eigen::MatrixXd NeuralNetwork::predict_proba(const Eigen::MatrixXd& X) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        
        return forward_pass(X, false);
    }

    double NeuralNetwork::score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");
        
        Eigen::VectorXd y_pred = predict(X);
        
        if (n_classes_ == 1) {
            double y_mean = y.mean();
            double ss_tot = (y.array() - y_mean).square().sum();
            double ss_res = (y.array() - y_pred.array()).square().sum();
            return 1.0 - (ss_res / ss_tot);
        } else {
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
        
        file.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&n_classes_), sizeof(int));
        file.write(reinterpret_cast<const char*>(&fitted_), sizeof(bool));
        
        size_t num_layers = layers_.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
        
        for (const auto& layer : layers_) {
            layer->serialize(file);
        }
    }

    void NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            ML_THROW_IO_ERROR(filename, "load", "NeuralNetwork");
        }
        
        file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_classes_), sizeof(int));
        file.read(reinterpret_cast<char*>(&fitted_), sizeof(bool));
        
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
        
        layers_.clear();
        // Nota: qui dovresti implementare la deserializzazione dei layer
        // in base al tipo salvato
    }

    //===========================================================================
    // UTILITY
    //===========================================================================
    
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

} // namespace models