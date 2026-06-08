#include <memory>
#include <Eigen/Dense>
#include "models/neural_network.h"
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

    NeuralNetwork::NeuralNetwork()
        : loss_function_name_("mse"),
          learning_rate_(0.01),
          verbose_(false),
          n_features_(0),
          n_classes_(0),
          batch_size_(32),
          fitted_(false),
          epochs_(100),
          validation_split_(0.0),
          rng_(42) {
        // Usa optimizer_ ereditato da Estimator
        optimizer_ = OptimizerFactory::create(OptimizerType::SGD, learning_rate_);
        loss_function_ = loss::LossFactory::create("mse");
        // regularizer_ rimane nullptr (nessuna regolarizzazione di default)
    }

    NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                             const std::string& activation,
                             const std::string& output_activation,
                             OptimizerType optimizer_type,
                             double learning_rate,
                             RegularizerType regularizer_type,
                             double regularizer_strength)
        : loss_function_name_("categorical_crossentropy"),
          learning_rate_(learning_rate),
          verbose_(false),
          n_features_(layer_sizes.front()),
          n_classes_(layer_sizes.back()),
          batch_size_(32),
          fitted_(false),
          epochs_(100),
          validation_split_(0.0),
          rng_(42) {

        ML_CHECK_PARAM(layer_sizes.size() >= 2, "layer_sizes",
                    "must have at least input and output layer", "NeuralNetwork");

        // Usa optimizer_ e regularizer_ ereditati da Estimator
        optimizer_   = OptimizerFactory::create(optimizer_type, learning_rate);
        regularizer_ = RegularizerFactory::create(regularizer_type, regularizer_strength);
        loss_function_ = loss::LossFactory::create("categorical_crossentropy");

        for (size_t i = 1; i < layer_sizes.size() - 1; ++i) {
            add_dense_layer(layer_sizes[i], activation);
        }
        add_dense_layer(layer_sizes.back(), output_activation);
    }

    //===========================================================================
    // DETERMINE PROBLEM TYPE
    //===========================================================================

    void NeuralNetwork::determine_problem_type(const Eigen::VectorXd& y) {
        double min_y = y.minCoeff();
        double max_y = y.maxCoeff();

        bool is_binary = true;
        for (int i = 0; i < y.size(); ++i) {
            if (y(i) > 0.1 && y(i) < 0.9) {
                is_binary = false;
                break;
            }
        }

        if (is_binary && max_y <= 1.0 && min_y >= 0.0) {
            n_classes_ = 2;
        } else if (n_classes_ > 1) {
            // già impostato come multi-class
        } else {
            n_classes_ = 1;
        }
    }

    //===========================================================================
    // LAYER MANAGEMENT
    //===========================================================================

    void NeuralNetwork::add_layer(std::unique_ptr<layers::Layer> layer) {
        if (!layers_.empty()) {
            int prev_output = layers_.back()->get_output_size();
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
                // FIX: le stringhe di attivazione vanno passate come chiavi separate,
                // non convertite da double. Usa "relu" come default sicuro.
                std::string activation = "relu";
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
                layer = create_conv2d_layer(filters, kernel_size, strides, "valid", "relu");
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
                RecurrentType rnn_type;
                switch (type) {
                    case LayerType::SIMPLE_RNN: rnn_type = RecurrentType::SIMPLE_RNN; break;
                    case LayerType::LSTM:       rnn_type = RecurrentType::LSTM;       break;
                    case LayerType::GRU:        rnn_type = RecurrentType::GRU;        break;
                    default:                    rnn_type = RecurrentType::SIMPLE_RNN;
                }
                layer = create_recurrent_layer(rnn_type, units, return_sequences, "tanh");
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
        add_layer(create_dense_layer(units, activation, use_bias));
    }

    void NeuralNetwork::add_conv2d_layer(int filters, int kernel_size, int strides,
                                         const std::string& padding,
                                         const std::string& activation) {
        add_layer(create_conv2d_layer(filters, kernel_size, strides, padding, activation));
    }

    void NeuralNetwork::add_pooling_layer(int pool_size, int strides, const std::string& pool_type) {
        add_layer(create_pooling_layer(pool_size, strides, pool_type));
    }

    void NeuralNetwork::add_flatten_layer() {
        add_layer(std::make_unique<layers::FlattenLayer>());
    }

    void NeuralNetwork::add_dropout_layer(double rate) {
        ML_CHECK_PARAM(rate >= 0.0 && rate < 1.0, "rate", "must be in [0, 1)", "NeuralNetwork");
        add_layer(std::make_unique<layers::DropoutLayer>(rate));
    }

    void NeuralNetwork::add_batch_norm_layer() {
        add_layer(std::make_unique<layers::BatchNormLayer>());
    }

    void NeuralNetwork::add_recurrent_layer(RecurrentType type, int units,
                                            bool return_sequences,
                                            const std::string& activation) {
        add_layer(create_recurrent_layer(type, units, return_sequences, activation));
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

        return std::make_unique<layers::Conv2DLayer>(
            filters, kernel_size, strides, padding, activation);
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_pooling_layer(
        int pool_size, int strides, const std::string& pool_type) {

        layers::PoolingLayer::PoolType type = (pool_type == "max") ?
            layers::PoolingLayer::MAX : layers::PoolingLayer::AVG;
        return std::make_unique<layers::PoolingLayer>(pool_size, strides, type, 1);
    }

    std::unique_ptr<layers::Layer> NeuralNetwork::create_recurrent_layer(
        RecurrentType type, int units, bool return_sequences,
        const std::string& activation) {

        int input_size = layers_.empty() ? n_features_ : layers_.back()->get_output_size();
        ML_CHECK_PARAM(input_size > 0, "input_size", "must be > 0", "NeuralNetwork");

        switch (type) {
            case RecurrentType::SIMPLE_RNN:
                return std::make_unique<layers::SimpleRNNLayer>(units, input_size, activation);
            case RecurrentType::LSTM:
                return std::make_unique<layers::LSTMLayer>(units, input_size, activation);
            case RecurrentType::GRU:
                return std::make_unique<layers::GRULayer>(units, input_size, activation);
        }
        return nullptr;
    }

    //===========================================================================
    // TRAINING
    //===========================================================================

    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit(X, y, epochs_, batch_size_, verbose_);
    }

    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
        fit(X, y, epochs_, batch_size_, verbose_);
    }

    // ---------------------------------------------------------------------------
    // fit con y vettore (classificazione binaria / regressione)
    // ---------------------------------------------------------------------------
    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                        int epochs, int batch_size, bool verbose) {

        ML_CHECK_NOT_EMPTY(X, "X", "NeuralNetwork");
        ML_CHECK_NOT_EMPTY(y, "y", "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");
        ML_CHECK_PARAM(epochs > 0,     "epochs",     "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", "NeuralNetwork");

        verbose_    = verbose;
        n_features_ = X.cols();
        determine_problem_type(y);

        if (layers_.empty()) {
            add_dense_layer(1, n_classes_ == 2 ? "sigmoid" : "linear");
        }

        build(n_features_, n_classes_);

        int n_samples = X.rows();
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        loss_history_.clear();
        loss_history_.reserve(epochs);

        if (verbose) {
            std::cout << "Optimizer: " << (optimizer_ ? optimizer_->get_type_str() : "none")
                      << "  LR: " << learning_rate_ << std::endl;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng_);

            double epoch_loss = 0.0;

            for (int i = 0; i < n_samples; i += batch_size) {
                int end               = std::min(i + batch_size, n_samples);
                int current_batch_sz  = end - i;

                Eigen::MatrixXd X_batch(current_batch_sz, X.cols());
                Eigen::VectorXd y_batch(current_batch_sz);
                for (int j = i; j < end; ++j) {
                    X_batch.row(j - i) = X.row(indices[j]);
                    y_batch(j - i)     = y(indices[j]);
                }

                // Forward
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);

                // Loss
                Eigen::MatrixXd y_true_mat(y_batch.size(), 1);
                for (int k = 0; k < y_batch.size(); ++k) y_true_mat(k, 0) = y_batch(k);
                double batch_loss = compute_loss(y_true_mat, y_pred);
                epoch_loss += batch_loss * current_batch_sz;

                // Gradient dalla loss
                Eigen::MatrixXd gradient = compute_loss_gradient(y_batch, y_pred);

                // Gradient clipping UNA SOLA VOLTA sul gradiente della loss
                clip_gradient(gradient, 1.0);

                // Backward — passa learning_rate 0: i layer calcolano i gradienti
                // ma NON aggiornano i pesi (solo DenseLayer rispetta questo;
                // RNN/GRU/LSTM verranno fixati nel passo 2)
                for (int j = static_cast<int>(layers_.size()) - 1; j >= 0; --j) {
                    gradient = layers_[j]->backward(gradient);
                }

                // Aggiornamento pesi tramite optimizer_ del padre (Estimator)
                for (auto& layer : layers_) {
                    if (!layer->has_weights()) continue;

                    // Ora weights e weights_gradient hanno le STESSE dimensioni!
                    Eigen::MatrixXd w = layer->get_weights();
                    Eigen::MatrixXd w_grad = layer->get_weights_gradient();
                    w_grad /= current_batch_sz;
                    
                    // Aggiungi regolarizzazione (escludendo l'ultima colonna se sono bias)
                    if (regularizer_) {
                        Eigen::MatrixXd weights_for_reg = w;
                        if (layer->get_use_bias() && w.cols() > 0) {
                            // Non regolarizzare l'ultima colonna (bias)
                            weights_for_reg.rightCols(1).setZero();
                        }
                        w_grad += regularizer_->compute_gradient(weights_for_reg);
                    }
                    
                    // Aggiorna pesi (inclusi i bias nell'ultima colonna!)
                    optimizer_->update(w, w_grad);
                    layer->set_weights(w);
                    
                    // NON serve più gestire i bias separatamente!
                }
            }

            epoch_loss /= n_samples;
            loss_history_.push_back(epoch_loss);

            if (verbose && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "  Loss: " << epoch_loss << std::endl;
            }
        }

        fitted_ = true;
    }

    // ---------------------------------------------------------------------------
    // fit con y matrice (multi-class / multi-output)
    // ORA USA L'OTTIMIZZATORE come il fit con VectorXd
    // ---------------------------------------------------------------------------
    void NeuralNetwork::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                        int epochs, int batch_size, bool verbose) {

        ML_CHECK_NOT_EMPTY(X, "X", "NeuralNetwork");
        ML_CHECK_NOT_EMPTY(y, "y", "NeuralNetwork");
        ML_CHECK_XY_SIZE_MATRIX(X.rows(), y.rows(), "NeuralNetwork");
        ML_CHECK_PARAM(epochs > 0,     "epochs",     "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", "NeuralNetwork");

        verbose_    = verbose;
        n_features_ = X.cols();
        n_classes_  = y.cols();  // ← FONDAMENTALE!

        if (layers_.empty()) {
            // Crea un layer di output appropriato
            if (n_classes_ == 1) {
                add_dense_layer(1, "linear", true);
            } else {
                add_dense_layer(n_classes_, "softmax", false);
            }
        }

        // Costruisci la rete se necessario
        if (!fitted_) {
            build(n_features_, n_classes_);
        }

        int n_samples = X.rows();
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        forward_cache_.resize(layers_.size());

        loss_history_.clear();
        loss_history_.reserve(epochs);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng_);

            double epoch_loss = 0.0;

            for (int i = 0; i < n_samples; i += batch_size) {
                int end = std::min(i + batch_size, n_samples);
                int current_batch_size = end - i;

                Eigen::MatrixXd X_batch(current_batch_size, X.cols());
                Eigen::MatrixXd y_batch(current_batch_size, y.cols());

                for (int j = i; j < end; ++j) {
                    X_batch.row(j - i) = X.row(indices[j]);
                    y_batch.row(j - i) = y.row(indices[j]);
                }

                // Forward pass
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);

                // Loss
                double batch_loss = compute_loss(y_batch, y_pred);
                epoch_loss += batch_loss * current_batch_size;

                // Gradient
                Eigen::MatrixXd gradient = loss_function_->gradient(y_batch, y_pred);

                // Gradient clipping
                double norm = gradient.norm();
                if (norm > 1.0 && norm > 0) {
                    gradient *= (1.0 / norm);
                }

                // Backward pass
                for (int j = static_cast<int>(layers_.size()) - 1; j >= 0; --j) {
                    gradient = layers_[j]->backward(gradient);
                }

                // Update weights
                for (auto& layer : layers_) {
                    if (!layer->has_weights()) continue;

                    Eigen::MatrixXd weights = layer->get_weights();
                    Eigen::MatrixXd w_grad = layer->get_weights_gradient();
                    w_grad /= static_cast<double>(current_batch_size);

                    if (regularizer_) {
                        Eigen::MatrixXd weights_for_reg = weights;
                        if (layer->get_use_bias() && weights.cols() > 0) {
                            weights_for_reg.rightCols(1).setZero();
                        }
                        w_grad += regularizer_->compute_gradient(weights_for_reg);
                    }

                    optimizer_->update(weights, w_grad);
                    layer->set_weights(weights);
                }
            }

            epoch_loss /= n_samples;
            loss_history_.push_back(epoch_loss);

            if (verbose_ && epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "  Loss: " << epoch_loss << std::endl;
            }
        }

        fitted_ = true;
    }

    void NeuralNetwork::build(int n_features, int n_classes) {
        // 1. Verifica che ci siano layer
        if (layers_.empty()) {
            throw ml_exception::MLException(
                "NeuralNetwork::build: cannot build a network with no layers. "
                "Use add_layer() first.");
        }
        
        // 2. Validazione dimensioni
        ML_CHECK_PARAM(n_features > 0, "n_features", "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(n_classes > 0, "n_classes", "must be > 0", "NeuralNetwork");
        
        // 3. Imposta le dimensioni della rete
        n_features_ = n_features;
        n_classes_ = n_classes;
        
        // 4. Propaga le dimensioni e inizializza i pesi
        int current_dim = n_features_;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            // Imposta l'input shape per il layer
            layers_[i]->set_input_shape(current_dim);
            
            // Inizializza i pesi (se il layer li ha)
            layers_[i]->initialize_weights();
            
            // Aggiorna la dimensione per il layer successivo
            current_dim = layers_[i]->get_output_size();
            
            // Verifica che la dimensione sia valida dopo ogni layer
            if (current_dim <= 0) {
                throw ml_exception::MLException(
                    "NeuralNetwork::build: layer " + std::to_string(i) + 
                    " produced invalid output dimension: " + std::to_string(current_dim));
            }
        }
        
        // 5. Verifica opzionale: l'output dell'ultimo layer dovrebbe coincidere con n_classes
        if (current_dim != n_classes_) {
            // Avviso ma non errore (potrebbe essere regressione o configurazione speciale)
            if (verbose_) {
                std::cout << "Warning: NeuralNetwork::build - Last layer output (" 
                        << current_dim << ") != n_classes (" << n_classes_ << ")" << std::endl;
            }
        }
        
        // 6. La rete è costruita ma non ancora addestrata
        fitted_ = false;
        
        // 7. Log di successo (opzionale)
        if (verbose_) {
            std::cout << "✓ NeuralNetwork built successfully: " 
                    << n_features_ << " -> " << current_dim 
                    << " (expected output: " << n_classes_ << ")" << std::endl;
        }
    }

    //===========================================================================
    // FORWARD / BACKWARD
    //===========================================================================

    void NeuralNetwork::clip_gradient(Eigen::MatrixXd& gradient, double max_norm) {
        double norm = gradient.norm();
        if (norm > max_norm && norm > 0) {
            gradient *= (max_norm / norm);
        }
    }

    Eigen::MatrixXd NeuralNetwork::forward_pass(const Eigen::MatrixXd& X, bool training) const {
        Eigen::MatrixXd activation = X;
        for (size_t i = 0; i < layers_.size(); ++i) {
            activation = layers_[i]->forward(activation, training);
            if (i < forward_cache_.size()) {
                forward_cache_[i] = layers_[i]->get_cache();
            }
        }
        return activation;
    }

    Eigen::VectorXd NeuralNetwork::backward_pass(const Eigen::VectorXd& y_true,
                                                const Eigen::MatrixXd& y_pred) {
        if (!loss_function_) {
            ML_THROW_FITTING_ERROR("NeuralNetwork", "loss function not set");
        }

        Eigen::MatrixXd y_true_mat(y_true.size(), 1);
        for (int i = 0; i < y_true.size(); ++i) y_true_mat(i, 0) = y_true(i);

        Eigen::MatrixXd gradient = loss_function_->gradient(y_true_mat, y_pred);
        clip_gradient(gradient, 1.0);

        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            gradient = layers_[i]->backward(gradient);
        }

        return gradient;
    }

    //===========================================================================
    // LOSS
    //===========================================================================

    void NeuralNetwork::set_loss_function(const std::string& loss) {
        loss_function_name_ = loss;
        loss_function_ = loss::LossFactory::create(loss);
    }

    double NeuralNetwork::compute_loss(const Eigen::VectorXd& y_true,
                                        const Eigen::MatrixXd& y_pred) const {
        if (!loss_function_) ML_THROW_FITTING_ERROR("NeuralNetwork", "loss function not set");

        Eigen::MatrixXd y_true_mat(y_true.size(), 1);
        for (int i = 0; i < y_true.size(); ++i) y_true_mat(i, 0) = y_true(i);

        double data_loss = loss_function_->compute(y_true_mat, y_pred);
        double reg_loss  = 0.0;
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
        if (!loss_function_) ML_THROW_FITTING_ERROR("NeuralNetwork", "loss function not set");

        double data_loss = loss_function_->compute(y_true, y_pred);
        double reg_loss  = 0.0;
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
        if (!loss_function_) ML_THROW_FITTING_ERROR("NeuralNetwork", "loss function not set");

        Eigen::MatrixXd y_true_mat(y_true.size(), 1);
        for (int i = 0; i < y_true.size(); ++i) y_true_mat(i, 0) = y_true(i);

        return loss_function_->gradient(y_true_mat, y_pred);
    }

    //===========================================================================
    // PREDICT / SCORE
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

    double NeuralNetwork::score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");

        Eigen::VectorXd y_pred = predict(X);

        if (n_classes_ == 2) {
            int correct = 0;
            for (int i = 0; i < y.size(); ++i) {
                if ((y_pred(i) > 0.5) == (y(i) > 0.5)) correct++;
            }
            return static_cast<double>(correct) / y.size();
        } else if (n_classes_ == 1) {
            double y_mean = y.mean();
            double ss_tot = (y.array() - y_mean).square().sum();
            double ss_res = (y.array() - y_pred.array()).square().sum();
            return 1.0 - (ss_res / (ss_tot + 1e-7));
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
        if (!file.is_open()) ML_THROW_IO_ERROR(filename, "save", "NeuralNetwork");
        serialize_binary(file);
    }

    void NeuralNetwork::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) ML_THROW_IO_ERROR(filename, "load", "NeuralNetwork");
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

    //===========================================================================
// SERIALIZZAZIONE CORRETTA
//===========================================================================

    void NeuralNetwork::serialize_binary(std::ostream& out) const {
        // Verifica che la rete sia in uno stato valido
        if (n_features_ <= 0 || n_classes_ <= 0) {
            throw std::runtime_error("NeuralNetwork::serialize_binary: "
                "Network not initialized (n_features or n_classes = 0)");
        }
        
        // 1. MAGIC NUMBER per verifica formato (opzionale ma consigliato)
        uint32_t magic = 0x4E4E574F; // "NNWO" - Neural NetWork Object
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // 2. Versione del formato
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // 3. Parametri base (usa tipi a dimensione fissa!)
        int32_t n_features = static_cast<int32_t>(n_features_);
        int32_t n_classes = static_cast<int32_t>(n_classes_);
        out.write(reinterpret_cast<const char*>(&n_features), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&n_classes), sizeof(int32_t));
        
        int8_t fitted_flag = fitted_ ? 1 : 0;
        out.write(reinterpret_cast<const char*>(&fitted_flag), sizeof(int8_t));
        
        // 4. Loss function (con lunghezza int32_t)
        int32_t loss_name_len = static_cast<int32_t>(loss_function_name_.size());
        out.write(reinterpret_cast<const char*>(&loss_name_len), sizeof(int32_t));
        out.write(loss_function_name_.c_str(), loss_name_len);
        
        // 5. Learning rate
        out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
        
        // 6. Ottimizzatore
        if (optimizer_) {
            int32_t opt_type = static_cast<int32_t>(optimizer_->get_type());
            out.write(reinterpret_cast<const char*>(&opt_type), sizeof(int32_t));
            optimizer_->serialize(out);
        } else {
            int32_t opt_type = -1;
            out.write(reinterpret_cast<const char*>(&opt_type), sizeof(int32_t));
        }
        
        // 7. Regularizer
        if (regularizer_) {
            int32_t reg_type = static_cast<int32_t>(regularizer_->get_type());
            out.write(reinterpret_cast<const char*>(&reg_type), sizeof(int32_t));
            regularizer_->serialize(out);
        } else {
            int32_t reg_type = -1;
            out.write(reinterpret_cast<const char*>(&reg_type), sizeof(int32_t));
        }
        
        // 8. Numero di layer (int32_t)
        int32_t num_layers = static_cast<int32_t>(layers_.size());
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(int32_t));
        
        // 9. Per ogni layer: tipo + dati
        for (const auto& layer : layers_) {
            if (!layer) {
                throw std::runtime_error("NeuralNetwork::serialize_binary: null layer");
            }
            
            // Scrivi il tipo con lunghezza int32_t
            std::string layer_type = layer->get_type();
            int32_t type_len = static_cast<int32_t>(layer_type.size());
            out.write(reinterpret_cast<const char*>(&type_len), sizeof(int32_t));
            out.write(layer_type.c_str(), type_len);
            
            // Il layer si serializza da solo (DEVE usare int32_t per le dimensioni)
            layer->serialize(out);
        }
        
        if (!out.good()) {
            throw std::runtime_error("NeuralNetwork::serialize_binary: stream error");
        }
    }

    void NeuralNetwork::deserialize_binary(std::istream& in) {
        if (!in.good()) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: input stream not good");
        }
        
        // 1. Leggi magic number
        uint32_t magic;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x4E4E574F) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid file format");
        }
        
        // 2. Leggi versione
        uint32_t version;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: unsupported version");
        }
        
        // 3. Leggi parametri base
        int32_t n_features, n_classes;
        in.read(reinterpret_cast<char*>(&n_features), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&n_classes), sizeof(int32_t));
        
        if (n_features <= 0 || n_features > 1000000) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid n_features");
        }
        if (n_classes <= 0 || n_classes > 1000000) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid n_classes");
        }
        
        n_features_ = n_features;
        n_classes_ = n_classes;
        
        int8_t fitted_flag;
        in.read(reinterpret_cast<char*>(&fitted_flag), sizeof(int8_t));
        fitted_ = (fitted_flag != 0);
        
        // 4. Leggi loss function
        int32_t loss_name_len;
        in.read(reinterpret_cast<char*>(&loss_name_len), sizeof(int32_t));
        if (loss_name_len < 0 || loss_name_len > 1024) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid loss name length");
        }
        std::vector<char> loss_name_buf(loss_name_len + 1, '\0');
        in.read(loss_name_buf.data(), loss_name_len);
        loss_function_name_ = std::string(loss_name_buf.data());
        loss_function_ = loss::LossFactory::create(loss_function_name_);
        
        // 5. Leggi learning rate
        in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
        
        // 6. Leggi ottimizzatore
        int32_t opt_type;
        in.read(reinterpret_cast<char*>(&opt_type), sizeof(int32_t));
        if (opt_type >= 0) {
            optimizer_ = OptimizerFactory::create(static_cast<OptimizerType>(opt_type), learning_rate_);
            if (optimizer_) {
                optimizer_->deserialize(in);
            }
        }
        
        // 7. Leggi regularizer
        int32_t reg_type;
        in.read(reinterpret_cast<char*>(&reg_type), sizeof(int32_t));
        if (reg_type >= 0) {
            regularizer_ = RegularizerFactory::create(static_cast<RegularizerType>(reg_type), 0.01);
            if (regularizer_) {
                regularizer_->deserialize(in);
            }
        }
        
        // 8. Leggi numero di layer
        int32_t num_layers;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(int32_t));
        if (num_layers < 0 || num_layers > 10000) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid number of layers");
        }
        
        // 9. Pulisci e ricostruisci i layer
        layers_.clear();
        forward_cache_.clear();
        layers_.reserve(num_layers);
        forward_cache_.resize(num_layers);
        
        // 10. Leggi ogni layer
        for (int32_t i = 0; i < num_layers; ++i) {
            // Leggi tipo (lunghezza int32_t)
            int32_t type_len;
            in.read(reinterpret_cast<char*>(&type_len), sizeof(int32_t));
            if (type_len < 0 || type_len > 256) {
                throw std::runtime_error("NeuralNetwork::deserialize_binary: invalid layer type length");
            }
            
            std::vector<char> type_buf(type_len + 1, '\0');
            in.read(type_buf.data(), type_len);
            std::string layer_type(type_buf.data());
            
            // Crea il layer - ORA usa deserialize che legge le dimensioni reali
            std::unique_ptr<layers::Layer> layer;
            
            // IMPORTANTE: Ogni layer DEVE avere un costruttore default
            // e il metodo deserialize DEVE leggere TUTTI i parametri (incluse dimensioni)
            if (layer_type == "DenseLayer") {
                auto dense = std::make_unique<layers::DenseLayer>();  // Costruttore default
                dense->deserialize(in);  // Legge input_size, output_size, activation, weights, bias
                layer = std::move(dense);
            }
            else if (layer_type == "Conv2DLayer") {
                auto conv = std::make_unique<layers::Conv2DLayer>();
                conv->deserialize(in);
                layer = std::move(conv);
            }
            else if (layer_type == "FlattenLayer") {
                auto flatten = std::make_unique<layers::FlattenLayer>();
                flatten->deserialize(in);
                layer = std::move(flatten);
            }
            else if (layer_type == "DropoutLayer") {
                auto dropout = std::make_unique<layers::DropoutLayer>();
                dropout->deserialize(in);
                layer = std::move(dropout);
            }
            else if (layer_type == "BatchNormLayer") {
                auto bn = std::make_unique<layers::BatchNormLayer>();
                bn->deserialize(in);
                layer = std::move(bn);
            }
            else if (layer_type == "SimpleRNNLayer") {
                auto rnn = std::make_unique<layers::SimpleRNNLayer>();
                rnn->deserialize(in);
                layer = std::move(rnn);
            }
            else if (layer_type == "LSTMLayer") {
                auto lstm = std::make_unique<layers::LSTMLayer>();
                lstm->deserialize(in);
                layer = std::move(lstm);
            }
            else if (layer_type == "GRULayer") {
                auto gru = std::make_unique<layers::GRULayer>();
                gru->deserialize(in);
                layer = std::move(gru);
            }
            else if (layer_type == "PoolingLayer") {
                auto pool = std::make_unique<layers::PoolingLayer>();
                pool->deserialize(in);
                layer = std::move(pool);
            }
            else {
                throw std::runtime_error("NeuralNetwork::deserialize_binary: unknown layer type: " + layer_type);
            }
            
            if (!layer) {
                throw std::runtime_error("NeuralNetwork::deserialize_binary: failed to create layer: " + layer_type);
            }
            
            layers_.push_back(std::move(layer));
        }
        
        // 11. Ricostruisci le connessioni tra layer (input shapes)
        int expected_input = n_features_;
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i]->set_input_shape(expected_input);
            expected_input = layers_[i]->get_output_size();
        }
        
        if (!in.good() && !in.eof()) {
            throw std::runtime_error("NeuralNetwork::deserialize_binary: stream error");
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
        // Usa regularizer_ del padre (Estimator)
        regularizer_ = RegularizerFactory::create(type, strength, params);
    }

    void NeuralNetwork::set_optimizer(OptimizerType type, double learning_rate) {
        learning_rate_ = learning_rate;
        // Usa optimizer_ del padre (Estimator)
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
        std::cout << "          NEURAL NETWORK SUMMARY" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Input size:       " << n_features_   << std::endl;
        std::cout << "Output size:      " << n_classes_    << std::endl;
        std::cout << "Layers:           " << layers_.size()<< std::endl;
        std::cout << "Parameters:       " << get_num_parameters() << std::endl;
        std::cout << "Loss:             " << loss_function_name_  << std::endl;
        std::cout << "Optimizer:        " << (optimizer_ ? optimizer_->get_type_str() : "none") << std::endl;
        std::cout << "Learning rate:    " << learning_rate_ << std::endl;
        std::cout << "Fitted:           " << (fitted_ ? "yes" : "no") << std::endl;
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
        for (const auto& layer : layers_) total += layer->get_parameter_count();
        return total;
    }

} // namespace models