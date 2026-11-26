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
#include <sstream>
#include <cstring>
#include <cstdint>
#include <limits>
#include <cmath>

namespace models {

namespace {
    constexpr int NETWORK_FORMAT_MARKER = -1296977744;
    constexpr std::uint32_t NETWORK_FORMAT_VERSION = 4;

    template<typename T>
    void read_checked(std::istream& in, T& value) {
        if (!in.read(reinterpret_cast<char*>(&value), sizeof(value))) {
            ML_THROW_DESERIALIZATION_ERROR("Truncated network data", "NeuralNetwork");
        }
    }

    void write_rng_state(std::ostream& out, const std::mt19937& rng) {
        std::ostringstream state;
        state << rng;
        const std::string serialized_state = state.str();
        const std::uint64_t state_size = serialized_state.size();
        utils::write_scalar(out, state_size);
        out.write(serialized_state.data(), static_cast<std::streamsize>(serialized_state.size()));
        if (!out) {
            ML_THROW_SERIALIZATION_ERROR("Failed to write random generator state", "NeuralNetwork");
        }
    }

    void read_rng_state(std::istream& in, std::mt19937& rng) {
        std::uint64_t state_size = 0;
        read_checked(in, state_size);
        if (state_size == 0 || state_size > 16384) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid random generator state size", "NeuralNetwork");
        }
        std::string serialized_state(static_cast<std::size_t>(state_size), '\0');
        if (!in.read(serialized_state.data(), static_cast<std::streamsize>(state_size))) {
            ML_THROW_DESERIALIZATION_ERROR("Truncated random generator state", "NeuralNetwork");
        }
        std::istringstream state(serialized_state);
        if (!(state >> rng)) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid random generator state", "NeuralNetwork");
        }
        state >> std::ws;
        if (!state.eof()) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid random generator state", "NeuralNetwork");
        }
    }

    class OptimizerPayloadReader {
    public:
        explicit OptimizerPayloadReader(const std::string& data) : data_(data) {}

        template<typename T>
        T read() {
            if (sizeof(T) > data_.size() - offset_) {
                ML_THROW_DESERIALIZATION_ERROR("Truncated optimizer state", "NeuralNetwork");
            }
            T value;
            std::memcpy(&value, data_.data() + offset_, sizeof(T));
            offset_ += sizeof(T);
            return value;
        }

        void skip(std::size_t count) {
            if (count > data_.size() - offset_) {
                ML_THROW_DESERIALIZATION_ERROR("Truncated optimizer state", "NeuralNetwork");
            }
            offset_ += count;
        }

        void matrix(Eigen::Index expected_rows, Eigen::Index expected_cols, bool initialized) {
            const int rows = read<int>();
            const int cols = read<int>();
            if (initialized ? (rows != expected_rows || cols != expected_cols)
                            : (rows != 0 || cols != 0)) {
                ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer matrix dimensions", "NeuralNetwork");
            }
            skip(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(double));
        }

        void vector(Eigen::Index expected_size, bool initialized) {
            const int size = read<int>();
            if (size != (initialized ? expected_size : 0)) {
                ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer vector dimensions", "NeuralNetwork");
            }
            skip(static_cast<std::size_t>(size) * sizeof(double));
        }

        void boolean() {
            if (sizeof(bool) > data_.size() - offset_) {
                ML_THROW_DESERIALIZATION_ERROR("Truncated optimizer state", "NeuralNetwork");
            }
            const bool false_value = false;
            const bool true_value = true;
            const char* value = data_.data() + offset_;
            if (std::memcmp(value, &false_value, sizeof(bool)) != 0 &&
                std::memcmp(value, &true_value, sizeof(bool)) != 0) {
                ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer boolean", "NeuralNetwork");
            }
            offset_ += sizeof(bool);
        }

        bool finished() const { return offset_ == data_.size(); }

    private:
        const std::string& data_;
        std::size_t offset_ = 0;
    };

    void validate_optimizer_payload(const std::string& data, OptimizerType type,
                                    Eigen::Index rows, Eigen::Index cols, Eigen::Index bias_size) {
        OptimizerPayloadReader reader(data);
        const double learning_rate = reader.read<double>();
        const double decay = reader.read<double>();
        const int iterations = reader.read<int>();
        if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
            !std::isfinite(decay) || decay < 0 || iterations < 0) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer settings", "NeuralNetwork");
        }
        const bool initialized = iterations > 0;
        switch (type) {
            case OptimizerType::SGD:
                break;
            case OptimizerType::MOMENTUM:
                if (!std::isfinite(reader.read<double>())) {
                    ML_THROW_DESERIALIZATION_ERROR("Invalid momentum", "NeuralNetwork");
                }
                reader.boolean();
                reader.matrix(rows, cols, initialized);
                reader.vector(bias_size, initialized);
                break;
            case OptimizerType::ADAM:
            {
                const double beta1 = reader.read<double>();
                const double beta2 = reader.read<double>();
                const double epsilon = reader.read<double>();
                if (!(beta1 > 0 && beta1 < 1 && beta2 > 0 && beta2 < 1 &&
                      std::isfinite(epsilon) && epsilon > 0)) {
                    ML_THROW_DESERIALIZATION_ERROR("Invalid Adam settings", "NeuralNetwork");
                }
                reader.matrix(rows, cols, initialized);
                reader.matrix(rows, cols, initialized);
                reader.vector(bias_size, initialized);
                reader.vector(bias_size, initialized);
                break;
            }
            case OptimizerType::RMSPROP:
            {
                const double rho = reader.read<double>();
                const double epsilon = reader.read<double>();
                const double momentum = reader.read<double>();
                if (!(rho > 0 && rho < 1 && std::isfinite(epsilon) && epsilon > 0 &&
                      momentum >= 0 && momentum < 1)) {
                    ML_THROW_DESERIALIZATION_ERROR("Invalid RMSprop settings", "NeuralNetwork");
                }
                reader.matrix(rows, cols, initialized);
                reader.vector(bias_size, initialized);
                reader.matrix(rows, cols, initialized);
                reader.vector(bias_size, initialized);
                break;
            }
            case OptimizerType::ADAGRAD:
            {
                const double epsilon = reader.read<double>();
                if (!std::isfinite(epsilon) || epsilon <= 0) {
                    ML_THROW_DESERIALIZATION_ERROR("Invalid Adagrad settings", "NeuralNetwork");
                }
                reader.matrix(rows, cols, initialized);
                reader.vector(bias_size, initialized);
                break;
            }
        }
        if (!reader.finished()) {
            ML_THROW_DESERIALIZATION_ERROR("Unexpected optimizer state data", "NeuralNetwork");
        }
    }

    void write_optimizer(std::ostream& out, const Optimizer& optimizer) {
        const auto type = static_cast<std::uint8_t>(optimizer.get_type());
        std::ostringstream payload(std::ios::out | std::ios::binary);
        optimizer.serialize(payload);
        if (!payload) {
            ML_THROW_SERIALIZATION_ERROR("Failed to serialize optimizer", "NeuralNetwork");
        }
        const std::string data = payload.str();
        const std::uint64_t size = data.size();
        utils::write_scalar(out, type);
        utils::write_scalar(out, size);
        out.write(data.data(), data.size());
    }

    std::unique_ptr<Optimizer> read_optimizer(std::istream& in, double learning_rate,
                                               Eigen::Index rows, Eigen::Index cols,
                                               Eigen::Index bias_size) {
        std::uint8_t type_value = 0;
        std::uint64_t size = 0;
        read_checked(in, type_value);
        read_checked(in, size);
        if (type_value > static_cast<std::uint8_t>(OptimizerType::ADAGRAD)) {
            ML_THROW_DESERIALIZATION_ERROR("Unknown optimizer type", "NeuralNetwork");
        }
        const auto type = static_cast<OptimizerType>(type_value);
        const auto elements = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols)
                            + static_cast<std::uint64_t>(bias_size);
        if (elements > (std::numeric_limits<std::uint64_t>::max() - 512) / (2 * sizeof(double)) ||
            size > 512 + elements * 2 * sizeof(double) ||
            size > std::numeric_limits<std::size_t>::max() ||
            size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer state size", "NeuralNetwork");
        }
        std::string data(static_cast<std::size_t>(size), '\0');
        if (!in.read(data.data(), static_cast<std::streamsize>(size))) {
            ML_THROW_DESERIALIZATION_ERROR("Truncated optimizer state", "NeuralNetwork");
        }
        validate_optimizer_payload(data, type, rows, cols, bias_size);
        auto optimizer = OptimizerFactory::create(type, learning_rate);
        std::istringstream payload(data, std::ios::in | std::ios::binary);
        optimizer->deserialize(payload);
        if (!payload) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer state", "NeuralNetwork");
        }
        return optimizer;
    }
}

    //===========================================================================
    // COSTRUTTORI
    //===========================================================================

    NeuralNetwork::NeuralNetwork()
        : batch_size_(32),
          epochs_(100),
          validation_split_(0.0),
          loss_function_name_("mse"),
          learning_rate_(0.01),
          verbose_(false),
          n_features_(0),
          n_classes_(0),
          fitted_(false),
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
                         double regularizer_strength,
                         std::uint32_t seed)
    : batch_size_(32),
      epochs_(100),
      validation_split_(0.0),
      loss_function_name_(layer_sizes.back() <= 2 ? "binary_crossentropy" : "categorical_crossentropy"),
      learning_rate_(learning_rate),
      verbose_(false),
      n_features_(layer_sizes.front()),
      n_classes_(layer_sizes.back()),
      fitted_(false),
      rng_(seed) {

        ML_CHECK_PARAM(layer_sizes.size() >= 2, "layer_sizes",
                    "must have at least input and output layer", "NeuralNetwork");

        optimizer_   = OptimizerFactory::create(optimizer_type, learning_rate);
        regularizer_ = RegularizerFactory::create(regularizer_type, regularizer_strength);
        loss_function_ = loss::LossFactory::create(loss_function_name_);

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
        layer_optimizers_.clear();
        fitted_ = false;
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

        const int input_size = layers_.empty() ? std::max(n_features_, 1) : layers_.back()->get_output_size();
        std::unique_ptr<layers::RecurrentLayer> layer;
        switch (type) {
            case RecurrentType::SIMPLE_RNN:
                layer = std::make_unique<layers::SimpleRNNLayer>(units, input_size, activation);
                break;
            case RecurrentType::LSTM:
                layer = std::make_unique<layers::LSTMLayer>(units, input_size, activation);
                break;
            case RecurrentType::GRU:
                layer = std::make_unique<layers::GRULayer>(units, input_size, activation);
                break;
            default:
                ML_THROW_PARAMETER_ERROR("type", "unknown recurrent layer type", "NeuralNetwork");
        }
        layer->set_return_sequences(return_sequences);
        return layer;
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
        if (fitted_ && X.cols() != n_features_) {
            ML_THROW_PARAMETER_ERROR("X", "feature count differs from the fitted network", "NeuralNetwork");
        }
        n_features_ = X.cols();
        determine_problem_type(y);

        if (layers_.empty()) {
            add_dense_layer(1, n_classes_ == 2 ? "sigmoid" : "linear");
        }

        if (!fitted_) {
            build(n_features_, n_classes_);
        }
        initialize_optimizer_states();

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

                // 1. Forward pass
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);

                // 2. Accumula la loss locale per monitoraggio
                epoch_loss += compute_loss(y_batch, y_pred) * current_batch_sz;

                // 3. Calcola il gradiente iniziale dalla loss function
                Eigen::MatrixXd loss_grad = compute_loss_gradient(y_batch, y_pred);

                // 4. Backward pass lungo tutta la rete (aggiorna i gradienti interni dei layer)
                // Se la tua backward_pass accetta matrici/vettori di gradiente:
                Eigen::MatrixXd current_grad = loss_grad;
                for (int l = layers_.size() - 1; l >= 0; --l) {
                    current_grad = layers_[l]->backward(current_grad);
                }

                update_parameters();
            }

            epoch_loss /= n_samples;
            loss_history_.push_back(epoch_loss);

            if (verbose && (epoch % (epochs / 10 == 0 ? 1 : epochs / 10) == 0)) {
                std::cout << "Epoch " << epoch << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
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
        if (fitted_ && X.cols() != n_features_) {
            ML_THROW_PARAMETER_ERROR("X", "feature count differs from the fitted network", "NeuralNetwork");
        }
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
        initialize_optimizer_states();

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
                int end               = std::min(i + batch_size, n_samples);
                int current_batch_sz  = end - i;

                // Sostituisci il ciclo di estrazione del batch nella seconda fit con questo:
                Eigen::MatrixXd X_batch(current_batch_sz, X.cols());
                Eigen::MatrixXd y_batch(current_batch_sz, y.cols()); // <-- Ora è una MatrixXd corretta

                for (int j = i; j < end; ++j) {
                    X_batch.row(j - i) = X.row(indices[j]);
                    y_batch.row(j - i) = y.row(indices[j]); // <-- Estrazione della riga intera
                }

                // 1. Forward pass
                Eigen::MatrixXd y_pred = forward_pass(X_batch, true);

                // 2. Accumula la loss locale per monitoraggio
                epoch_loss += compute_loss(y_batch, y_pred) * current_batch_sz;

                // 3. Calcola il gradiente iniziale dalla loss function
                Eigen::MatrixXd loss_grad = compute_loss_gradient(y_batch, y_pred);

                // 4. Backward pass lungo tutta la rete (aggiorna i gradienti interni dei layer)
                // Se la tua backward_pass accetta matrici/vettori di gradiente:
                Eigen::MatrixXd current_grad = loss_grad;
                for (int l = layers_.size() - 1; l >= 0; --l) {
                    current_grad = layers_[l]->backward(current_grad);
                }

                update_parameters();
            }

            epoch_loss /= n_samples;
            loss_history_.push_back(epoch_loss);

            if (verbose && (epoch % (epochs / 10 == 0 ? 1 : epochs / 10) == 0)) {
                std::cout << "Epoch " << epoch << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
            }
        }

        fitted_ = true;
    }

    namespace {
        Eigen::MatrixXd sequence_matrix(const layers::SequenceTensor& tensor) {
            tensor.validate();
            using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            return Eigen::Map<const RowMajorMatrix>(tensor.data.data(),
                static_cast<Eigen::Index>(tensor.batch_size) * tensor.time_steps, tensor.features);
        }

        layers::SequenceTensor matrix_tensor(const Eigen::MatrixXd& matrix) {
            layers::SequenceTensor tensor{{}, static_cast<int>(matrix.rows()), 1,
                                           static_cast<int>(matrix.cols()), false};
            tensor.data.resize(static_cast<std::size_t>(matrix.rows()) * matrix.cols());
            for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
                for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
                    tensor.data[static_cast<std::size_t>(row) * matrix.cols() + col] = matrix(row, col);
                }
            }
            return tensor;
        }

        layers::SequenceTensor sequence_batch(const layers::SequenceTensor& tensor,
                                               const std::vector<int>& indices,
                                               int begin, int end) {
            layers::SequenceTensor batch{{}, end - begin, tensor.time_steps,
                                          tensor.features, tensor.is_sequence};
            const auto sample_size = static_cast<std::size_t>(tensor.time_steps) * tensor.features;
            batch.data.resize(static_cast<std::size_t>(end - begin) * sample_size);
            for (int i = begin; i < end; ++i) {
                std::copy_n(tensor.data.begin() + static_cast<std::size_t>(indices[i]) * sample_size,
                            sample_size, batch.data.begin() + static_cast<std::size_t>(i - begin) * sample_size);
            }
            return batch;
        }
    }

    void NeuralNetwork::fit_sequences(const layers::SequenceTensor& X, const layers::SequenceTensor& y) {
        fit_sequences(X, y, epochs_, batch_size_, verbose_);
    }

    void NeuralNetwork::fit_sequences(const layers::SequenceTensor& X, const layers::SequenceTensor& y,
                                      int epochs, int batch_size, bool verbose) {
        X.validate();
        y.validate();
        ML_CHECK_PARAM(X.is_sequence, "X", "must be a 3D sequence", "NeuralNetwork");
        ML_CHECK_PARAM(X.batch_size == y.batch_size, "y", "batch size must match X", "NeuralNetwork");
        ML_CHECK_PARAM(!y.is_sequence || X.time_steps == y.time_steps,
                       "y", "time steps must match X", "NeuralNetwork");
        ML_CHECK_PARAM(epochs > 0, "epochs", "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(batch_size > 0, "batch_size", "must be > 0", "NeuralNetwork");
        ML_CHECK_PARAM(!layers_.empty(), "layers", "add a recurrent layer before fitting sequences", "NeuralNetwork");
        if (fitted_ && X.features != n_features_) {
            ML_THROW_PARAMETER_ERROR("X", "feature count differs from the fitted network", "NeuralNetwork");
        }
        verbose_ = verbose;
        if (!fitted_) {
            build(X.features, y.features);
        } else if (y.features != n_classes_) {
            ML_THROW_PARAMETER_ERROR("y", "feature count differs from the fitted network", "NeuralNetwork");
        }
        initialize_optimizer_states();
        std::vector<int> indices(X.batch_size);
        std::iota(indices.begin(), indices.end(), 0);
        loss_history_.clear();
        loss_history_.reserve(epochs);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng_);
            double epoch_loss = 0.0;
            for (int begin = 0; begin < X.batch_size; begin += batch_size) {
                const int end = std::min(begin + batch_size, X.batch_size);
                const auto x_batch = sequence_batch(X, indices, begin, end);
                const auto y_batch = sequence_batch(y, indices, begin, end);
                const auto prediction = forward_sequences(x_batch, true);
                if (prediction.is_sequence != y_batch.is_sequence ||
                    prediction.time_steps != y_batch.time_steps ||
                    prediction.features != y_batch.features) {
                    ML_THROW_PARAMETER_ERROR("y", "shape must match the network output", "NeuralNetwork");
                }
                const Eigen::MatrixXd expected = sequence_matrix(y_batch);
                const Eigen::MatrixXd actual = sequence_matrix(prediction);
                epoch_loss += compute_loss(expected, actual) * (end - begin);
                const Eigen::MatrixXd loss_gradient = loss_function_->gradient(expected, actual);
                layers::SequenceTensor gradient = prediction;
                for (Eigen::Index row = 0; row < loss_gradient.rows(); ++row) {
                    for (Eigen::Index col = 0; col < loss_gradient.cols(); ++col) {
                        gradient.data[static_cast<std::size_t>(row) * loss_gradient.cols() + col] =
                            loss_gradient(row, col);
                    }
                }
                backward_sequences(gradient);
                update_parameters();
            }
            loss_history_.push_back(epoch_loss / X.batch_size);
            if (verbose && epoch % std::max(1, epochs / 10) == 0) {
                std::cout << "Epoch " << epoch << "/" << epochs
                          << " - Loss: " << loss_history_.back() << std::endl;
            }
        }
        fitted_ = true;
    }

    layers::SequenceTensor NeuralNetwork::forward_sequences(const layers::SequenceTensor& X,
                                                             bool training) const {
        X.validate();
        ML_CHECK_PARAM(X.is_sequence, "X", "must be a 3D sequence", "NeuralNetwork");
        ML_CHECK_PARAM(!layers_.empty(), "layers", "network must contain a recurrent layer", "NeuralNetwork");
        const layers::SequenceTensor* current = &X;
        layers::SequenceTensor next;
        for (const auto& layer : layers_) {
            auto* recurrent = dynamic_cast<layers::RecurrentLayer*>(layer.get());
            if (recurrent) {
                if (!current->is_sequence) {
                    ML_THROW_PARAMETER_ERROR("layers", "a recurrent layer requires sequence output from the previous layer", "NeuralNetwork");
                }
                next = recurrent->forward_sequence(*current, training);
            } else {
                if (current->is_sequence) {
                    ML_THROW_PARAMETER_ERROR("layers", "sequence output must be followed by a recurrent layer", "NeuralNetwork");
                }
                next = matrix_tensor(layer->forward(sequence_matrix(*current), training));
            }
            current = &next;
        }
        return next;
    }

    layers::SequenceTensor NeuralNetwork::backward_sequences(const layers::SequenceTensor& gradient) {
        layers::SequenceTensor current = gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            auto* recurrent = dynamic_cast<layers::RecurrentLayer*>(it->get());
            if (recurrent) {
                current = recurrent->backward_sequence(current);
            } else {
                if (current.is_sequence) {
                    ML_THROW_FITTING_ERROR("NeuralNetwork", "unexpected sequence gradient for a non-recurrent layer");
                }
                current = matrix_tensor((*it)->backward(sequence_matrix(current)));
            }
        }
        return current;
    }

    layers::SequenceTensor NeuralNetwork::predict_sequences(const layers::SequenceTensor& X) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        X.validate();
        ML_CHECK_FEATURE_DIMENSIONS(X.features, n_features_, "NeuralNetwork");
        return forward_sequences(X, false);
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
        layer_optimizers_.clear();
        
        // 3. Imposta le dimensioni della rete
        n_features_ = n_features;
        n_classes_ = n_classes;
        
        // 4. Propaga le dimensioni e inizializza i pesi
        int current_dim = n_features_;
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            // Imposta l'input shape per il layer
            layers_[i]->set_input_shape(current_dim);
            
            // Inizializza i pesi (se il layer li ha)
            layers_[i]->initialize_weights(rng_);
            
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

    void NeuralNetwork::initialize_optimizer_states() {
        if (!layer_optimizers_.empty()) {
            return;
        }
        if (!optimizer_) {
            ML_THROW_FITTING_ERROR("NeuralNetwork", "optimizer not set");
        }
        std::vector<std::unique_ptr<Optimizer>> optimizers;
        optimizers.reserve(layers_.size());
        for (const auto& layer : layers_) {
            if (layer->has_weights()) {
                auto optimizer = optimizer_->clone();
                optimizer->reset();
                optimizers.push_back(std::move(optimizer));
            } else {
                optimizers.push_back(nullptr);
            }
        }
        layer_optimizers_ = std::move(optimizers);
    }

    void NeuralNetwork::update_parameters() {
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto& layer = layers_[i];
            if (!layer->has_weights()) {
                continue;
            }
            auto& optimizer = layer_optimizers_[i];
            if (!optimizer) {
                ML_THROW_FITTING_ERROR("NeuralNetwork", "optimizer state missing for trainable layer");
            }
            Eigen::MatrixXd weights = layer->get_weights();
            Eigen::MatrixXd gradient = layer->get_weights_gradient();
            if (regularizer_) {
                gradient += regularizer_->compute_gradient(weights);
            }
            clip_gradient(gradient, 1.0);
            optimizer->update_weights(weights, gradient);
            layer->set_weights(weights);

            // BatchNorm packs beta into its weights matrix and updates it above.
            if (layer->get_use_bias() && layer->get_layer_type() != layers::LayerType::BATCH_NORM) {
                Eigen::VectorXd bias = layer->get_biases();
                optimizer->update_bias(bias, layer->get_bias_gradient());
                layer->set_biases(bias);
            }
        }
    }

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

    Eigen::MatrixXd NeuralNetwork::compute_loss_gradient(const Eigen::MatrixXd& y_true,
                                                          const Eigen::MatrixXd& y_pred) const {
        if (!loss_function_) ML_THROW_FITTING_ERROR("NeuralNetwork", "loss function not set");
        return loss_function_->gradient(y_true, y_pred);
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
        } else if (n_classes_ == 2 && output.cols() == 1) {
            // Classificazione binaria con singola colonna (Sigmoide)
            Eigen::VectorXd predictions(X.rows());
            for (int i = 0; i < X.rows(); ++i) {
                predictions(i) = (output(i, 0) >= 0.5) ? 1.0 : 0.0;
            }
            return predictions;
        } else {
            // Multiclasse (Softmax)
            Eigen::VectorXd predictions(X.rows());
            for (int i = 0; i < X.rows(); ++i) {
                Eigen::Index max_idx;
                output.row(i).maxCoeff(&max_idx);
                predictions(i) = static_cast<double>(max_idx);
            }
            return predictions;
        }
    }

    double NeuralNetwork::score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        ML_CHECK_XY_SIZE(X.rows(), y.size(), "NeuralNetwork");

        Eigen::VectorXd y_pred = predict(X);

        if (n_classes_ == 2) {
            int correct = 0;
            for (int i = 0; i < y.size(); ++i) {
                // Entrambi scalati a 0 o 1 corretto
                double p = y_pred(i) > 0.5 ? 1.0 : 0.0;
                double t = y(i) > 0.5 ? 1.0 : 0.0;
                if (p == t) correct++;
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

    Eigen::MatrixXd NeuralNetwork::predict_proba(const Eigen::MatrixXd& X) const {
        ML_CHECK_FITTED(fitted_, "NeuralNetwork");
        ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, "NeuralNetwork");
        return forward_pass(X, false);
    }

    
    //===========================================================================
    // SERIALIZZAZIONE
    //===========================================================================

    void NeuralNetwork::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) ML_THROW_IO_ERROR(filename, "save", "NeuralNetwork");
        serialize_binary(file);
        file.flush();
        if (!file) ML_THROW_IO_ERROR(filename, "save", "NeuralNetwork");
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
// SERIALIZZAZIONE (Sfruttando l'architettura LayerFactory)
//===========================================================================

    void NeuralNetwork::serialize_binary(std::ostream& out) const {
        if (n_features_ <= 0 || n_classes_ <= 0) {
            throw std::runtime_error("NeuralNetwork::serialize_binary: Network not initialized.");
        }

        utils::write_scalar(out, NETWORK_FORMAT_MARKER);
        utils::write_scalar(out, NETWORK_FORMAT_VERSION);

        // 1. Salva i metadati scalari fondamentali della rete
        out.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
        out.write(reinterpret_cast<const char*>(&n_classes_), sizeof(n_classes_));
        out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
        out.write(reinterpret_cast<const char*>(&fitted_), sizeof(fitted_));
        out.write(reinterpret_cast<const char*>(&epochs_), sizeof(epochs_));
        out.write(reinterpret_cast<const char*>(&batch_size_), sizeof(batch_size_));

        // 2. Salva la stringa del nome della Loss (Lunghezza + Caratteri)
        size_t loss_len = loss_function_name_.size();
        out.write(reinterpret_cast<const char*>(&loss_len), sizeof(loss_len));
        out.write(loss_function_name_.data(), loss_len);

        // 3. Salva il numero totale di layer presenti
        size_t num_layers = layers_.size();
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        // 4. Delega ogni singolo layer all'helper globale dell'architettura
        for (const auto& layer : layers_) {
            // Questa funzione scrive autonomamente: LayerType -> Versione -> Dati interni del Layer
            layers::serialize_layer(out, *layer);
        }

        if (!optimizer_) {
            ML_THROW_SERIALIZATION_ERROR("Optimizer not set", "NeuralNetwork");
        }
        write_optimizer(out, *optimizer_);
        const std::uint64_t optimizer_count = layer_optimizers_.size();
        if (optimizer_count != 0 && optimizer_count != layers_.size()) {
            ML_THROW_SERIALIZATION_ERROR("Invalid optimizer state count", "NeuralNetwork");
        }
        utils::write_scalar(out, optimizer_count);
        for (size_t i = 0; i < layer_optimizers_.size(); ++i) {
            const std::uint8_t present = layer_optimizers_[i] ? 1 : 0;
            if (static_cast<bool>(present) != layers_[i]->has_weights()) {
                ML_THROW_SERIALIZATION_ERROR("Optimizer state does not match layer", "NeuralNetwork");
            }
            utils::write_scalar(out, present);
            if (present) {
                write_optimizer(out, *layer_optimizers_[i]);
            }
        }
        const auto regularizer_type = regularizer_ ? regularizer_->get_type() : RegularizerType::NONE;
        const auto regularizer_tag = static_cast<std::uint8_t>(regularizer_type);
        utils::write_scalar(out, regularizer_tag);
        if (regularizer_) {
            regularizer_->serialize(out);
        }
        write_rng_state(out, rng_);
        if (!out) {
            ML_THROW_SERIALIZATION_ERROR("Failed to write network", "NeuralNetwork");
        }
    }

    void NeuralNetwork::deserialize_binary(std::istream& in) {
        layer_optimizers_.clear();
        rng_.seed(42);
        read_checked(in, n_features_);
        const bool versioned = n_features_ == NETWORK_FORMAT_MARKER;
        std::uint32_t version = 0;
        if (versioned) {
            read_checked(in, version);
            if (version == 0 || version > NETWORK_FORMAT_VERSION) {
                ML_THROW_DESERIALIZATION_ERROR("Unsupported network version", "NeuralNetwork");
            }
            read_checked(in, n_features_);
        }
        // 1. Leggi i metadati scalari
        read_checked(in, n_classes_);
        read_checked(in, learning_rate_);
        read_checked(in, fitted_);
        read_checked(in, epochs_);
        read_checked(in, batch_size_);

        // 2. Leggi il nome della Loss e ricostruiscila tramite la LossFactory
        size_t loss_len = 0;
        read_checked(in, loss_len);
        if (loss_len > 1024) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid loss name length", "NeuralNetwork");
        }
        loss_function_name_.resize(loss_len);
        if (!in.read(loss_function_name_.data(), loss_len)) {
            ML_THROW_DESERIALIZATION_ERROR("Truncated loss name", "NeuralNetwork");
        }
        loss_function_ = loss::LossFactory::create(loss_function_name_);

        // 4. Leggi il numero di layer, svuota la rete attuale e prepara la cache
        size_t num_layers = 0;
        read_checked(in, num_layers);
        layers_.clear();
        forward_cache_.resize(num_layers);

        // 5. Ricostruisci polimorficamente i layer usando l'helper globale dell'architettura
        for (size_t i = 0; i < num_layers; ++i) {
            // Questa funzione legge il tipo, interroga la LayerFactory per allocare il layer corretto,
            // chiama internamente la sua deserialize(in) e restituisce lo unique_ptr pronto.
            std::unique_ptr<layers::Layer> layer = layers::deserialize_layer(in);
            layers_.push_back(std::move(layer));
        }

        if (!versioned) {
            optimizer_ = OptimizerFactory::create(OptimizerType::SGD, learning_rate_);
            regularizer_.reset();
            return;
        }

        auto optimizer = read_optimizer(in, learning_rate_, 0, 0, 0);
        if (optimizer->get_learning_rate() != learning_rate_) {
            ML_THROW_DESERIALIZATION_ERROR("Inconsistent optimizer learning rate", "NeuralNetwork");
        }
        std::uint64_t optimizer_count = 0;
        read_checked(in, optimizer_count);
        if (optimizer_count != 0 && optimizer_count != layers_.size()) {
            ML_THROW_DESERIALIZATION_ERROR("Invalid optimizer state count", "NeuralNetwork");
        }
        std::vector<std::unique_ptr<Optimizer>> layer_optimizers;
        layer_optimizers.reserve(static_cast<size_t>(optimizer_count));
        for (size_t i = 0; i < optimizer_count; ++i) {
            std::uint8_t present = 0;
            read_checked(in, present);
            if (present > 1 || static_cast<bool>(present) != layers_[i]->has_weights()) {
                ML_THROW_DESERIALIZATION_ERROR("Optimizer state does not match layer", "NeuralNetwork");
            }
            if (!present) {
                layer_optimizers.push_back(nullptr);
                continue;
            }
            const Eigen::MatrixXd weights = layers_[i]->get_weights();
            const Eigen::Index bias_size = layers_[i]->get_use_bias()
                ? layers_[i]->get_biases().size() : 0;
            auto layer_optimizer = read_optimizer(in, learning_rate_, weights.rows(), weights.cols(), bias_size);
            if (layer_optimizer->get_type() != optimizer->get_type() ||
                layer_optimizer->get_learning_rate() != optimizer->get_learning_rate()) {
                ML_THROW_DESERIALIZATION_ERROR("Inconsistent optimizer settings", "NeuralNetwork");
            }
            layer_optimizers.push_back(std::move(layer_optimizer));
        }

        std::unique_ptr<Regularizer> regularizer;
        if (version >= 2) {
            std::uint8_t tag = 0;
            read_checked(in, tag);
            if (tag > static_cast<std::uint8_t>(RegularizerType::ELASTIC_NET)) {
                ML_THROW_DESERIALIZATION_ERROR("Unknown regularizer type", "NeuralNetwork");
            }
            const auto type = static_cast<RegularizerType>(tag);
            if (type != RegularizerType::NONE) {
                double strength = 0.0;
                read_checked(in, strength);
                if (type == RegularizerType::ELASTIC_NET) {
                    double ratio = 0.0;
                    read_checked(in, ratio);
                    if (ratio < 0.0 || ratio > 1.0) {
                        ML_THROW_DESERIALIZATION_ERROR("Invalid Elastic Net ratio", "NeuralNetwork");
                    }
                    regularizer = RegularizerFactory::create(type, strength, {{"l1_ratio", ratio}});
                } else {
                    regularizer = RegularizerFactory::create(type, strength);
                }
            }
        }
        if (version >= 4) {
            read_rng_state(in, rng_);
        }
        optimizer_ = std::move(optimizer);
        layer_optimizers_ = std::move(layer_optimizers);
        regularizer_ = std::move(regularizer);
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
        auto optimizer = OptimizerFactory::create(type, learning_rate);
        optimizer_ = std::move(optimizer);
        learning_rate_ = learning_rate;
        layer_optimizers_.clear();
    }

    void NeuralNetwork::set_learning_rate(double learning_rate) {
        Estimator::set_learning_rate(learning_rate);
        learning_rate_ = learning_rate;
        for (auto& optimizer : layer_optimizers_) {
            if (optimizer) {
                optimizer->set_learning_rate(learning_rate);
            }
        }
    }

    void NeuralNetwork::initialize_layers(int input_size) {
        if (layers_.empty()) {
            add_dense_layer(n_classes_, "softmax");
        }
    }

    void NeuralNetwork::reset() {
        layers_.clear();
        layer_optimizers_.clear();
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