#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>
#include <cstring>
#include <limits>

// Include dei tuoi header ufficiali
#include "models/neural_network.h"
#include "components/layers/layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/dropout_layer.h"

using namespace models;

class NeuralNetworkSerializationDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        filename_ = "test_debug_network.bin";
        nn_ = std::make_unique<NeuralNetwork>();
    }
    
    void TearDown() override {
        nn_.reset();
        std::remove(filename_.c_str());
    }
    
    std::string filename_;
    std::unique_ptr<NeuralNetwork> nn_;
};

TEST_F(NeuralNetworkSerializationDebugTest, FullBinaryDumpAndAnalysis) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DEBUG: ALLINEAMENTO HEADER REALE RETE" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "1. COSTRUZIONE RETE:" << std::endl;
    nn_->add_layer(std::make_unique<layers::DenseLayer>(128, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(64, "relu", true));
    nn_->add_layer(std::make_unique<layers::DropoutLayer>(0.5));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(32, "relu", true));
    nn_->add_layer(std::make_unique<layers::DenseLayer>(10, "softmax", false));
    
    // Configura parametri per riflettere i campi scritti nell'header
    nn_->build(128, 10);
    
    std::cout << "\n2. SERIALIZZAZIONE BINARIA CON METODO UFFICIALE:" << std::endl;
    std::ofstream out(filename_, std::ios::binary);
    ASSERT_TRUE(out.is_open());
    
    nn_->serialize_binary(out);
    out.close();
    
    std::cout << "   File salvato con successo." << std::endl;
    
    std::cout << "\n3. ANALISI PARSING SPERIMENTALE (Speculare al sorgente):" << std::endl;
    std::ifstream in(filename_, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    
    // Variabili speculari per catturare i campi esatti
    int n_features = 0;
    int n_classes = 0;
    double learning_rate = 0.0; // Cambia in float se nella classe usi float
    bool fitted = false;
    int epochs = 0;
    int batch_size = 0;
    
    int marker = 0;
    std::uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    ASSERT_LT(marker, 0);
    ASSERT_EQ(version, 1u);
    in.read(reinterpret_cast<char*>(&n_features), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_classes), sizeof(int));
    in.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    in.read(reinterpret_cast<char*>(&fitted), sizeof(bool));
    in.read(reinterpret_cast<char*>(&epochs), sizeof(int));
    in.read(reinterpret_cast<char*>(&batch_size), sizeof(int));
    
    std::cout << "   [HEADER SCALARE]" << std::endl;
    std::cout << "     n_features:    " << n_features << " (Atteso: 128)" << std::endl;
    std::cout << "     n_classes:     " << n_classes << " (Atteso: 10)" << std::endl;
    std::cout << "     learning_rate: " << learning_rate << std::endl;
    std::cout << "     fitted:        " << (fitted ? "true" : "false") << " (Atteso: true)" << std::endl;
    std::cout << "     epochs:        " << epochs << std::endl;
    std::cout << "     batch_size:    " << batch_size << std::endl;
    
    // Verifica allineamento stringa loss
    size_t loss_len = 0;
    in.read(reinterpret_cast<char*>(&loss_len), sizeof(size_t));
    std::cout << "     loss_len:      " << loss_len << " bytes" << std::endl;
    
    // Evitiamo crash se la lettura iniziale è corrotta
    ASSERT_LT(loss_len, 1000) << "Errore critico: la lunghezza del nome della loss è assurda. Disallineamento tipi nell'header.";
    
    std::string loss_name;
    loss_name.resize(loss_len);
    if (loss_len > 0) {
        in.read(&loss_name[0], loss_len);
    }
    std::cout << "     loss_name:     \"" << loss_name << "\"" << std::endl;

    size_t num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    EXPECT_EQ(n_features, 128);
    EXPECT_EQ(n_classes, 10);
    EXPECT_EQ(num_layers, 5);
    in.close();

    NeuralNetwork loaded_nn;
    std::ifstream in_official(filename_, std::ios::binary);
    ASSERT_TRUE(in_official.is_open());
    EXPECT_NO_THROW(loaded_nn.deserialize_binary(in_official));
}

TEST_F(NeuralNetworkSerializationDebugTest, RestoresOptimizerStateForEachLayer) {
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    for (auto type : {OptimizerType::SGD, OptimizerType::MOMENTUM, OptimizerType::ADAM,
                      OptimizerType::RMSPROP, OptimizerType::ADAGRAD}) {
        NeuralNetwork original({1, 2, 1}, "linear", "linear", type, 0.01);
        original.set_loss_function("mse");
        original.fit(X, y, 4, 1, false);
        original.save(filename_);

        NeuralNetwork restored;
        restored.load(filename_);
        ASSERT_TRUE(restored.is_fitted());
        original.fit(X, y, 3, 1, false);
        restored.fit(X, y, 3, 1, false);

        ASSERT_EQ(original.get_layers().size(), restored.get_layers().size());
        for (size_t i = 0; i < original.get_layers().size(); ++i) {
            EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
                restored.get_layers()[i]->get_weights(), 1e-12))
                << "optimizer " << static_cast<int>(type) << ", layer " << i;
            EXPECT_TRUE(original.get_layers()[i]->get_biases().isApprox(
                restored.get_layers()[i]->get_biases(), 1e-12))
                << "optimizer " << static_cast<int>(type) << ", layer " << i;
        }
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, PreservesOptimizerBeforeTraining) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.set_loss_function("mse");
    original.build(1, 1);
    original.save(filename_);
    NeuralNetwork restored;
    restored.load(filename_);
    restored.build(1, 1);

    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 1, 1, false);
    restored.fit(X, y, 1, 1, false);
    for (size_t i = 0; i < original.get_layers().size(); ++i) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, RestoresStateWithUnweightedAndBiaslessLayers) {
    NeuralNetwork original({1, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.reset();
    original.add_dense_layer(1, "linear", false);
    original.add_dropout_layer(0.0);
    original.add_dense_layer(1, "linear", false);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 4, 1, false);
    original.save(filename_);

    NeuralNetwork restored;
    restored.load(filename_);
    original.fit(X, y, 2, 1, false);
    restored.fit(X, y, 2, 1, false);
    for (int i : {0, 2}) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, PreservesOptimizerChangedAfterTraining) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::SGD, 0.01);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 2, 1, false);
    original.set_optimizer(OptimizerType::ADAM, 0.02);
    original.save(filename_);
    NeuralNetwork restored;
    restored.load(filename_);
    original.fit(X, y, 2, 1, false);
    restored.fit(X, y, 2, 1, false);
    for (size_t i = 0; i < original.get_layers().size(); ++i) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, RestoresOptimizerForRecurrentLayers) {
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 2, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    for (auto type : {RecurrentType::SIMPLE_RNN, RecurrentType::LSTM, RecurrentType::GRU}) {
        NeuralNetwork original({2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
        original.reset();
        original.add_recurrent_layer(type, 2);
        original.add_dense_layer(1, "linear");
        original.set_loss_function("mse");
        original.fit(X, y, 3, 1, false);
        original.save(filename_);
        NeuralNetwork restored;
        restored.load(filename_);

        auto* original_recurrent = dynamic_cast<layers::RecurrentLayer*>(original.get_layers()[0].get());
        auto* restored_recurrent = dynamic_cast<layers::RecurrentLayer*>(restored.get_layers()[0].get());
        ASSERT_NE(original_recurrent, nullptr);
        ASSERT_NE(restored_recurrent, nullptr);
        original_recurrent->reset_state();
        restored_recurrent->reset_state();
        original.fit(X, y, 1, 1, false);
        restored.fit(X, y, 1, 1, false);
        for (size_t i = 0; i < original.get_layers().size(); ++i) {
            EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
                restored.get_layers()[i]->get_weights(), 1e-12))
                << "recurrent type " << static_cast<int>(type) << ", layer " << i;
        }
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, RestoresOptimizerForConvolutionalLayer) {
    NeuralNetwork original({36, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.reset();
    auto conv = std::make_unique<layers::Conv2DLayer>(1, 3, 1, "valid", "linear");
    conv->set_input_shape(36);
    original.add_layer(std::move(conv));
    original.add_flatten_layer();
    original.add_dense_layer(1, "linear");
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 36, 0.1);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 3, 1, false);
    original.save(filename_);
    NeuralNetwork restored;
    restored.load(filename_);
    original.fit(X, y, 1, 1, false);
    restored.fit(X, y, 1, 1, false);
    for (int i : {0, 2}) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, RestoresOptimizerThroughBinarySerializer) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 3, 1, false);
    utils::BinarySerializer<NeuralNetwork>::save(original, filename_);
    NeuralNetwork restored;
    utils::BinarySerializer<NeuralNetwork>::load(restored, filename_);
    original.fit(X, y, 1, 1, false);
    restored.fit(X, y, 1, 1, false);
    for (size_t i = 0; i < original.get_layers().size(); ++i) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, ReadsLegacyNetworkWithoutConsumingNextRecord) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 2, 1, false);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    const int features = 1, classes = 1, epochs = 100, batch_size = 32;
    const double learning_rate = 0.01;
    const bool fitted = true;
    const std::string loss = "mse";
    const size_t loss_size = loss.size(), layer_count = original.get_layers().size();
    utils::write_scalar(stream, features);
    utils::write_scalar(stream, classes);
    utils::write_scalar(stream, learning_rate);
    utils::write_scalar(stream, fitted);
    utils::write_scalar(stream, epochs);
    utils::write_scalar(stream, batch_size);
    utils::write_scalar(stream, loss_size);
    stream.write(loss.data(), loss.size());
    utils::write_scalar(stream, layer_count);
    for (const auto& layer : original.get_layers()) {
        layers::serialize_layer(stream, *layer);
    }
    const std::uint32_t next_record = 0x12345678;
    utils::write_scalar(stream, next_record);
    stream.seekg(0);

    NeuralNetwork restored;
    restored.deserialize_binary(stream);
    std::uint32_t actual = 0;
    utils::read_scalar(stream, actual);
    EXPECT_EQ(actual, next_record);
    EXPECT_TRUE(original.predict_proba(X).isApprox(restored.predict_proba(X)));
    EXPECT_DOUBLE_EQ(restored.get_learning_rate(), learning_rate);

    {
        std::ofstream file(filename_, std::ios::binary);
        const std::string legacy_data = stream.str();
        file.write(legacy_data.data(), legacy_data.size());
    }
    NeuralNetwork restored_file;
    restored_file.load(filename_);
    EXPECT_TRUE(original.predict_proba(X).isApprox(restored_file.predict_proba(X)));

    {
        std::ofstream file(filename_, std::ios::binary);
        utils::write_scalar(file, utils::SERIALIZATION_MAGIC);
        utils::write_scalar(file, utils::SERIALIZATION_VERSION);
        const std::string legacy_data = stream.str();
        file.write(legacy_data.data(), legacy_data.size());
    }
    NeuralNetwork restored_wrapped;
    utils::BinarySerializer<NeuralNetwork>::load(restored_wrapped, filename_);
    EXPECT_TRUE(original.predict_proba(X).isApprox(restored_wrapped.predict_proba(X)));

    original.set_optimizer(OptimizerType::SGD, learning_rate);
    original.fit(X, y, 1, 1, false);
    restored.fit(X, y, 1, 1, false);
    for (size_t i = 0; i < original.get_layers().size(); ++i) {
        EXPECT_TRUE(original.get_layers()[i]->get_weights().isApprox(
            restored.get_layers()[i]->get_weights(), 1e-12));
    }
}

TEST_F(NeuralNetworkSerializationDebugTest, RejectsTruncatedOptimizerState) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 2, 1, false);
    std::ostringstream out(std::ios::binary);
    original.serialize_binary(out);
    std::string data = out.str();
    ASSERT_GT(data.size(), 1u);
    data.pop_back();
    std::istringstream in(data, std::ios::binary);
    NeuralNetwork restored;
    EXPECT_THROW(restored.deserialize_binary(in), ml_exception::MLException);
}

TEST_F(NeuralNetworkSerializationDebugTest, RejectsUnsupportedNetworkVersion) {
    NeuralNetwork original({1, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.build(1, 1);
    std::ostringstream out(std::ios::binary);
    original.serialize_binary(out);
    std::string data = out.str();
    const std::uint32_t unsupported_version = 999;
    std::memcpy(&data[sizeof(int)], &unsupported_version, sizeof(unsupported_version));
    std::istringstream in(data, std::ios::binary);
    NeuralNetwork restored;
    EXPECT_THROW(restored.deserialize_binary(in), ml_exception::MLException);
}

TEST_F(NeuralNetworkSerializationDebugTest, RejectsInvalidOptimizerDimensions) {
    NeuralNetwork original({1, 2, 1}, "linear", "linear", OptimizerType::ADAM, 0.01);
    original.set_loss_function("mse");
    const Eigen::MatrixXd X = Eigen::MatrixXd::Constant(1, 1, 0.5);
    const Eigen::MatrixXd y = Eigen::MatrixXd::Constant(1, 1, 0.25);
    original.fit(X, y, 2, 1, false);
    std::ostringstream out(std::ios::binary);
    original.serialize_binary(out);
    std::string data = out.str();
    std::istringstream parser(data, std::ios::binary);
    parser.seekg(sizeof(int) + sizeof(std::uint32_t) + 2 * sizeof(int) + sizeof(double)
                 + sizeof(bool) + 2 * sizeof(int));
    size_t loss_size = 0, layer_count = 0;
    utils::read_scalar(parser, loss_size);
    parser.seekg(loss_size, std::ios::cur);
    utils::read_scalar(parser, layer_count);
    ASSERT_EQ(layer_count, 2u);
    for (size_t i = 0; i < layer_count; ++i) {
        layers::deserialize_layer(parser);
    }
    std::uint8_t type = 0, present = 0;
    std::uint64_t payload_size = 0, optimizer_count = 0;
    utils::read_scalar(parser, type);
    utils::read_scalar(parser, payload_size);
    parser.seekg(payload_size, std::ios::cur);
    utils::read_scalar(parser, optimizer_count);
    utils::read_scalar(parser, present);
    utils::read_scalar(parser, type);
    utils::read_scalar(parser, payload_size);
    ASSERT_EQ(optimizer_count, 2u);
    ASSERT_EQ(present, 1u);
    ASSERT_EQ(type, static_cast<std::uint8_t>(OptimizerType::ADAM));
    parser.seekg(5 * sizeof(double) + sizeof(int), std::ios::cur);
    ASSERT_TRUE(parser.good());
    const auto offset = static_cast<std::size_t>(parser.tellg());
    ASSERT_LE(offset + sizeof(int), data.size());
    const int invalid_rows = std::numeric_limits<int>::max();
    std::memcpy(&data[offset], &invalid_rows, sizeof(invalid_rows));

    std::istringstream in(data, std::ios::binary);
    NeuralNetwork restored;
    EXPECT_THROW(restored.deserialize_binary(in), ml_exception::MLException);
}