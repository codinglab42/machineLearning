#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "models/neural_network.h"

using namespace models;
using namespace std::chrono;

struct SequenceData {
    layers::SequenceTensor x;
    layers::SequenceTensor y;
};

SequenceData make_sequences(int count, unsigned seed) {
    constexpr int steps = 8;
    constexpr int features = 2;
    SequenceData result{{{}, count, steps, features, true}, {{}, count, 1, 1, false}};
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < count; ++i) {
        double target = 0.0;
        for (int t = 0; t < steps; ++t) {
            double a = normal(gen);
            double b = normal(gen);
            result.x.data.push_back(a);
            result.x.data.push_back(b);
            target += (a + 0.5 * b) / steps;
        }
        result.y.data.push_back(target);
    }
    return result;
}

int main() {
    const auto train = make_sequences(256, 42);
    const auto test = make_sequences(64, 43);
    const std::vector<std::pair<std::string, RecurrentType>> types = {
        {"SimpleRNN", RecurrentType::SIMPLE_RNN},
        {"LSTM", RecurrentType::LSTM}, {"GRU", RecurrentType::GRU}
    };
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nSEQUENCE - Recurrent training\n" << std::string(80, '-') << '\n'
              << "Type Time_ms Test_MSE Zero_baseline_MSE\n";
    for (const auto& type : types) {
        NeuralNetwork network;
        network.add_recurrent_layer(type.second, 8, false);
        network.add_dense_layer(1, "linear");
        network.set_optimizer(OptimizerType::ADAM, 0.01);
        network.set_loss_function("mse");
        auto start = steady_clock::now();
        network.fit_sequences(train.x, train.y, 20, 32, false);
        double ms = duration<double, std::milli>(steady_clock::now() - start).count();
        auto prediction = network.predict_sequences(test.x);
        if (prediction.data.size() != test.y.data.size()) {
            throw std::runtime_error("Unexpected sequence prediction size");
        }
        double mse = 0.0;
        double baseline = 0.0;
        for (std::size_t i = 0; i < prediction.data.size(); ++i) {
            mse += std::pow(prediction.data[i] - test.y.data[i], 2);
            baseline += std::pow(test.y.data[i], 2);
        }
        mse /= prediction.data.size();
        baseline /= prediction.data.size();
        if (!std::isfinite(mse)) throw std::runtime_error("Non-finite sequence test MSE");
        std::cout << type.first << ' ' << ms << ' ' << mse << ' ' << baseline << '\n';
    }

    NeuralNetwork original;
    original.add_recurrent_layer(RecurrentType::GRU, 8, false);
    original.add_dense_layer(1, "linear");
    original.set_optimizer(OptimizerType::ADAM, 0.01);
    original.set_regularizer(RegularizerType::L2, 0.001);
    original.set_loss_function("mse");
    original.fit_sequences(train.x, train.y, 5, 32, false);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto start = steady_clock::now();
    original.serialize_binary(stream);
    double save_ms = duration<double, std::milli>(steady_clock::now() - start).count();
    const auto bytes = stream.str().size();
    stream.seekg(0);
    NeuralNetwork restored;
    start = steady_clock::now();
    restored.deserialize_binary(stream);
    double load_ms = duration<double, std::milli>(steady_clock::now() - start).count();
    if (original.predict_sequences(test.x).data != restored.predict_sequences(test.x).data) {
        throw std::runtime_error("Checkpoint prediction mismatch");
    }
    original.fit_sequences(train.x, train.y, 1, 32, false);
    restored.fit_sequences(train.x, train.y, 1, 32, false);
    if (original.predict_sequences(test.x).data != restored.predict_sequences(test.x).data) {
        throw std::runtime_error("Checkpoint continuation mismatch");
    }
    std::cout << "\nSEQUENCE - Checkpoint in memory\n" << std::string(80, '-') << '\n'
              << "Operation Time_ms Bytes\n"
              << "Save " << save_ms << ' ' << bytes << '\n'
              << "Load " << load_ms << ' ' << bytes << '\n'
              << "\nBenchmark completato!\n";
}
