#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "models/linear_regression.h"
#include "models/logistic_regression.h"
#include "models/neural_network.h"

using namespace models;

struct MatrixData {
    Eigen::MatrixXd x;
    Eigen::VectorXd regression;
    Eigen::VectorXd classification;
};

MatrixData make_matrix(int samples, int features, unsigned seed) {
    MatrixData data{Eigen::MatrixXd(samples, features), Eigen::VectorXd(samples),
                    Eigen::VectorXd(samples)};
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < features; ++j) data.x(i, j) = normal(generator);
        double signal = data.x(i, 0) + 0.5 * data.x(i, 1) * data.x(i, 2);
        data.regression(i) = signal + 0.1 * normal(generator);
        data.classification(i) = signal + 0.3 * normal(generator) > 0.0 ? 1.0 : 0.0;
    }
    return data;
}

struct SequenceData {
    layers::SequenceTensor x;
    layers::SequenceTensor y;
};

SequenceData make_sequences(int count, unsigned seed) {
    constexpr int steps = 8;
    constexpr int features = 2;
    SequenceData result{{{}, count, steps, features, true}, {{}, count, 1, 1, false}};
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < count; ++i) {
        double target = 0.0;
        for (int t = 0; t < steps; ++t) {
            double a = normal(generator);
            double b = normal(generator);
            result.x.data.push_back(a);
            result.x.data.push_back(b);
            target += (a + 0.5 * b) / steps;
        }
        result.y.data.push_back(target);
    }
    return result;
}

template <typename Predict>
void measure(const std::string& model, int batch_size, int calls, Predict predict) {
    double checksum = 0.0;
    for (int i = 0; i < 5; ++i) checksum += predict();
    for (int run = 1; run <= 5; ++run) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < calls; ++i) checksum += predict();
        double elapsed_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        double per_item_us = elapsed_us / (batch_size * calls);
        if (!std::isfinite(checksum) || !std::isfinite(per_item_us) || per_item_us <= 0.0) {
            throw std::runtime_error("Invalid inference measurement for " + model);
        }
        std::cout << model << ',' << batch_size << ',' << calls << ',' << run << ','
                  << per_item_us << ',' << 1e6 / per_item_us << ',' << checksum << '\n';
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(9)
              << "model,batch_size,calls,run,us_per_item,items_per_second,checksum\n";
    const auto train50 = make_matrix(8000, 50, 42);
    const auto test50 = make_matrix(1024, 50, 43);
    LinearRegression ridge(0.01, 1000, 0.1, LinearRegression::GRADIENT_DESCENT);
    ridge.fit(train50.x, train50.regression);
    measure("Ridge", 1024, 50, [&]() { return ridge.predict(test50.x)(0); });

    const auto train10 = make_matrix(8000, 10, 42);
    const auto test10 = make_matrix(1024, 10, 43);
    LogisticRegression logistic(0.01, 1000, 0.001, 1e-4, false);
    logistic.fit(train10.x, train10.classification);
    measure("Logistic", 1024, 50, [&]() { return logistic.predict(test10.x)(0); });

    NeuralNetwork mlp({10, 32, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.01);
    mlp.set_loss_function("binary_crossentropy");
    mlp.fit(train10.x, train10.classification, 100, 32, false);
    measure("MLP", 1024, 50, [&]() { return mlp.predict_proba(test10.x)(0, 0); });

    const auto train_seq = make_sequences(256, 42);
    const auto test_seq = make_sequences(64, 43);
    for (const auto& model : std::vector<std::pair<std::string, RecurrentType>>{
             {"SimpleRNN", RecurrentType::SIMPLE_RNN}, {"LSTM", RecurrentType::LSTM},
             {"GRU", RecurrentType::GRU}}) {
        NeuralNetwork network;
        network.add_recurrent_layer(model.second, 8, false);
        network.add_dense_layer(1, "linear");
        network.set_optimizer(OptimizerType::ADAM, 0.01);
        network.set_loss_function("mse");
        network.fit_sequences(train_seq.x, train_seq.y, 20, 32, false);
        measure(model.first, 64, 50, [&]() {
            auto prediction = network.predict_sequences(test_seq.x);
            return prediction.data.at(0);
        });
    }
}
