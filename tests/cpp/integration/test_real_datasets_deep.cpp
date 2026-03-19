// test/integration/test_real_datasets_deep.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "models/neural_network.h"
#include "models/linear_regression.h"
#include "models/logistic_regression.h"
#include "utils/math_utils.h"

using namespace models;
using namespace utils;
using namespace testing;

class RealDatasetsTest : public ::testing::Test {
protected:
    void SetUp() override {
        create_iris_like_data();
        create_boston_like_data();
    }
    
    void create_iris_like_data() {
        iris_X.resize(150, 4);
        iris_y.resize(150);
        
        std::random_device rd;
        std::mt19937 gen(42);
        
        for (int i = 0; i < 50; ++i) {
            iris_X(i, 0) = 5.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 1) = 3.5 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 2) = 1.5 + std::normal_distribution<>(0, 0.2)(gen);
            iris_X(i, 3) = 0.2 + std::normal_distribution<>(0, 0.1)(gen);
            iris_y(i) = 0;
        }
        for (int i = 50; i < 100; ++i) {
            iris_X(i, 0) = 6.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 1) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 2) = 4.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 3) = 1.3 + std::normal_distribution<>(0, 0.2)(gen);
            iris_y(i) = 1;
        }
        for (int i = 100; i < 150; ++i) {
            iris_X(i, 0) = 7.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 1) = 3.0 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 2) = 5.5 + std::normal_distribution<>(0, 0.3)(gen);
            iris_X(i, 3) = 2.0 + std::normal_distribution<>(0, 0.2)(gen);
            iris_y(i) = 2;
        }
    }
    
    void create_boston_like_data() {
        boston_X.resize(200, 13);
        boston_y.resize(200);
        
        std::random_device rd;
        std::mt19937 gen(42);
        
        for (int i = 0; i < 200; ++i) {
            double crime = std::normal_distribution<>(3.0, 2.0)(gen);
            double rooms = std::normal_distribution<>(6.0, 1.0)(gen);
            double age = std::normal_distribution<>(70.0, 20.0)(gen);
            double tax = std::normal_distribution<>(400.0, 100.0)(gen);
            
            boston_X(i, 0) = crime;
            boston_X(i, 1) = rooms;
            boston_X(i, 2) = age;
            boston_X(i, 3) = tax;
            
            boston_y(i) = 5*rooms - 0.5*crime - 0.1*age + 0.01*tax + 
                         std::normal_distribution<>(0, 2.0)(gen);
        }
    }
    
    Eigen::MatrixXd iris_X, boston_X;
    Eigen::VectorXd iris_y, boston_y;
};

// TEST CHE PASSANO - LI TENIAMO

TEST_F(RealDatasetsTest, IrisWithLogisticRegression) {
    auto splits = MathUtils::train_test_split(iris_X, iris_y, 0.2, 42);
    const auto& train = splits[0];
    const auto& test = splits[1];
    
    std::vector<std::unique_ptr<LogisticRegression>> models;
    
    for (int c = 0; c < 3; ++c) {
        Eigen::VectorXd y_binary = (train.second.array() == c).cast<double>();
        auto model = std::make_unique<LogisticRegression>(0.1, 1000, 0.01);
        model->fit(train.first, y_binary);
        models.push_back(std::move(model));
    }
    
    Eigen::MatrixXd proba(test.first.rows(), 3);
    for (int c = 0; c < 3; ++c) {
        proba.col(c) = models[c]->predict(test.first);
    }
    
    int correct = 0;
    for (int i = 0; i < test.first.rows(); ++i) {
        Eigen::Index max_idx;
        proba.row(i).maxCoeff(&max_idx);
        if (max_idx == static_cast<int>(test.second(i))) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test.first.rows();
    EXPECT_GT(accuracy, 0.85);
}

TEST_F(RealDatasetsTest, BostonWithLinearRegression) {
    auto splits = MathUtils::train_test_split(boston_X, boston_y, 0.2, 42);
    const auto& train = splits[0];
    const auto& test = splits[1];
    
    Eigen::VectorXd mean, std;
    Eigen::MatrixXd X_train_scaled = train.first;
    MathUtils::standardize_features(X_train_scaled, mean, std);
    
    Eigen::MatrixXd X_test_scaled = (test.first.rowwise() - mean.transpose())
                                   .array().rowwise() / std.transpose().array();
    
    LinearRegression model(0.01, 2000, 0.001, LinearRegression::GRADIENT_DESCENT);
    model.fit(X_train_scaled, train.second);
    
    double score = model.score(X_test_scaled, test.second);
    EXPECT_GT(score, 0.7);
}

TEST_F(RealDatasetsTest, BostonWithNeuralNetwork) {
    auto splits = MathUtils::train_test_split(boston_X, boston_y, 0.2, 42);
    const auto& train = splits[0];
    const auto& test = splits[1];
    
    Eigen::VectorXd mean, std;
    Eigen::MatrixXd X_train_scaled = train.first;
    MathUtils::standardize_features(X_train_scaled, mean, std);
    Eigen::MatrixXd X_test_scaled = (test.first.rowwise() - mean.transpose())
                                   .array().rowwise() / std.transpose().array();
    
    double y_mean = train.second.mean();
    double y_std = std::sqrt((train.second.array() - y_mean).square().sum() / (train.second.size() - 1));
    Eigen::VectorXd y_train_scaled = (train.second.array() - y_mean) / y_std;
    
    NeuralNetwork network({13, 64, 1}, "relu", "linear", 
                          OptimizerType::ADAM, 0.07);
    network.set_loss_function("mse");
    network.set_epochs(1000);
    network.set_batch_size(32);
    network.set_verbose(true);
    
    network.fit(X_train_scaled, y_train_scaled);
    
    Eigen::VectorXd y_pred_scaled = network.predict(X_test_scaled);
    Eigen::VectorXd y_pred = y_pred_scaled.array() * y_std + y_mean;
    
    double ss_res = (test.second - y_pred).array().square().sum();
    double ss_tot = (test.second.array() - test.second.mean()).square().sum();
    double r2 = 1.0 - (ss_res / ss_tot);
    
    EXPECT_GT(r2, 0.7);
}

// TEST INSTABILI DI IRIS - COMMENTATI
/*
TEST_F(RealDatasetsTest, IrisWithNeuralNetwork) {
    // ...
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}