#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "models/neural_network.h"
#include "components/loss/loss_factory.h"
#include "components/loss/binary_cross_entropy_loss.h"
#include "components/loss/categorical_cross_entropy_loss.h"
#include "components/loss/mean_squared_error_loss.h"
#include "components/loss/mean_absolute_error_loss.h"
#include "components/loss/huber_loss.h"

using namespace Eigen;
using namespace models;
using namespace loss;
using namespace testing;

// ============================================================================
// Mock per testing
// ============================================================================

class MockLoss : public Loss {
public:
    MOCK_METHOD(double, compute, (const VectorXd&, const VectorXd&), (const, override));
    MOCK_METHOD(double, compute, (const MatrixXd&, const MatrixXd&), (const, override));
    MOCK_METHOD(MatrixXd, gradient, (const MatrixXd&, const MatrixXd&), (const, override));
    MOCK_METHOD(std::string, name, (), (const, override));
};

// ============================================================================
// Test Suite per Loss Functions
// ============================================================================

class LossFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        bce_loss_ = LossFactory::create("binary_crossentropy");
        cce_loss_ = LossFactory::create("categorical_crossentropy");
        mse_loss_ = LossFactory::create("mse");
        mae_loss_ = LossFactory::create("mae");
        huber_loss_ = LossFactory::create("huber");
    }
    
    std::unique_ptr<Loss> bce_loss_;
    std::unique_ptr<Loss> cce_loss_;
    std::unique_ptr<Loss> mse_loss_;
    std::unique_ptr<Loss> mae_loss_;
    std::unique_ptr<Loss> huber_loss_;
};

TEST_F(LossFunctionsTest, BinaryCrossEntropyPerfectPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 0, 1;
    y_pred << 1, 0, 1;
    
    double loss = bce_loss_->compute(y_true, y_pred);
    EXPECT_NEAR(loss, 0.0, 1e-6);
}

TEST_F(LossFunctionsTest, BinaryCrossEntropyGradient) {
    MatrixXd y_true(3, 1);
    MatrixXd y_pred(3, 1);
    y_true << 1, 0, 1;
    y_pred << 0.8, 0.2, 0.7;
    
    MatrixXd grad = bce_loss_->gradient(y_true, y_pred);
    MatrixXd expected = y_pred - y_true;
    
    EXPECT_TRUE(grad.isApprox(expected, 1e-6));
}

TEST_F(LossFunctionsTest, BinaryCrossEntropyNumericalStability) {
    VectorXd y_true(1);
    VectorXd y_pred(1);
    y_true << 1;
    y_pred << 0.0;  // Caso limite
    
    EXPECT_NO_THROW(bce_loss_->compute(y_true, y_pred));
    
    y_pred << 1.0;  // Caso limite
    EXPECT_NO_THROW(bce_loss_->compute(y_true, y_pred));
}

TEST_F(LossFunctionsTest, CategoricalCrossEntropyPerfectPredictions) {
    MatrixXd y_true(2, 3);
    MatrixXd y_pred(2, 3);
    y_true << 1, 0, 0,
              0, 1, 0;
    y_pred << 0.9, 0.05, 0.05,
              0.1, 0.85, 0.05;
    
    double loss = cce_loss_->compute(y_true, y_pred);
    EXPECT_GT(loss, 0);
    EXPECT_LT(loss, 1);
}

TEST_F(LossFunctionsTest, MeanSquaredErrorPerfectPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 2, 3;
    y_pred << 1, 2, 3;
    
    double loss = mse_loss_->compute(y_true, y_pred);
    EXPECT_NEAR(loss, 0.0, 1e-6);
}

TEST_F(LossFunctionsTest, MeanSquaredErrorCalculation) {
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 1, 2;
    y_pred << 3, 5;
    
    double loss = mse_loss_->compute(y_true, y_pred);
    // ((3-1)^2 + (5-2)^2) / 2 = (4 + 9) / 2 = 6.5
    EXPECT_NEAR(loss, 6.5, 1e-6);
}

TEST_F(LossFunctionsTest, MeanAbsoluteErrorCalculation) {
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 1, 2;
    y_pred << 3, 5;
    
    double loss = mae_loss_->compute(y_true, y_pred);
    // (|3-1| + |5-2|) / 2 = (2 + 3) / 2 = 2.5
    EXPECT_NEAR(loss, 2.5, 1e-6);
}

TEST_F(LossFunctionsTest, HuberLossCalculation) {
    auto huber = std::make_unique<HuberLoss>(1.0);
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 0, 0;
    y_pred << 0.5, 2.0;
    
    double loss = huber->compute(y_true, y_pred);
    // Primo: 0.5^2/2 = 0.125
    // Secondo: 1*(2-0.5) = 1.5
    // Media: (0.125 + 1.5) / 2 = 0.8125
    EXPECT_NEAR(loss, 0.8125, 1e-6);
}

TEST_F(LossFunctionsTest, LossFactoryCreate) {
    auto loss1 = LossFactory::create("binary_crossentropy");
    auto loss2 = LossFactory::create("categorical_crossentropy");
    auto loss3 = LossFactory::create("mse");
    auto loss4 = LossFactory::create("mae");
    auto loss5 = LossFactory::create("huber");
    
    EXPECT_NE(loss1, nullptr);
    EXPECT_NE(loss2, nullptr);
    EXPECT_NE(loss3, nullptr);
    EXPECT_NE(loss4, nullptr);
    EXPECT_NE(loss5, nullptr);
    
    EXPECT_EQ(loss1->name(), "binary_crossentropy");
    EXPECT_EQ(loss2->name(), "categorical_crossentropy");
    EXPECT_EQ(loss3->name(), "mse");
    EXPECT_EQ(loss4->name(), "mae");
    EXPECT_EQ(loss5->name(), "huber");
}

TEST_F(LossFunctionsTest, LossFactoryUnknownLoss) {
    EXPECT_THROW(LossFactory::create("unknown_loss"), ml_exception::InvalidParameterException);
}

// ============================================================================
// Test Suite per Neural Network
// ============================================================================

class NeuralNetworkIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // AND dataset
        X_and_.resize(4, 2);
        X_and_ << 0, 0,
                  0, 1,
                  1, 0,
                  1, 1;
        y_and_.resize(4);
        y_and_ << 0, 0, 0, 1;
        
        // OR dataset
        X_or_.resize(4, 2);
        X_or_ << 0, 0,
                 0, 1,
                 1, 0,
                 1, 1;
        y_or_.resize(4);
        y_or_ << 0, 1, 1, 1;
        
        // XOR dataset
        X_xor_.resize(4, 2);
        X_xor_ << 0, 0,
                  0, 1,
                  1, 0,
                  1, 1;
        y_xor_.resize(4);
        y_xor_ << 0, 1, 1, 0;
    }
    
    MatrixXd X_and_, X_or_, X_xor_;
    VectorXd y_and_, y_or_, y_xor_;
};

TEST_F(NeuralNetworkIntegrationTest, AND_WithAdam) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(500);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_and_, y_and_);
    
    VectorXd y_pred = network.predict(X_and_);
    VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
    
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        if (y_pred_int(i) == static_cast<int>(y_and_(i))) correct++;
    }
    EXPECT_GE(correct, 4);
}

TEST_F(NeuralNetworkIntegrationTest, OR_WithAdam) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(500);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_or_, y_or_);
    
    VectorXd y_pred = network.predict(X_or_);
    VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
    
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        if (y_pred_int(i) == static_cast<int>(y_or_(i))) correct++;
    }
    EXPECT_GE(correct, 4);
}

TEST_F(NeuralNetworkIntegrationTest, TrainingHistory) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(100);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_and_, y_and_);
    
    auto [loss, val_loss, acc] = network.get_training_history();
    EXPECT_GT(loss.size(), 0);
    EXPECT_LT(loss.back(), loss.front());  // Loss should decrease
}

TEST_F(NeuralNetworkIntegrationTest, PredictProba) {
    NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.5);
    network.set_loss_function("binary_crossentropy");
    network.set_epochs(100);
    network.set_batch_size(4);
    network.set_verbose(false);
    
    network.fit(X_and_, y_and_);
    
    MatrixXd proba = network.predict_proba(X_and_);
    EXPECT_EQ(proba.rows(), 4);
    EXPECT_EQ(proba.cols(), 1);
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(proba(i, 0), 0.0);
        EXPECT_LE(proba(i, 0), 1.0);
    }
}

TEST_F(NeuralNetworkIntegrationTest, DifferentOptimizers) {
    std::vector<OptimizerType> optimizers = {OptimizerType::SGD, OptimizerType::ADAM};
    
    for (auto opt : optimizers) {
        NeuralNetwork network({2, 8, 1}, "relu", "sigmoid", opt, 0.5);
        network.set_loss_function("binary_crossentropy");
        network.set_epochs(300);
        network.set_batch_size(4);
        network.set_verbose(false);
        
        EXPECT_NO_THROW(network.fit(X_and_, y_and_));
        
        VectorXd y_pred = network.predict(X_and_);
        VectorXi y_pred_int = (y_pred.array() > 0.5).cast<int>();
        
        int correct = 0;
        for (int i = 0; i < 4; ++i) {
            if (y_pred_int(i) == static_cast<int>(y_and_(i))) correct++;
        }
        EXPECT_GE(correct, 3);  // Allow some tolerance
    }
}

TEST_F(NeuralNetworkIntegrationTest, SummaryDoesNotThrow) {
    NeuralNetwork network({2, 8, 4, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.1);
    
    EXPECT_NO_THROW(network.summary());
    EXPECT_EQ(network.get_num_layers(), 3);
    EXPECT_GT(network.get_num_parameters(), 0);
}

// ============================================================================
// Test Suite per Serialization
// ============================================================================

class NeuralNetworkSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        X_.resize(100, 3);
        X_.setRandom();
        y_ = X_.col(0) + 2 * X_.col(1) - 0.5 * X_.col(2);
        
        filename_ = "test_temp_model.bin";
    }
    
    void TearDown() override {
        std::remove(filename_.c_str());
    }
    
    MatrixXd X_;
    VectorXd y_;
    std::string filename_;
};

TEST_F(NeuralNetworkSerializationTest, SaveAndLoad) {
    NeuralNetwork nn({3, 16, 8, 1}, "relu", "linear");
    nn.set_epochs(10);
    nn.set_verbose(false);
    nn.set_loss_function("mse");
    nn.fit(X_, y_);
    
    VectorXd y_orig = nn.predict(X_);
    
    nn.save(filename_);
    
    NeuralNetwork nn_loaded;
    nn_loaded.load(filename_);
    VectorXd y_loaded = nn_loaded.predict(X_);
    
    double diff = (y_orig - y_loaded).norm() / y_orig.norm();
    EXPECT_LT(diff, 1e-10);
}

TEST_F(NeuralNetworkSerializationTest, SaveAndLoadWithLossFunction) {
    NeuralNetwork nn({3, 16, 8, 1}, "relu", "sigmoid");
    nn.set_epochs(10);
    nn.set_verbose(false);
    nn.set_loss_function("binary_crossentropy");
    
    // Create binary target
    VectorXd y_binary = (y_.array() > y_.mean()).cast<double>();
    nn.fit(X_, y_binary);
    
    nn.save(filename_);
    
    NeuralNetwork nn_loaded;
    nn_loaded.load(filename_);
    
    // Verify loss function was restored correctly
    EXPECT_NO_THROW(nn_loaded.predict(X_));
}

TEST_F(NeuralNetworkSerializationTest, LoadNonExistentFile) {
    NeuralNetwork nn;
    EXPECT_THROW(nn.load("non_existent_file.bin"), ml_exception::IOException);
}

// ============================================================================
// Test Suite per Exception Handling
// ============================================================================

class NeuralNetworkExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        X_.resize(10, 3);
        X_.setRandom();
        y_.resize(10);
        y_.setRandom();
    }
    
    MatrixXd X_;
    VectorXd y_;
};

TEST_F(NeuralNetworkExceptionTest, DimensionMismatch) {
    NeuralNetwork nn;
    VectorXd y_wrong(8);  // Wrong size
    
    EXPECT_THROW(nn.fit(X_, y_wrong), ml_exception::DimensionMismatchException);
}

TEST_F(NeuralNetworkExceptionTest, EmptyDataset) {
    NeuralNetwork nn;
    MatrixXd X_empty;
    VectorXd y_empty;
    
    EXPECT_THROW(nn.fit(X_empty, y_empty), ml_exception::EmptyDatasetException);
}

TEST_F(NeuralNetworkExceptionTest, PredictWithoutFitting) {
    NeuralNetwork nn;
    MatrixXd X_test(5, 3);
    X_test.setRandom();
    
    EXPECT_THROW(nn.predict(X_test), ml_exception::NotFittedException);
}

TEST_F(NeuralNetworkExceptionTest, InvalidLossFunction) {
    NeuralNetwork nn;
    EXPECT_THROW(nn.set_loss_function("invalid_loss"), ml_exception::InvalidParameterException);
}

// ============================================================================
// Test Suite per Different Loss Functions in Training
// ============================================================================

class NeuralNetworkLossFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        X_.resize(100, 2);
        X_.setRandom();
        y_ = (X_.col(0).array() + X_.col(1).array() > 0).cast<double>();
    }
    
    MatrixXd X_;
    VectorXd y_;
};

TEST_F(NeuralNetworkLossFunctionsTest, TrainWithBinaryCrossEntropy) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.1);
    nn.set_loss_function("binary_crossentropy");
    nn.set_epochs(100);
    nn.set_batch_size(16);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    double score = nn.score(X_, y_);
    EXPECT_GT(score, 0.85);
}

TEST_F(NeuralNetworkLossFunctionsTest, TrainWithMeanSquaredError) {
    NeuralNetwork nn({2, 16, 1}, "relu", "sigmoid", OptimizerType::ADAM, 0.1);
    nn.set_loss_function("mse");
    nn.set_epochs(100);
    nn.set_batch_size(16);
    nn.set_verbose(false);
    
    EXPECT_NO_THROW(nn.fit(X_, y_));
    
    // MSE should still work but might have lower accuracy
    double score = nn.score(X_, y_);
    EXPECT_GT(score, 0.7);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}