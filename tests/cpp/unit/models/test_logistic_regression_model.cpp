// tests/cpp/unit/models/test_logistic_regression.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "models/logistic_regression.h"

using namespace models;
using namespace Eigen;

class LogisticRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea dati separabili linearmente
        X.resize(200, 2);
        y.resize(200);
        for (int i = 0; i < 200; ++i) {
            X(i, 0) = (i % 100) * 0.1;
            X(i, 1) = (i % 100) * 0.15;
            // Classe 1 se x0 + x1 > 5
            y(i) = (X(i, 0) + X(i, 1) > 5.0) ? 1.0 : 0.0;
        }
    }
    
    MatrixXd X;
    VectorXd y;
};

TEST_F(LogisticRegressionTest, DefaultConstructor) {
    LogisticRegression lr;
    EXPECT_EQ(lr.get_model_type(), "LogisticRegression");
}

TEST_F(LogisticRegressionTest, FitAndPredict) {
    LogisticRegression lr(0.1, 1000, 0.01);
    lr.fit(X, y);
    
    VectorXd probabilities = lr.predict(X);
    VectorXi predictions = lr.predict_class(X);
    
    EXPECT_EQ(probabilities.size(), 200);
    EXPECT_EQ(predictions.size(), 200);
    
    double accuracy = lr.score(X, y);
    EXPECT_GT(accuracy, 0.85);
}

TEST_F(LogisticRegressionTest, PredictWithThreshold) {
    LogisticRegression lr;
    lr.fit(X, y);
    
    VectorXi pred_low = lr.predict_class(X, 0.3);
    VectorXi pred_high = lr.predict_class(X, 0.7);
    
    // Diverse soglie dovrebbero produrre risultati diversi
    bool all_equal = true;
    for (int i = 0; i < pred_low.size(); ++i) {
        if (pred_low(i) != pred_high(i)) {
            all_equal = false;
            break;
        }
    }
    EXPECT_FALSE(all_equal);
}

TEST_F(LogisticRegressionTest, ConfusionMatrix) {
    LogisticRegression lr;
    lr.fit(X, y);
    
    MatrixXd cm = lr.confusion_matrix(X, y);
    
    EXPECT_EQ(cm.rows(), 2);
    EXPECT_EQ(cm.cols(), 2);
    EXPECT_GT(cm(0, 0), 0);  // True negatives
    EXPECT_GT(cm(1, 1), 0);  // True positives
}

TEST_F(LogisticRegressionTest, PrecisionRecallF1) {
    LogisticRegression lr;
    lr.fit(X, y);
    
    Vector3d metrics = lr.precision_recall_f1(X, y);
    
    EXPECT_GT(metrics(0), 0.8);  // precision
    EXPECT_GT(metrics(1), 0.8);  // recall
    EXPECT_GT(metrics(2), 0.8);  // f1
}

TEST_F(LogisticRegressionTest, RegularizationEffect) {
    // Con regolarizzazione
    LogisticRegression lr_reg(0.1, 500, 0.1);
    lr_reg.fit(X, y);
    double acc_reg = lr_reg.score(X, y);
    
    // Senza regolarizzazione
    LogisticRegression lr_no_reg(0.1, 500, 0.0);
    lr_no_reg.fit(X, y);
    double acc_no_reg = lr_no_reg.score(X, y);
    
    EXPECT_GT(acc_reg, 0.8);
    EXPECT_GT(acc_no_reg, 0.8);
}

TEST_F(LogisticRegressionTest, TrainingHistory) {
    LogisticRegression lr(0.1, 200, 0.01, 1e-4, true);
    lr.fit(X, y);
    
    const auto& cost_history = lr.cost_history();
    const auto& acc_history = lr.accuracy_history();
    
    EXPECT_GT(cost_history.size(), 0);
    EXPECT_GT(acc_history.size(), 0);
    EXPECT_GT(cost_history[0], cost_history.back());
}

TEST_F(LogisticRegressionTest, Serialization) {
    LogisticRegression lr;
    lr.fit(X, y);
    
    std::string filename = "test_log_model.bin";
    lr.save(filename);
    
    LogisticRegression loaded_lr;
    loaded_lr.load(filename);
    
    VectorXd pred_original = lr.predict(X);
    VectorXd pred_loaded = loaded_lr.predict(X);
    
    EXPECT_TRUE(pred_original.isApprox(pred_loaded, 1e-6));
    
    std::remove(filename.c_str());
}