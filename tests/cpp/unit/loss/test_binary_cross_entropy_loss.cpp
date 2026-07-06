#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/loss/binary_cross_entropy_loss.h"
#include "exceptions/exception_macros.h"

using namespace loss;
using namespace Eigen;

class BinaryCrossEntropyLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        loss = std::make_unique<BinaryCrossEntropyLoss>();
    }
    
    std::unique_ptr<BinaryCrossEntropyLoss> loss;
};

TEST_F(BinaryCrossEntropyLossTest, PerfectPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 0, 1;
    y_pred << 1, 0, 1;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    EXPECT_NEAR(loss_value, 0.0, 1e-6);
}

TEST_F(BinaryCrossEntropyLossTest, WorstPredictions) {
    VectorXd y_true(3);
    VectorXd y_pred(3);
    y_true << 1, 0, 1;
    y_pred << 0, 1, 0;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    // Dovrebbe essere un valore grande (tende a infinito ma clip previene)
    EXPECT_GT(loss_value, 10.0);
}

TEST_F(BinaryCrossEntropyLossTest, MatrixInput) {
    MatrixXd y_true(3, 1);
    MatrixXd y_pred(3, 1);
    y_true << 1, 0, 1;
    y_pred << 0.9, 0.1, 0.8;
    
    double loss_value = loss->compute(y_true, y_pred);
    
    // Calcolo manuale
    double expected = -((1*log(0.9) + 0*log(0.1) + 1*log(0.8) + 
                         0*log(0.1) + 1*log(0.9) + 0*log(0.2)) / 3.0);
    EXPECT_NEAR(loss_value, expected, 1e-6);
}

TEST_F(BinaryCrossEntropyLossTest, Gradient) {
    MatrixXd y_true(3, 1);
    MatrixXd y_pred(3, 1);
    y_true << 1, 0, 1;
    y_pred << 0.8, 0.2, 0.7;
    
    MatrixXd grad = loss->gradient(y_true, y_pred);
    
    // Binary cross-entropy con mean reduction: grad = (pred - true) / N
    int n = y_true.rows();
    MatrixXd expected = (y_pred - y_true) / static_cast<double>(n);
    
    // Valori attesi:
    // Campione 0: (0.8 - 1.0) / 3 = -0.0666667
    // Campione 1: (0.2 - 0.0) / 3 =  0.0666667
    // Campione 2: (0.7 - 1.0) / 3 = -0.1000000
    
    EXPECT_EQ(grad.rows(), expected.rows());
    EXPECT_EQ(grad.cols(), expected.cols());
    
    for (int i = 0; i < grad.rows(); ++i) {
        EXPECT_NEAR(grad(i, 0), expected(i, 0), 1e-6);
    }
}

TEST_F(BinaryCrossEntropyLossTest, NumericalStability) {
    VectorXd y_true(1);
    VectorXd y_pred(1);
    y_true << 1;
    y_pred << 0.0;  // Estremo
    
    // Non dovrebbe lanciare eccezioni
    EXPECT_NO_THROW(loss->compute(y_true, y_pred));
    
    y_pred << 1.0;  // Estremo
    EXPECT_NO_THROW(loss->compute(y_true, y_pred));
}

TEST_F(BinaryCrossEntropyLossTest, Name) {
    EXPECT_EQ(loss->name(), "binary_crossentropy");
}

TEST_F(BinaryCrossEntropyLossTest, DimensionMismatch) {
    VectorXd y_true(3);
    VectorXd y_pred(2);
    
    EXPECT_THROW(loss->compute(y_true, y_pred), ml_exception::DimensionMismatchException);
}

TEST_F(BinaryCrossEntropyLossTest, ProbabilityRange) {
    VectorXd y_true(2);
    VectorXd y_pred(2);
    y_true << 1, 0;
    y_pred << 1.5, -0.5;  // Fuori range [0,1]
    
    // Il loss function dovrebbe comunque gestirlo (clip)
    EXPECT_NO_THROW(loss->compute(y_true, y_pred));
}