#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <limits>
#include "components/layers/dense_layer.h"
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
    
    MatrixXd expected(y_pred.rows(), 1);
    for (Eigen::Index i = 0; i < y_pred.rows(); ++i) {
        const double p = y_pred(i, 0);
        const double y = y_true(i, 0);
        expected(i, 0) = (p - y) / (p * (1.0 - p) * y_true.rows());
    }
    
    EXPECT_EQ(grad.rows(), expected.rows());
    EXPECT_EQ(grad.cols(), expected.cols());
    
    for (int i = 0; i < grad.rows(); ++i) {
        EXPECT_NEAR(grad(i, 0), expected(i, 0), 1e-6);
    }
}

TEST_F(BinaryCrossEntropyLossTest, SigmoidDenseGradientMatchesFiniteDifferences) {
    constexpr double epsilon = 1e-6;
    layers::DenseLayer output_layer(1, "sigmoid", false);
    output_layer.set_input_shape(2);

    Eigen::MatrixXd weights(2, 1);
    weights << 0.35, -0.2;
    output_layer.set_weights(weights);

    Eigen::MatrixXd input(3, 2);
    input << 1.0, 0.5,
             -0.4, 1.2,
             0.7, -0.8;
    Eigen::MatrixXd target(3, 1);
    target << 1.0, 0.0, 1.0;

    const Eigen::MatrixXd prediction = output_layer.forward(input, true);
    const Eigen::MatrixXd output_gradient = loss->gradient(target, prediction);
    output_layer.backward(output_gradient);
    const Eigen::MatrixXd analytical = output_layer.get_weights_gradient();

    Eigen::MatrixXd numerical = Eigen::MatrixXd::Zero(2, 1);
    for (Eigen::Index i = 0; i < weights.rows(); ++i) {
        Eigen::MatrixXd perturbed = weights;
        perturbed(i, 0) += epsilon;
        output_layer.set_weights(perturbed);
        const double loss_plus = loss->compute(target, output_layer.forward(input, false));

        perturbed(i, 0) -= 2.0 * epsilon;
        output_layer.set_weights(perturbed);
        const double loss_minus = loss->compute(target, output_layer.forward(input, false));
        numerical(i, 0) = (loss_plus - loss_minus) / (2.0 * epsilon);
    }

    EXPECT_TRUE(analytical.isApprox(numerical, 1e-6))
        << "analytical:\n" << analytical << "\nnumerical:\n" << numerical;
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

TEST_F(BinaryCrossEntropyLossTest, RejectsNonFiniteVectorPredictions) {
    VectorXd y_true = VectorXd::Ones(1);
    VectorXd y_pred(1);
    const double non_finite_values[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };

    for (const double value : non_finite_values) {
        y_pred(0) = value;
        EXPECT_THROW(loss->compute(y_true, y_pred), ml_exception::MLException);
    }
}

TEST_F(BinaryCrossEntropyLossTest, RejectsNonFiniteMatrixPredictions) {
    MatrixXd y_true = MatrixXd::Ones(1, 1);
    MatrixXd y_pred(1, 1);
    const double non_finite_values[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };

    for (const double value : non_finite_values) {
        y_pred(0, 0) = value;
        EXPECT_THROW(loss->compute(y_true, y_pred), ml_exception::MLException);
        EXPECT_THROW(loss->gradient(y_true, y_pred), ml_exception::MLException);
    }
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