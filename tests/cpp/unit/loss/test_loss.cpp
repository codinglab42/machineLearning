#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/loss/loss.h"
#include "exceptions/exception_macros.h"

using namespace loss;
using namespace Eigen;

// Mock class per testare la classe base astratta
class MockLoss : public Loss {
public:
    MOCK_METHOD(double, compute, (const VectorXd&, const VectorXd&), (const, override));
    MOCK_METHOD(double, compute, (const MatrixXd&, const MatrixXd&), (const, override));
    MOCK_METHOD(MatrixXd, gradient, (const MatrixXd&, const MatrixXd&), (const, override));
    MOCK_METHOD(std::string, name, (), (const, override));
};

TEST(LossTest, MockCanBeCreated) {
    MockLoss loss;
    EXPECT_CALL(loss, name()).WillOnce(testing::Return("mock_loss"));
    EXPECT_EQ(loss.name(), "mock_loss");
}

TEST(LossTest, VirtualDestructorWorks) {
    std::unique_ptr<Loss> loss = std::make_unique<MockLoss>();
    EXPECT_NE(loss, nullptr);
}