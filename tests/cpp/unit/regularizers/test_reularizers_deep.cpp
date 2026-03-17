// test/regularizers/regularizer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/regularizers/l1_regularizer.h"
#include "components/regularizers/l2_regularizer.h"
#include "components/regularizers/elastic_net_regularizer.h"
#include "components/regularizers/regularizer_factory.h"
#include "exceptions/ml_exception.h"

using namespace models;
using namespace testing;

//=============================================================================
// L1 REGULARIZER TESTS
//=============================================================================

class L1RegularizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        regularizer = std::make_unique<L1Regularizer>(0.1);
    }
    
    std::unique_ptr<L1Regularizer> regularizer;
};

TEST_F(L1RegularizerTest, ComputeLoss) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    // L1 loss = 0.1 * (|1| + | -2| + |0| + |3|) = 0.1 * 6 = 0.6
    double expected_loss = 0.1 * (1.0 + 2.0 + 0.0 + 3.0);
    
    double loss = regularizer->compute_loss(w);
    EXPECT_DOUBLE_EQ(loss, expected_loss);
}

TEST_F(L1RegularizerTest, ComputeBiasLoss) {
    Eigen::VectorXd b(3);
    b << 1.0, -2.0, 0.5;
    
    // L1 loss = 0.1 * (|1| + | -2| + |0.5|) = 0.1 * 3.5 = 0.35
    double expected_loss = 0.1 * (1.0 + 2.0 + 0.5);
    
    double loss = regularizer->compute_loss(b);
    EXPECT_DOUBLE_EQ(loss, expected_loss);
}

TEST_F(L1RegularizerTest, ComputeGradient) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    Eigen::MatrixXd grad = regularizer->compute_gradient(w);
    
    // Gradiente L1 = 0.1 * sign(w)
    EXPECT_DOUBLE_EQ(grad(0,0), 0.1);   // sign(1) = 1
    EXPECT_DOUBLE_EQ(grad(0,1), -0.1);  // sign(-2) = -1
    EXPECT_DOUBLE_EQ(grad(1,0), 0.0);   // sign(0) = 0
    EXPECT_DOUBLE_EQ(grad(1,1), 0.1);   // sign(3) = 1
}

TEST_F(L1RegularizerTest, ComputeBiasGradient) {
    Eigen::VectorXd b(3);
    b << 1.0, -2.0, 0.0;
    
    Eigen::VectorXd grad = regularizer->compute_gradient(b);
    
    EXPECT_DOUBLE_EQ(grad(0), 0.1);
    EXPECT_DOUBLE_EQ(grad(1), -0.1);
    EXPECT_DOUBLE_EQ(grad(2), 0.0);
}

TEST_F(L1RegularizerTest, Clone) {
    auto clone = regularizer->clone();
    
    EXPECT_EQ(clone->get_type(), regularizer->get_type());
    EXPECT_DOUBLE_EQ(clone->get_strength(), regularizer->get_strength());
}

TEST_F(L1RegularizerTest, Serialization) {
    std::stringstream ss;
    regularizer->serialize(ss);
    
    L1Regularizer deserialized(0.0);
    deserialized.deserialize(ss);
    
    EXPECT_DOUBLE_EQ(deserialized.get_strength(), regularizer->get_strength());
}

//=============================================================================
// L2 REGULARIZER TESTS
//=============================================================================

class L2RegularizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        regularizer = std::make_unique<L2Regularizer>(0.1);
    }
    
    std::unique_ptr<L2Regularizer> regularizer;
};

TEST_F(L2RegularizerTest, ComputeLoss) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    // L2 loss = 0.5 * 0.1 * (1^2 + (-2)^2 + 0^2 + 3^2) 
    //        = 0.05 * (1 + 4 + 0 + 9) = 0.05 * 14 = 0.7
    double expected_loss = 0.5 * 0.1 * (1.0 + 4.0 + 0.0 + 9.0);
    
    double loss = regularizer->compute_loss(w);
    EXPECT_DOUBLE_EQ(loss, expected_loss);
}

TEST_F(L2RegularizerTest, ComputeBiasLoss) {
    Eigen::VectorXd b(3);
    b << 1.0, -2.0, 0.5;
    
    // L2 loss = 0.5 * 0.1 * (1^2 + 4 + 0.25) = 0.05 * 5.25 = 0.2625
    double expected_loss = 0.5 * 0.1 * (1.0 + 4.0 + 0.25);
    
    double loss = regularizer->compute_loss(b);
    EXPECT_DOUBLE_EQ(loss, expected_loss);
}

TEST_F(L2RegularizerTest, ComputeGradient) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    Eigen::MatrixXd grad = regularizer->compute_gradient(w);
    
    // Gradiente L2 = 0.1 * w
    EXPECT_DOUBLE_EQ(grad(0,0), 0.1);   // 0.1 * 1
    EXPECT_DOUBLE_EQ(grad(0,1), -0.2);  // 0.1 * -2
    EXPECT_DOUBLE_EQ(grad(1,0), 0.0);   // 0.1 * 0
    EXPECT_DOUBLE_EQ(grad(1,1), 0.3);   // 0.1 * 3
}

TEST_F(L2RegularizerTest, Clone) {
    auto clone = regularizer->clone();
    
    EXPECT_EQ(clone->get_type(), regularizer->get_type());
    EXPECT_DOUBLE_EQ(clone->get_strength(), regularizer->get_strength());
}

//=============================================================================
// ELASTIC NET REGULARIZER TESTS
//=============================================================================

class ElasticNetRegularizerTest : public ::testing::TestWithParam<double> {
protected:
    void SetUp() override {
        l1_ratio = GetParam();
        regularizer = std::make_unique<ElasticNetRegularizer>(0.1, l1_ratio);
    }
    
    double l1_ratio;
    std::unique_ptr<ElasticNetRegularizer> regularizer;
};

TEST_P(ElasticNetRegularizerTest, ConstructorValidation) {
    EXPECT_THROW(ElasticNetRegularizer(0.1, -0.1), std::invalid_argument);
    EXPECT_THROW(ElasticNetRegularizer(0.1, 1.1), std::invalid_argument);
    EXPECT_NO_THROW(ElasticNetRegularizer(0.1, 0.5));
}

TEST_P(ElasticNetRegularizerTest, ComputeLoss) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    // L1 part
    double l1 = 0.1 * l1_ratio * (1.0 + 2.0 + 0.0 + 3.0);
    // L2 part
    double l2 = 0.1 * (1.0 - l1_ratio) * 0.5 * (1.0 + 4.0 + 0.0 + 9.0);
    double expected_loss = l1 + l2;
    
    double loss = regularizer->compute_loss(w);
    EXPECT_NEAR(loss, expected_loss, 1e-10);
}

TEST_P(ElasticNetRegularizerTest, ComputeGradient) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, -2.0,
         0.0, 3.0;
    
    Eigen::MatrixXd grad = regularizer->compute_gradient(w);
    
    // Verifica che per l1_ratio=0 sia equivalente a L2
    if (l1_ratio == 0.0) {
        Eigen::MatrixXd expected = 0.1 * w;
        EXPECT_TRUE(grad.isApprox(expected));
    }
    // Verifica che per l1_ratio=1 sia equivalente a L1
    else if (l1_ratio == 1.0) {
        EXPECT_DOUBLE_EQ(grad(0,0), 0.1);
        EXPECT_DOUBLE_EQ(grad(0,1), -0.1);
        EXPECT_DOUBLE_EQ(grad(1,0), 0.0);
        EXPECT_DOUBLE_EQ(grad(1,1), 0.1);
    }
}

TEST_P(ElasticNetRegularizerTest, Clone) {
    auto clone = regularizer->clone();
    
    EXPECT_EQ(clone->get_type(), regularizer->get_type());
    EXPECT_DOUBLE_EQ(clone->get_strength(), regularizer->get_strength());
    
    auto elastic_clone = dynamic_cast<ElasticNetRegularizer*>(clone.get());
    ASSERT_NE(elastic_clone, nullptr);
    EXPECT_DOUBLE_EQ(elastic_clone->get_l1_ratio(), regularizer->get_l1_ratio());
}

TEST_P(ElasticNetRegularizerTest, Serialization) {
    std::stringstream ss;
    regularizer->serialize(ss);
    
    ElasticNetRegularizer deserialized;
    deserialized.deserialize(ss);
    
    EXPECT_DOUBLE_EQ(deserialized.get_strength(), regularizer->get_strength());
    EXPECT_DOUBLE_EQ(deserialized.get_l1_ratio(), regularizer->get_l1_ratio());
}

INSTANTIATE_TEST_SUITE_P(
    ElasticNetVariants,
    ElasticNetRegularizerTest,
    ::testing::Values(0.0, 0.3, 0.5, 0.7, 1.0)
);

//=============================================================================
// REGULARIZER FACTORY TESTS
//=============================================================================

class RegularizerFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(RegularizerFactoryTest, CreateL1) {
    auto reg = RegularizerFactory::create("l1", 0.1);
    
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->get_type(), RegularizerType::L1);
    EXPECT_EQ(reg->get_type_str(), "l1");
    EXPECT_DOUBLE_EQ(reg->get_strength(), 0.1);
}

TEST_F(RegularizerFactoryTest, CreateL2) {
    auto reg = RegularizerFactory::create("l2", 0.2);
    
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->get_type(), RegularizerType::L2);
    EXPECT_EQ(reg->get_type_str(), "l2");
    EXPECT_DOUBLE_EQ(reg->get_strength(), 0.2);
}

TEST_F(RegularizerFactoryTest, CreateElasticNet) {
    std::unordered_map<std::string, double> params;
    params["l1_ratio"] = 0.3;
    
    auto reg = RegularizerFactory::create("elastic_net", 0.1, params);
    
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->get_type(), RegularizerType::ELASTIC_NET);
    EXPECT_EQ(reg->get_type_str(), "elastic_net");
    EXPECT_DOUBLE_EQ(reg->get_strength(), 0.1);
    
    auto elastic = dynamic_cast<ElasticNetRegularizer*>(reg.get());
    ASSERT_NE(elastic, nullptr);
    EXPECT_DOUBLE_EQ(elastic->get_l1_ratio(), 0.3);
}

TEST_F(RegularizerFactoryTest, CreateNone) {
    auto reg = RegularizerFactory::create("none");
    EXPECT_EQ(reg, nullptr);
}

TEST_F(RegularizerFactoryTest, InvalidType) {
    EXPECT_THROW(RegularizerFactory::create("invalid"), ml_exception::InvalidParameterException);
}

TEST_F(RegularizerFactoryTest, StringConversion) {
    EXPECT_EQ(RegularizerFactory::string_to_type("l1"), RegularizerType::L1);
    EXPECT_EQ(RegularizerFactory::string_to_type("l2"), RegularizerType::L2);
    EXPECT_EQ(RegularizerFactory::string_to_type("elastic_net"), RegularizerType::ELASTIC_NET);
    EXPECT_EQ(RegularizerFactory::string_to_type("none"), RegularizerType::NONE);
    
    EXPECT_EQ(RegularizerFactory::type_to_string(RegularizerType::L1), "l1");
    EXPECT_EQ(RegularizerFactory::type_to_string(RegularizerType::L2), "l2");
    EXPECT_EQ(RegularizerFactory::type_to_string(RegularizerType::ELASTIC_NET), "elastic_net");
    EXPECT_EQ(RegularizerFactory::type_to_string(RegularizerType::NONE), "none");
}

//=============================================================================
// REGULARIZER INTERFACE TESTS
//=============================================================================

TEST(RegularizerInterfaceTest, AllRegularizersDeriveFromBase) {
    L1Regularizer l1(0.1);
    L2Regularizer l2(0.1);
    ElasticNetRegularizer elastic(0.1, 0.5);
    
    std::vector<Regularizer*> regularizers = {&l1, &l2, &elastic};
    
    for (auto* reg : regularizers) {
        EXPECT_NE(reg->get_type_str(), "");
    }
}

TEST(RegularizerInterfaceTest, ZeroStrength) {
    L1Regularizer l1(0.0);
    L2Regularizer l2(0.0);
    ElasticNetRegularizer elastic(0.0, 0.5);
    
    Eigen::MatrixXd w = Eigen::MatrixXd::Random(3, 3);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    
    // Con strength = 0, loss e gradient dovrebbero essere zero
    EXPECT_DOUBLE_EQ(l1.compute_loss(w), 0.0);
    EXPECT_DOUBLE_EQ(l2.compute_loss(w), 0.0);
    EXPECT_DOUBLE_EQ(elastic.compute_loss(w), 0.0);
    
    EXPECT_TRUE(l1.compute_gradient(w).isZero());
    EXPECT_TRUE(l2.compute_gradient(w).isZero());
    EXPECT_TRUE(elastic.compute_gradient(w).isZero());
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}