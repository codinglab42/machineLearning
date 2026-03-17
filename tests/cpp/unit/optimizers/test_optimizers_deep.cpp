// test/optimizers/optimizer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/optimizers/sgd_optimizer.h"
#include "components/optimizers/momentum_optimizer.h"
#include "components/optimizers/adam_optimizer.h"
#include "components/optimizers/optimizer_factory.h"
#include "exceptions/ml_exception.h"

using namespace models;
using namespace testing;

//=============================================================================
// TEST BASE PER TUTTI GLI OPTIMIZER
//=============================================================================

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Inizializzazione comune
    }
};

TEST_F(OptimizerTest, LearningRateDecay) {
    SGDOptimizer optimizer(1.0, 0.1);  // lr=1.0, decay=0.1
    
    // iterations = 0
    EXPECT_DOUBLE_EQ(optimizer.get_current_learning_rate(), 1.0);
    
    // Simula alcune iterazioni
    Eigen::MatrixXd w(2, 2);
    Eigen::MatrixXd g(2, 2);
    w.setRandom();
    g.setOnes();
    
    for (int i = 0; i < 5; ++i) {
        optimizer.update(w, g);
    }
    
    // Dopo 5 iterazioni: lr = 1.0 / (1.0 + 0.1*5) = 1.0/1.5 = 0.666...
    EXPECT_NEAR(optimizer.get_current_learning_rate(), 0.6666667, 1e-6);
    EXPECT_EQ(optimizer.get_iterations(), 5);
}

TEST_F(OptimizerTest, Reset) {
    MomentumOptimizer optimizer(0.01, 0.9);
    
    Eigen::MatrixXd w(2, 2);
    Eigen::MatrixXd g(2, 2);
    w.setRandom();
    g.setOnes();
    
    optimizer.update(w, g);
    EXPECT_EQ(optimizer.get_iterations(), 1);
    
    optimizer.reset();
    EXPECT_EQ(optimizer.get_iterations(), 0);
    
    // Dopo reset, lo stato interno dovrebbe essere vuoto
    auto clone = optimizer.clone();
    // Non possiamo testare direttamente, ma non dovrebbe crashare
}

//=============================================================================
// SGD OPTIMIZER TESTS
//=============================================================================

class SGDOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        optimizer = std::make_unique<SGDOptimizer>(0.01, 0.0);
    }
    
    std::unique_ptr<SGDOptimizer> optimizer;
};

TEST_F(SGDOptimizerTest, BasicUpdate) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, 2.0,
         3.0, 4.0;
    
    Eigen::MatrixXd g(2, 2);
    g << 0.1, 0.1,
         0.1, 0.1;
    
    Eigen::MatrixXd expected = w - 0.01 * g;
    
    optimizer->update(w, g);
    
    EXPECT_TRUE(w.isApprox(expected));
}

TEST_F(SGDOptimizerTest, BiasUpdate) {
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    
    Eigen::VectorXd g(3);
    g << 0.1, 0.1, 0.1;
    
    Eigen::VectorXd expected = b - 0.01 * g;
    
    optimizer->update(b, g);
    
    EXPECT_TRUE(b.isApprox(expected));
}

TEST_F(SGDOptimizerTest, Clone) {
    auto clone = optimizer->clone();
    
    EXPECT_EQ(clone->get_type(), optimizer->get_type());
    EXPECT_EQ(clone->get_learning_rate(), optimizer->get_learning_rate());
    EXPECT_EQ(clone->get_decay(), optimizer->get_decay());
}

//=============================================================================
// MOMENTUM OPTIMIZER TESTS
//=============================================================================

class MomentumOptimizerTest : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        nesterov = GetParam();
        optimizer = std::make_unique<MomentumOptimizer>(0.01, 0.9, 0.0, nesterov);
    }
    
    bool nesterov;
    std::unique_ptr<MomentumOptimizer> optimizer;
};

TEST_P(MomentumOptimizerTest, FirstUpdate) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, 2.0,
         3.0, 4.0;
    
    Eigen::MatrixXd g(2, 2);
    g << 0.1, 0.1,
         0.1, 0.1;
    
    Eigen::MatrixXd w_before = w;
    optimizer->update(w, g);
    
    // Al primo update, velocity = -lr * g
    // weights += velocity (o formula Nesterov)
    EXPECT_FALSE(w.isApprox(w_before));  // Dovrebbe cambiare
}

TEST_P(MomentumOptimizerTest, SecondUpdate) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, 2.0,
         3.0, 4.0;
    
    Eigen::MatrixXd g(2, 2);
    g << 0.1, 0.1,
         0.1, 0.1;
    
    Eigen::MatrixXd w1 = w;
    optimizer->update(w1, g);  // primo update
    
    Eigen::MatrixXd w2 = w1;
    optimizer->update(w2, g);  // secondo update
    
    // Il secondo update dovrebbe essere diverso dal primo (momentum accumulato)
    EXPECT_FALSE((w2 - w1).isApprox(w1 - w));
}

TEST_P(MomentumOptimizerTest, BiasUpdate) {
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    
    Eigen::VectorXd g(3);
    g << 0.1, 0.1, 0.1;
    
    Eigen::VectorXd b_before = b;
    optimizer->update(b, g);
    
    EXPECT_FALSE(b.isApprox(b_before));
}

TEST_P(MomentumOptimizerTest, Clone) {
    auto clone = optimizer->clone();
    
    EXPECT_EQ(clone->get_type(), optimizer->get_type());
    EXPECT_EQ(clone->get_learning_rate(), optimizer->get_learning_rate());
    
    // Test che lo stato venga copiato
    Eigen::MatrixXd w(2, 2);
    w.setRandom();
    Eigen::MatrixXd g(2, 2);
    g.setOnes();
    
    optimizer->update(w, g);
    clone->update(w, g);  // Dovrebbe funzionare
}

INSTANTIATE_TEST_SUITE_P(
    MomentumVariants,
    MomentumOptimizerTest,
    ::testing::Bool()
);

//=============================================================================
// ADAM OPTIMIZER TESTS
//=============================================================================

class AdamOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        optimizer = std::make_unique<AdamOptimizer>(0.001, 0.9, 0.999, 1e-8);
    }
    
    std::unique_ptr<AdamOptimizer> optimizer;
};

TEST_F(AdamOptimizerTest, FirstUpdate) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, 2.0,
         3.0, 4.0;
    
    Eigen::MatrixXd g(2, 2);
    g << 0.1, 0.1,
         0.1, 0.1;
    
    Eigen::MatrixXd w_before = w;
    optimizer->update(w, g);
    
    EXPECT_FALSE(w.isApprox(w_before));
    EXPECT_EQ(optimizer->get_iterations(), 1);
}

TEST_F(AdamOptimizerTest, MultipleUpdates) {
    Eigen::MatrixXd w(2, 2);
    w << 1.0, 2.0,
         3.0, 4.0;
    
    Eigen::MatrixXd g(2, 2);
    g << 0.1, 0.1,
         0.1, 0.1;
    
    std::vector<Eigen::MatrixXd> steps;
    
    for (int i = 0; i < 5; ++i) {
        steps.push_back(w);
        optimizer->update(w, g);
    }
    
    // Verifica che gli step siano diversi
    for (size_t i = 1; i < steps.size(); ++i) {
        EXPECT_FALSE(steps[i].isApprox(steps[i-1]));
    }
}

TEST_F(AdamOptimizerTest, BiasUpdate) {
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    
    Eigen::VectorXd g(3);
    g << 0.1, 0.1, 0.1;
    
    Eigen::VectorXd b_before = b;
    optimizer->update(b, g);
    
    EXPECT_FALSE(b.isApprox(b_before));
}

TEST_F(AdamOptimizerTest, Clone) {
    auto clone = optimizer->clone();
    
    EXPECT_EQ(clone->get_type(), optimizer->get_type());
    EXPECT_EQ(clone->get_learning_rate(), optimizer->get_learning_rate());
    
    // Test che lo stato venga copiato
    Eigen::MatrixXd w(2, 2);
    w.setRandom();
    Eigen::MatrixXd g(2, 2);
    g.setOnes();
    
    optimizer->update(w, g);
    clone->update(w, g);  // Dovrebbe funzionare
}

TEST_F(AdamOptimizerTest, ParameterBounds) {
    // Beta devono essere tra 0 e 1
    EXPECT_THROW(AdamOptimizer(0.001, 1.5, 0.999), ml_exception::InvalidParameterException);
    EXPECT_THROW(AdamOptimizer(0.001, 0.9, 1.5), ml_exception::InvalidParameterException);
    EXPECT_THROW(AdamOptimizer(0.001, 0.9, 0.999, -1e-8), ml_exception::InvalidParameterException);
}

//=============================================================================
// OPTIMIZER FACTORY TESTS
//=============================================================================

class OptimizerFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Inizializzazione
    }
};

TEST_F(OptimizerFactoryTest, CreateSGD) {
    auto optimizer = OptimizerFactory::create("sgd", 0.01);
    
    EXPECT_EQ(optimizer->get_type(), OptimizerType::SGD);
    EXPECT_EQ(optimizer->get_type_str(), "sgd");
    EXPECT_DOUBLE_EQ(optimizer->get_learning_rate(), 0.01);
}

TEST_F(OptimizerFactoryTest, CreateMomentum) {
    std::unordered_map<std::string, double> params;
    params["momentum"] = 0.9;
    
    auto optimizer = OptimizerFactory::create("momentum", 0.01, params);
    
    EXPECT_EQ(optimizer->get_type(), OptimizerType::MOMENTUM);
    EXPECT_EQ(optimizer->get_type_str(), "momentum");
}

TEST_F(OptimizerFactoryTest, CreateAdam) {
    std::unordered_map<std::string, double> params;
    params["beta1"] = 0.9;
    params["beta2"] = 0.999;
    params["epsilon"] = 1e-8;
    
    auto optimizer = OptimizerFactory::create("adam", 0.001, params);
    
    EXPECT_EQ(optimizer->get_type(), OptimizerType::ADAM);
    EXPECT_EQ(optimizer->get_type_str(), "adam");
}

TEST_F(OptimizerFactoryTest, InvalidType) {
    EXPECT_THROW(OptimizerFactory::create("invalid"), ml_exception::InvalidParameterException);
}

TEST_F(OptimizerFactoryTest, StringConversion) {
    EXPECT_EQ(OptimizerFactory::string_to_type("sgd"), OptimizerType::SGD);
    EXPECT_EQ(OptimizerFactory::string_to_type("momentum"), OptimizerType::MOMENTUM);
    EXPECT_EQ(OptimizerFactory::string_to_type("adam"), OptimizerType::ADAM);
    
    EXPECT_EQ(OptimizerFactory::type_to_string(OptimizerType::SGD), "sgd");
    EXPECT_EQ(OptimizerFactory::type_to_string(OptimizerType::MOMENTUM), "momentum");
    EXPECT_EQ(OptimizerFactory::type_to_string(OptimizerType::ADAM), "adam");
}

//=============================================================================
// SERIALIZATION TESTS
//=============================================================================

TEST_F(OptimizerTest, SGDSerialization) {
    SGDOptimizer original(0.01, 0.1);
    
    std::stringstream ss;
    original.serialize(ss);
    
    SGDOptimizer deserialized(0.0, 0.0);
    deserialized.deserialize(ss);
    
    EXPECT_DOUBLE_EQ(deserialized.get_learning_rate(), original.get_learning_rate());
    EXPECT_DOUBLE_EQ(deserialized.get_decay(), original.get_decay());
    EXPECT_EQ(deserialized.get_iterations(), original.get_iterations());
}

TEST_F(OptimizerTest, MomentumSerialization) {
    MomentumOptimizer original(0.01, 0.9, 0.1, true);
    
    std::stringstream ss;
    original.serialize(ss);
    
    MomentumOptimizer deserialized(0.0, 0.0);
    deserialized.deserialize(ss);
    
    EXPECT_DOUBLE_EQ(deserialized.get_learning_rate(), original.get_learning_rate());
    EXPECT_DOUBLE_EQ(deserialized.get_decay(), original.get_decay());
}

TEST_F(OptimizerTest, AdamSerialization) {
    AdamOptimizer original(0.001, 0.9, 0.999, 1e-8, 0.1);
    
    // Fai qualche update per popolare lo stato
    Eigen::MatrixXd w(2, 2);
    w.setRandom();
    Eigen::MatrixXd g(2, 2);
    g.setOnes();
    original.update(w, g);
    
    std::stringstream ss;
    original.serialize(ss);
    
    AdamOptimizer deserialized;
    deserialized.deserialize(ss);
    
    EXPECT_DOUBLE_EQ(deserialized.get_learning_rate(), original.get_learning_rate());
    EXPECT_DOUBLE_EQ(deserialized.get_decay(), original.get_decay());
    EXPECT_EQ(deserialized.get_iterations(), original.get_iterations());
    
    // Dopo deserializzazione, un altro update dovrebbe funzionare
    EXPECT_NO_THROW(deserialized.update(w, g));
}

//=============================================================================
// MAIN
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}