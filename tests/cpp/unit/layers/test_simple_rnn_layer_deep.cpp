// test/layers/simple_rnn_layer_test.cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/layers/simple_rnn_layer.h"
#include "components/cache/simple_rnn_cache.h"
#include "exceptions/ml_exception.h"

using namespace layers;

class SimpleRNNLayerTest : public ::testing::TestWithParam<std::tuple<std::string, bool>> {
protected:
    void SetUp() override {
        auto [activation, use_bias] = GetParam();
        layer = std::make_unique<SimpleRNNLayer>(4, 3, activation, use_bias);
        layer->set_input_shape(3);
    }
    
    std::unique_ptr<SimpleRNNLayer> layer;
};

TEST_P(SimpleRNNLayerTest, ConstructorValidation) {
    EXPECT_THROW(SimpleRNNLayer(0, 3, "tanh", true), ml_exception::InvalidParameterException);
    EXPECT_THROW(SimpleRNNLayer(4, 0, "tanh", true), ml_exception::InvalidParameterException);
    EXPECT_NO_THROW(SimpleRNNLayer(4, 3, "tanh", true));
}

TEST_P(SimpleRNNLayerTest, SetInputShape) {
    EXPECT_THROW(layer->set_input_shape(-1), ml_exception::InvalidParameterException);
    EXPECT_NO_THROW(layer->set_input_shape(5));
    EXPECT_EQ(layer->get_input_size(), 5);
}

TEST_P(SimpleRNNLayerTest, Forward) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    auto output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 4);
}

TEST_P(SimpleRNNLayerTest, StatePersistence) {
    Eigen::MatrixXd input1(1, 3);
    input1.setRandom();
    
    auto output1 = layer->forward(input1);
    auto state1 = layer->get_hidden_state();
    
    Eigen::MatrixXd input2(1, 3);
    input2.setRandom();
    
    auto output2 = layer->forward(input2);
    auto state2 = layer->get_hidden_state();
    
    EXPECT_FALSE(state1.isApprox(state2));
}

TEST_P(SimpleRNNLayerTest, ResetState) {
    Eigen::MatrixXd input(1, 3);
    // Usa input non-zero per assicurarsi che l'output non sia zero
    input << 1.0, 2.0, 3.0;
    
    auto output = layer->forward(input);
    
    // Lo stato dovrebbe essere diverso da zero
    EXPECT_FALSE(layer->get_hidden_state().isZero());
    
    layer->reset_state();
    EXPECT_TRUE(layer->get_hidden_state().size() == 0);
}

TEST_P(SimpleRNNLayerTest, Backward) {
    Eigen::MatrixXd input(3, 3);
    input.setRandom();
    
    auto output = layer->forward(input, true);
    
    Eigen::MatrixXd gradient(3, 4);
    gradient.setOnes();
    
    auto weights_before = layer->get_weights();
    auto dX = layer->backward(gradient, 0.1);  // LR più alto
    
    EXPECT_EQ(dX.rows(), input.rows());
    EXPECT_EQ(dX.cols(), input.cols());
    
    auto weights_after = layer->get_weights();
    EXPECT_FALSE(weights_before.isApprox(weights_after));
}

TEST_P(SimpleRNNLayerTest, BackwardInference) {
    Eigen::MatrixXd input(3, 3);
    input.setRandom();
    
    auto output = layer->forward(input, false); // inference
    
    Eigen::MatrixXd gradient(3, 4);
    gradient.setOnes();
    
    auto dX = layer->backward(gradient, 0.01);
    
    EXPECT_TRUE(dX.isApprox(gradient));
}

TEST_P(SimpleRNNLayerTest, ActivationFunctions) {
    Eigen::MatrixXd input(1, 3);
    input.setOnes();
    
    auto output = layer->forward(input);
    
    auto [activation, use_bias] = GetParam();
    if (activation == "tanh") {
        EXPECT_GE(output.minCoeff(), -1);
        EXPECT_LE(output.maxCoeff(), 1);
    } else if (activation == "relu") {
        EXPECT_GE(output.minCoeff(), 0);
    } else if (activation == "sigmoid") {
        EXPECT_GE(output.minCoeff(), 0);
        EXPECT_LE(output.maxCoeff(), 1);
    }
}

TEST_P(SimpleRNNLayerTest, Serialization) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    // Forward per popolare pesi
    layer->reset_state();
    auto output_before = layer->forward(input, true);
    auto config_before = layer->get_config();
    auto weights_before = layer->get_weights();
    
    auto [activation, use_bias] = GetParam();
    
    // DEBUG specifico per il caso problematico
    if (activation == "sigmoid" && use_bias) {
        std::cout << "\n=== DEBUG SIGMOID+BIAS ===" << std::endl;
        std::cout << "weights_before dims: " << weights_before.rows() 
                  << "x" << weights_before.cols() << std::endl;
        
        // Stampa statistiche dei pesi
        std::cout << "weights_before - min: " << weights_before.minCoeff() 
                  << ", max: " << weights_before.maxCoeff() 
                  << ", mean: " << weights_before.mean() << std::endl;
        
        // Stampa l'ultima colonna (bias)
        int last_col = weights_before.cols() - 1;
        std::cout << "Bias column before: " << std::endl;
        for (int i = 0; i < weights_before.rows(); ++i) {
            std::cout << "  [" << i << "] = " << weights_before(i, last_col) << std::endl;
        }
    }
    
    std::stringstream ss;
    layer->serialize(ss);
    
    // Crea nuovo layer con STESSI parametri
    auto new_layer = std::make_unique<SimpleRNNLayer>(4, 3, activation, use_bias);
    new_layer->set_input_shape(3);
    new_layer->deserialize(ss);
    
    auto weights_new = new_layer->get_weights();
    
    if (activation == "sigmoid" && use_bias) {
        std::cout << "\nweights_new dims: " << weights_new.rows() 
                  << "x" << weights_new.cols() << std::endl;
        std::cout << "weights_new - min: " << weights_new.minCoeff() 
                  << ", max: " << weights_new.maxCoeff() 
                  << ", mean: " << weights_new.mean() << std::endl;
        
        int last_col = weights_new.cols() - 1;
        std::cout << "Bias column after: " << std::endl;
        for (int i = 0; i < weights_new.rows(); ++i) {
            std::cout << "  [" << i << "] = " << weights_new(i, last_col) << std::endl;
        }
        
        // Confronto dettagliato
        std::cout << "\nDetailed comparison:" << std::endl;
        for (int i = 0; i < weights_before.rows(); ++i) {
            for (int j = 0; j < weights_before.cols(); ++j) {
                double diff = std::abs(weights_before(i, j) - weights_new(i, j));
                if (diff > 1e-6) {
                    std::cout << "Mismatch at [" << i << "," << j << "]: "
                              << "before=" << weights_before(i, j) 
                              << ", after=" << weights_new(i, j)
                              << ", diff=" << diff << std::endl;
                }
            }
        }
        std::cout << "==========================\n" << std::endl;
    }
    
    // Verifica configurazione
    EXPECT_EQ(new_layer->get_config(), config_before);
    EXPECT_EQ(new_layer->get_input_size(), layer->get_input_size());
    EXPECT_EQ(new_layer->get_output_size(), layer->get_output_size());
    
    // Verifica pesi con tolleranza
    double tolerance = use_bias ? 1e-5 : 1e-10;  // Tolleranza più alta per bias
    EXPECT_TRUE(weights_before.isApprox(weights_new, tolerance));
    
    // Verifica stato vuoto
    EXPECT_EQ(new_layer->get_hidden_state().size(), 0);
    
    // Verifica output
    new_layer->reset_state();
    auto output_new = new_layer->forward(input, false);
    EXPECT_TRUE(output_before.isApprox(output_new));
}

TEST_P(SimpleRNNLayerTest, BiasInitialization) {
    auto [activation, use_bias] = GetParam();
    
    if (use_bias) {
        // Controlla bias prima del forward
        auto biases_before = layer->get_biases();
        std::cout << "Bias before forward: " << biases_before.transpose() << std::endl;
        
        // Fai un forward
        Eigen::MatrixXd input(2, 3);
        input.setRandom();
        layer->forward(input, true);
        
        // Controlla bias dopo il forward
        auto biases_after = layer->get_biases();
        std::cout << "Bias after forward: " << biases_after.transpose() << std::endl;
        
        // Verifica che siano ancora zero
        for (int i = 0; i < biases_after.size(); ++i) {
            EXPECT_DOUBLE_EQ(biases_after(i), 0.0);
        }
    }
}

TEST_P(SimpleRNNLayerTest, DebugBiasInConstructor) {
    auto [activation, use_bias] = GetParam();
    
    if (use_bias) {
        // Crea un nuovo layer e controlla subito i bias
        auto test_layer = std::make_unique<SimpleRNNLayer>(4, 3, activation, true);
        
        auto biases = test_layer->get_biases();
        std::cout << "Bias after constructor for " << activation << ": ";
        for (int i = 0; i < biases.size(); ++i) {
            std::cout << biases(i) << " ";
        }
        std::cout << std::endl;
        
        for (int i = 0; i < biases.size(); ++i) {
            EXPECT_DOUBLE_EQ(biases(i), 0.0);
        }
    }
}

TEST_P(SimpleRNNLayerTest, ParameterCount) {
    auto [activation, use_bias] = GetParam();
    int expected = 3*4 + 4*4; // kernel + recurrent
    if (use_bias) expected += 4; // bias
    
    EXPECT_EQ(layer->get_parameter_count(), expected);
}

TEST_P(SimpleRNNLayerTest, BiasManagement) {
    auto [activation, use_bias] = GetParam();
    
    if (use_bias) {
        auto biases = layer->get_biases();
        EXPECT_EQ(biases.size(), 4);
        
        Eigen::VectorXd new_biases(4);
        new_biases << 1.0, 2.0, 3.0, 4.0;
        layer->set_biases(new_biases);
        
        auto updated_biases = layer->get_biases();
        EXPECT_TRUE(updated_biases.isApprox(new_biases));
    }
}

TEST_P(SimpleRNNLayerTest, ClearCache) {
    Eigen::MatrixXd input(2, 3);
    input.setRandom();
    
    layer->forward(input, true);
    
    auto cache = std::dynamic_pointer_cast<SimpleRNNCache>(layer->get_cache());
    ASSERT_NE(cache, nullptr);
    EXPECT_FALSE(cache->hidden_states.empty());
    
    layer->clear_cache();
    
    auto cache_after = std::dynamic_pointer_cast<SimpleRNNCache>(layer->get_cache());
    ASSERT_NE(cache_after, nullptr);
    EXPECT_TRUE(cache_after->hidden_states.empty());
}

TEST_P(SimpleRNNLayerTest, EmptyInput) {
    Eigen::MatrixXd empty;
    EXPECT_THROW(layer->forward(empty), ml_exception::EmptyDatasetException);
}

TEST_P(SimpleRNNLayerTest, BackwardWithoutForward) {
    Eigen::MatrixXd gradient(2, 4);
    gradient.setOnes();
    EXPECT_THROW(layer->backward(gradient, 0.01), ml_exception::NotFittedException);
}

INSTANTIATE_TEST_SUITE_P(
    SimpleRNNVariants,
    SimpleRNNLayerTest,
    ::testing::Combine(
        ::testing::Values("tanh", "relu", "sigmoid"),
        ::testing::Bool()
    )
);