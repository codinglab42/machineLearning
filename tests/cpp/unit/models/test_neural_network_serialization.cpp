// tests/cpp/unit/models/test_neural_network_serialization.cpp
#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "models/neural_network.h"

using namespace models;
using namespace Eigen;

class SerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Inizializza direttamente senza assegnazione
        nn = std::make_unique<NeuralNetwork>(
            std::initializer_list<int>{2, 4, 3}, 
            "relu", "softmax", OptimizerType::ADAM, 0.001
        );
        
        // Genera dati casuali
        X.resize(10, 2);
        y.resize(10);
        X.setRandom();
        for (int i = 0; i < 10; ++i) y(i) = rand() % 3;
        
        // Allena per pochi epoch
        nn->fit(X, y, 5, 5, false);
        
        filename = "test_model_temp.bin";
    }
    
    void TearDown() override {
        std::remove(filename.c_str());
    }
    
    std::unique_ptr<NeuralNetwork> nn;
    MatrixXd X;
    VectorXd y;
    std::string filename;
};

TEST_F(SerializationTest, SaveAndLoadModel) {
    // Salva il modello
    EXPECT_NO_THROW(nn->save(filename));
    
    // Carica il modello in una nuova istanza
    NeuralNetwork loaded_nn;
    EXPECT_NO_THROW(loaded_nn.load(filename));
    
    // Verifica che le predizioni siano identiche
    MatrixXd original_pred = nn->predict_proba(X);
    MatrixXd loaded_pred = loaded_nn.predict_proba(X);
    
    EXPECT_TRUE(original_pred.isApprox(loaded_pred, 1e-5));
}

TEST_F(SerializationTest, SerializeToStream) {
    std::stringstream ss;
    
    // Serializza su stream
    EXPECT_NO_THROW(nn->serialize_binary(ss));
    
    // Deserializza
    NeuralNetwork deserialized_nn;
    EXPECT_NO_THROW(deserialized_nn.deserialize_binary(ss));
    
    // Verifica
    MatrixXd original_pred = nn->predict_proba(X);
    MatrixXd deserialized_pred = deserialized_nn.predict_proba(X);
    
    EXPECT_TRUE(original_pred.isApprox(deserialized_pred, 1e-5));
}

TEST_F(SerializationTest, LoadInvalidFileThrowsException) {
    NeuralNetwork nn2;
    EXPECT_THROW(nn2.load("non_existent_file.bin"), ml_exception::FileNotFoundException);
}