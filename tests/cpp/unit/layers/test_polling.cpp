#include <gtest/gtest.h>
#include "components/layers/pooling.h"
#include <sstream>

using namespace layers;
using Eigen::MatrixXd;

class PoolingLayerTest : public ::testing::Test {
protected:
    // Helper per creare una matrice di input "immagine" 4x4 appiattita
    // 1 canali, batch size 1. Input:
    // 1  2  3  4
    // 5  6  7  8
    // 9 10 11 12
    // 13 14 15 16
    MatrixXd create_4x4_input() {
        MatrixXd input(1, 16);
        for (int i = 0; i < 16; ++i) input(0, i) = i + 1;
        return input;
    }
};

// 1. Test Forward - Max Pooling
TEST_F(PoolingLayerTest, ForwardMaxPooling) {
    // Pool 2x2, Stride 2 -> Output 2x2
    Pooling pool(2, 2, Pooling::MAX, 1);
    MatrixXd input = create_4x4_input();
    
    MatrixXd output = pool.forward(input);

    // Risultato atteso:
    // max(1,2,5,6)=6,   max(3,4,7,8)=8
    // max(9,10,13,14)=14, max(11,12,15,16)=16
    ASSERT_EQ(output.cols(), 4); 
    EXPECT_DOUBLE_EQ(output(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(output(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(output(0, 2), 14.0);
    EXPECT_DOUBLE_EQ(output(0, 3), 16.0);
}

// 2. Test Forward - Average Pooling
TEST_F(PoolingLayerTest, ForwardAveragePooling) {
    Pooling pool(2, 2, Pooling::AVERAGE, 1);
    MatrixXd input = create_4x4_input();
    
    MatrixXd output = pool.forward(input);

    // Media dei blocchi 2x2:
    // (1+2+5+6)/4 = 3.5
    // (3+4+7+8)/4 = 5.5
    // ...
    EXPECT_DOUBLE_EQ(output(0, 0), 3.5);
    EXPECT_DOUBLE_EQ(output(0, 1), 5.5);
    EXPECT_DOUBLE_EQ(output(0, 2), 11.5);
    EXPECT_DOUBLE_EQ(output(0, 3), 13.5);
}

// 3. Test Backward - Max Pooling
TEST_F(PoolingLayerTest, BackwardMaxPooling) {
    Pooling pool(2, 2, Pooling::MAX, 1);
    MatrixXd input = create_4x4_input();
    pool.forward(input);

    // Gradiente in ingresso al backward (2x2 appiattito)
    MatrixXd grad_in(1, 4);
    grad_in << 1.0, 2.0, 3.0, 4.0;

    MatrixXd dInput = pool.backward(grad_in, 0.01);

    // Nel max pooling, il gradiente passa solo dove c'era il valore massimo.
    // Il valore 6 era all'indice 5 (0-indexed), l'8 all'indice 7, etc.
    EXPECT_EQ(dInput.cols(), 16);
    EXPECT_DOUBLE_EQ(dInput(0, 5), 1.0);  // Dove c'era il 6
    EXPECT_DOUBLE_EQ(dInput(0, 7), 2.0);  // Dove c'era l'8
    EXPECT_DOUBLE_EQ(dInput(0, 13), 3.0); // Dove c'era il 14
    EXPECT_DOUBLE_EQ(dInput(0, 15), 4.0); // Dove c'era il 16
    EXPECT_DOUBLE_EQ(dInput(0, 0), 0.0);  // Gli altri devono essere zero
}

// 4. Test Eccezioni e Dimensioni
TEST_F(PoolingLayerTest, InvalidInputDimensions) {
    Pooling pool(2, 2, Pooling::MAX, 1);
    
    // Caso 1: dimensioni non impostate
    MatrixXd bad_input(1, 10); 
    EXPECT_THROW(pool.forward(bad_input), std::runtime_error);
    
    // Caso 2: dimensioni impostate ma errate
    pool.set_input_shape(4, 4);  // Si aspetta 16 elementi
    MatrixXd wrong_size(1, 10);
    EXPECT_THROW(pool.forward(wrong_size), ml_exception::DimensionMismatchException);
}

//5. Test Serializzazione
TEST_F(PoolingLayerTest, Serialization) {
    Pooling original(3, 1, Pooling::AVERAGE, 2);
    
    std::stringstream ss;
    original.serialize(ss);
    
    Pooling loaded(2, 2, Pooling::MAX, 1); // Parametri diversi
    loaded.deserialize(ss);
    
    EXPECT_EQ(loaded.get_pool_type(), Pooling::AVERAGE);
    EXPECT_EQ(loaded.get_config(), original.get_config());
    // Verifica che la cache sia pulita dopo il caricamento
    EXPECT_EQ(loaded.get_cache().input.rows(), 0);
}