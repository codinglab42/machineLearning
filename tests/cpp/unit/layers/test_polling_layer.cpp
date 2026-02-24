#include <gtest/gtest.h>
#include "components/layers/pooling.h"
#include <sstream>

using namespace layers;
using Eigen::MatrixXd;

class PoolingLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Crea layer con dimensioni esplicite
        pool_ = std::make_unique<Pooling>(2, 2, Pooling::MAX, 1);
        pool_->set_input_shape(4, 4);  // Input 4x4
    }
    
    // Helper per creare una matrice di input "immagine" 4x4 appiattita
    // Input:
    // 1  2  3  4
    // 5  6  7  8
    // 9 10 11 12
    // 13 14 15 16
    MatrixXd create_4x4_input() {
        MatrixXd input(1, 16);
        for (int i = 0; i < 16; ++i) input(0, i) = i + 1;
        return input;
    }
    
    // Helper per creare input rettangolare 4x5
    MatrixXd create_4x5_input() {
        MatrixXd input(1, 20);
        for (int i = 0; i < 20; ++i) input(0, i) = i + 1;
        return input;
    }
    
    std::unique_ptr<Pooling> pool_;
};

// 1. Test Forward - Max Pooling con input quadrato
TEST_F(PoolingLayerTest, ForwardMaxPooling) {
    MatrixXd input = create_4x4_input();
    MatrixXd output = pool_->forward(input);

    // Risultato atteso:
    // max(1,2,5,6)=6,   max(3,4,7,8)=8
    // max(9,10,13,14)=14, max(11,12,15,16)=16
    ASSERT_EQ(output.cols(), 4); 
    EXPECT_DOUBLE_EQ(output(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(output(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(output(0, 2), 14.0);
    EXPECT_DOUBLE_EQ(output(0, 3), 16.0);
}

// 2. Test Forward - Max Pooling con input rettangolare
TEST_F(PoolingLayerTest, ForwardMaxPoolingRectangular) {
    Pooling pool(2, 2, Pooling::MAX, 1);
    pool.set_input_shape(4, 5);  // Input 4x5
    
    MatrixXd input = create_4x5_input();
    MatrixXd output = pool.forward(input);
    
    // Output atteso: (4-2)/2+1 = 2, (5-2)/2+1 = 2 -> 2x2
    ASSERT_EQ(output.cols(), 4);  // 2x2 = 4 elementi
    
    // Verifica valori (calcolati manualmente)
    EXPECT_DOUBLE_EQ(output(0, 0), 7.0);   // max(1,2,3,6,7,8)
    EXPECT_DOUBLE_EQ(output(0, 1), 9.0);   // max(4,5,9,10)
    EXPECT_DOUBLE_EQ(output(0, 2), 17.0);  // max(11,12,13,16,17,18)
    EXPECT_DOUBLE_EQ(output(0, 3), 19.0);  // max(13,14,18,19)
}

// 3. Test Forward - Average Pooling
TEST_F(PoolingLayerTest, ForwardAveragePooling) {
    Pooling pool(2, 2, Pooling::AVERAGE, 1);
    pool.set_input_shape(4, 4);
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

// 4. Test Backward - Max Pooling
TEST_F(PoolingLayerTest, BackwardMaxPooling) {
    MatrixXd input = create_4x4_input();
    MatrixXd output = pool_->forward(input);

    // Gradiente in ingresso al backward (2x2 appiattito)
    MatrixXd grad_in(1, 4);
    grad_in << 1.0, 2.0, 3.0, 4.0;

    MatrixXd dInput = pool_->backward(grad_in, 0.01);

    // Nel max pooling, il gradiente passa solo dove c'era il valore massimo.
    EXPECT_EQ(dInput.cols(), 16);
    EXPECT_DOUBLE_EQ(dInput(0, 5), 1.0);   // Dove c'era il 6 (indice 5)
    EXPECT_DOUBLE_EQ(dInput(0, 7), 2.0);   // Dove c'era l'8 (indice 7)
    EXPECT_DOUBLE_EQ(dInput(0, 13), 3.0);  // Dove c'era il 14 (indice 13)
    EXPECT_DOUBLE_EQ(dInput(0, 15), 4.0);  // Dove c'era il 16 (indice 15)
    EXPECT_DOUBLE_EQ(dInput(0, 0), 0.0);   // Gli altri devono essere zero
}

// 5. Test senza set_input_shape (dovrebbe fallire)
TEST_F(PoolingLayerTest, MissingInputShape) {
    Pooling pool(2, 2, Pooling::MAX, 1);
    MatrixXd input = create_4x5_input();  // 20 elementi
    
    // Dovrebbe lanciare eccezione perché non abbiamo impostato le dimensioni
    EXPECT_THROW(pool.forward(input), std::runtime_error);
}

// 6. Test dimensioni errate
TEST_F(PoolingLayerTest, WrongInputDimensions) {
    Pooling pool(2, 2, Pooling::MAX, 1);
    pool.set_input_shape(4, 4);  // Si aspetta 16 elementi
    
    MatrixXd wrong_input(1, 20);  // Invece ne arrivano 20
    EXPECT_THROW(pool.forward(wrong_input), ml_exception::DimensionMismatchException);
}

// 7. Test Serializzazione - CON DEBUG
TEST_F(PoolingLayerTest, Serialization) {
    Pooling original(3, 1, Pooling::AVERAGE, 2);
    original.set_input_shape(6, 6);
    
    std::cout << "\n--- Serialization Test ---" << std::endl;
    std::cout << "Original config: " << original.get_config() << std::endl;
    std::cout << "Original input_size: " << original.get_input_size() << std::endl;
    original.print_debug();  // <-- AGGIUNTO
    
    std::stringstream ss;
    original.serialize(ss);
    
    Pooling loaded(2, 2, Pooling::MAX, 1);
    std::cout << "Loaded before deserialize: " << loaded.get_config() << std::endl;
    loaded.print_debug();  // <-- AGGIUNTO
    
    loaded.deserialize(ss);
    
    std::cout << "Loaded after deserialize: " << loaded.get_config() << std::endl;
    std::cout << "Loaded input_size: " << loaded.get_input_size() << std::endl;
    loaded.print_debug();  // <-- AGGIUNTO
    
    EXPECT_EQ(loaded.get_pool_type(), Pooling::AVERAGE);
    EXPECT_EQ(loaded.get_config(), original.get_config());
    EXPECT_EQ(loaded.get_input_size(), 2 * 6 * 6);
}