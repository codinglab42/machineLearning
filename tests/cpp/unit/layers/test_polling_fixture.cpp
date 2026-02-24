#include <gtest/gtest.h>
#include "components/layers/pooling.h"

using namespace layers;

class PoolingParameterizedTest : public ::testing::TestWithParam<std::tuple<
    int,  // pool_size
    int,  // stride
    Pooling::PoolType,  // type
    int,  // channels
    int,  // input_height
    int   // input_width
>> {
protected:
    void SetUp() override {
        pool_size = std::get<0>(GetParam());
        stride = std::get<1>(GetParam());
        type = std::get<2>(GetParam());
        channels = std::get<3>(GetParam());
        input_height = std::get<4>(GetParam());
        input_width = std::get<5>(GetParam());
        
        // Crea layer e imposta dimensioni
        layer = std::make_unique<Pooling>(pool_size, stride, type, channels);
        layer->set_input_shape(input_height, input_width);
        
        // Calcola dimensioni output attese
        output_height = (input_height - pool_size) / stride + 1;
        output_width = (input_width - pool_size) / stride + 1;
        output_size = channels * output_height * output_width;
        input_size = channels * input_height * input_width;
        
        // Crea input di test
        batch_size = 2;
        input = MatrixXd::Random(batch_size, input_size);
    }
    
    int pool_size;
    int stride;
    Pooling::PoolType type;
    int channels;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int output_size;
    int input_size;
    int batch_size;
    MatrixXd input;
    std::unique_ptr<Pooling> layer;
};

TEST_P(PoolingParameterizedTest, ForwardDimensions) {
    MatrixXd output = layer->forward(input);
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), output_size);
}

TEST_P(PoolingParameterizedTest, BackwardDimensions) {
    MatrixXd output = layer->forward(input);
    MatrixXd gradient = MatrixXd::Random(batch_size, output_size);
    MatrixXd dInput = layer->backward(gradient, 0.01);
    
    EXPECT_EQ(dInput.rows(), batch_size);
    EXPECT_EQ(dInput.cols(), input_size);
}

// Ora possiamo testare sia input quadrati che rettangolari!
INSTANTIATE_TEST_SUITE_P(
    PoolingTests,
    PoolingParameterizedTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 3),           // pool_size
        ::testing::Values(1, 2),               // stride
        ::testing::Values(Pooling::MAX, Pooling::AVERAGE),  // type
        ::testing::Values(1, 3),                // channels
        ::testing::Values(4, 5, 6),             // input_height
        ::testing::Values(4, 5, 6)              // input_width - ORA SUPPORTA RETTANGOLI!
    )
);
