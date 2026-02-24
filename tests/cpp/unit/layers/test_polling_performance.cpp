#include <gtest/gtest.h>
#include <chrono>
#include "components/layers/pooling.h"

using namespace layers;

class PoolingPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        batch_size = 32;
        channels = 64;
        input_height = 32;
        input_width = 32;
        pool_size = 2;
        stride = 2;
        
        input_size = channels * input_height * input_width;
        output_size = channels * ((input_height - pool_size) / stride + 1) * 
                               ((input_width - pool_size) / stride + 1);
        
        input = MatrixXd::Random(batch_size, input_size);
    }
    
    int batch_size;
    int channels;
    int input_height;
    int input_width;
    int pool_size;
    int stride;
    int input_size;
    int output_size;
    MatrixXd input;
};

TEST_F(PoolingPerformanceTest, MaxPoolingForwardTime) {
    Pooling layer(pool_size, stride, Pooling::MAX, channels);
    
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd output = layer.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Max pooling forward time: " << duration.count() << " ms" << std::endl;
    EXPECT_LT(duration.count(), 1000);  // Dovrebbe essere sotto 1 secondo
}

TEST_F(PoolingPerformanceTest, AveragePoolingForwardTime) {
    Pooling layer(pool_size, stride, Pooling::AVERAGE, channels);
    
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd output = layer.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Average pooling forward time: " << duration.count() << " ms" << std::endl;
    EXPECT_LT(duration.count(), 1000);
}

TEST_F(PoolingPerformanceTest, BackwardTime) {
    Pooling layer(pool_size, stride, Pooling::MAX, channels);
    
    MatrixXd output = layer.forward(input);
    MatrixXd gradient = MatrixXd::Random(batch_size, output_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd dInput = layer.backward(gradient, 0.01);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Pooling backward time: " << duration.count() << " ms" << std::endl;
    EXPECT_LT(duration.count(), 1000);
}