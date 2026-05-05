// tests/cpp/unit/cache/test_conv_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/conv_cache.h"

using namespace layers;

class ConvCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<ConvCache>();
        
        // Simulate a CNN forward pass
        batch_size = 2;
        input_height = 28;
        input_width = 28;
        input_channels = 3;
        output_height = 26;
        output_width = 26;
        filters = 16;
        kernel_size = 3;
        strides = 1;
        padding = "valid";
        
        int input_size = batch_size * input_height * input_width * input_channels;
        int output_size = batch_size * output_height * output_width * filters;
        int col_size = batch_size * output_height * output_width * 
                       (kernel_size * kernel_size * input_channels);
        
        input_cache.resize(input_size, 1);
        z_cache.resize(output_size, 1);
        output_cache.resize(output_size, 1);
        col_cache.resize(batch_size, output_height * output_width * 
                         (kernel_size * kernel_size * input_channels));
        
        input_cache.setRandom();
        z_cache.setRandom();
        output_cache.setRandom();
        col_cache.setRandom();
    }
    
    std::shared_ptr<ConvCache> cache;
    int batch_size, input_height, input_width, input_channels;
    int output_height, output_width, filters;
    int kernel_size, strides;
    std::string padding;
    
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd z_cache;
    Eigen::MatrixXd output_cache;
    Eigen::MatrixXd col_cache;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(ConvCacheTest, DefaultConstruction) {
    auto new_cache = std::make_shared<ConvCache>();
    
    EXPECT_EQ(new_cache->get_input().rows(), 0);
    EXPECT_EQ(new_cache->get_output().rows(), 0);
    EXPECT_EQ(new_cache->mutable_z().rows(), 0);      // usa mutable_z() invece di get_z()
    EXPECT_EQ(new_cache->mutable_col().rows(), 0);    // usa mutable_col() invece di get_col()
    EXPECT_EQ(new_cache->input_height, 0);
    EXPECT_EQ(new_cache->input_width, 0);
    EXPECT_EQ(new_cache->input_channels, 0);
    EXPECT_EQ(new_cache->kernel_size, 0);
    EXPECT_EQ(new_cache->strides, 1);
    EXPECT_EQ(new_cache->padding, "valid");
    EXPECT_FALSE(new_cache->is_valid());
    EXPECT_EQ(new_cache->get_type(), "ConvCache");
}

// ============================================================================
// Shape Setters Tests
// ============================================================================

TEST_F(ConvCacheTest, SetInputShape) {
    cache->set_input_shape(input_height, input_width, input_channels);
    
    EXPECT_EQ(cache->input_height, input_height);
    EXPECT_EQ(cache->input_width, input_width);
    EXPECT_EQ(cache->input_channels, input_channels);
}

TEST_F(ConvCacheTest, SetOutputShape) {
    cache->set_output_shape(output_height, output_width, filters);
    
    EXPECT_EQ(cache->output_height, output_height);
    EXPECT_EQ(cache->output_width, output_width);
    EXPECT_EQ(cache->filters, filters);
}

TEST_F(ConvCacheTest, SetBatchSize) {
    cache->set_batch_size(batch_size);
    EXPECT_EQ(cache->batch_size, batch_size);
}

TEST_F(ConvCacheTest, SetKernelInfo) {
    cache->set_kernel_info(kernel_size, strides, padding);
    
    EXPECT_EQ(cache->kernel_size, kernel_size);
    EXPECT_EQ(cache->strides, strides);
    EXPECT_EQ(cache->padding, padding);
}

// ============================================================================
// Data Tests
// ============================================================================

TEST_F(ConvCacheTest, SetAndGetInput) {
    cache->mutable_input() = input_cache;
    EXPECT_TRUE(cache->get_input().isApprox(input_cache));
}

TEST_F(ConvCacheTest, SetAndGetZ) {
    cache->mutable_z() = z_cache;
    EXPECT_TRUE(cache->mutable_z().isApprox(z_cache));
}

TEST_F(ConvCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output_cache;
    EXPECT_TRUE(cache->get_output().isApprox(output_cache));
}

TEST_F(ConvCacheTest, SetAndGetCol) {
    cache->mutable_col() = col_cache;
    EXPECT_TRUE(cache->mutable_col().isApprox(col_cache));
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(ConvCacheTest, ClearResetsAllData) {
    // Populate all data
    cache->mutable_input() = input_cache;
    cache->mutable_z() = z_cache;
    cache->mutable_output() = output_cache;
    cache->mutable_col() = col_cache;
    cache->set_input_shape(input_height, input_width, input_channels);
    cache->set_output_shape(output_height, output_width, filters);
    cache->set_batch_size(batch_size);
    cache->set_kernel_info(kernel_size, strides, padding);
    
    EXPECT_TRUE(cache->is_valid());
    
    cache->clear();
    
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->mutable_z().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->mutable_col().rows(), 0);
    EXPECT_EQ(cache->input_height, 0);
    EXPECT_EQ(cache->input_width, 0);
    EXPECT_EQ(cache->input_channels, 0);
    EXPECT_EQ(cache->output_height, 0);
    EXPECT_EQ(cache->output_width, 0);
    EXPECT_EQ(cache->filters, 0);
    EXPECT_EQ(cache->batch_size, 0);
    EXPECT_EQ(cache->kernel_size, 0);
    EXPECT_EQ(cache->strides, 1);
    EXPECT_EQ(cache->padding, "valid");
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(ConvCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    
    EXPECT_TRUE(cache->is_valid());
}

TEST_F(ConvCacheTest, IsValidReturnsFalseWhenInputEmpty) {
    cache->mutable_output() = output_cache;
    
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(ConvCacheTest, IsValidReturnsFalseWhenOutputEmpty) {
    cache->mutable_input() = input_cache;
    
    EXPECT_FALSE(cache->is_valid());
}