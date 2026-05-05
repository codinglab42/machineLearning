// tests/cpp/cache/test_batchnorm_cache.cpp
#include <gtest/gtest.h>
#include "components/cache/batchnorm_cache.h"

using namespace layers;

class BatchNormCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<BatchNormCache>();
        
        batch_size = 4;
        features = 8;
        
        input.resize(batch_size, features);
        output.resize(batch_size, features);
        x_centered.resize(batch_size, features);
        x_norm.resize(batch_size, features);
        batch_mean.resize(features);
        batch_var.resize(features);
        inv_std.resize(features);
        
        input.setRandom();
        output.setRandom();
        x_centered.setRandom();
        x_norm.setRandom();
        batch_mean.setRandom();
        batch_var.setRandom();
        inv_std.setRandom();
    }
    
    std::shared_ptr<BatchNormCache> cache;
    int batch_size;
    int features;
    
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd x_centered;
    Eigen::MatrixXd x_norm;
    Eigen::VectorXd batch_mean;
    Eigen::VectorXd batch_var;
    Eigen::VectorXd inv_std;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(BatchNormCacheTest, DefaultConstruction) {
    auto new_cache = std::make_shared<BatchNormCache>();
    
    EXPECT_EQ(new_cache->get_input().rows(), 0);
    EXPECT_EQ(new_cache->get_output().rows(), 0);
    EXPECT_EQ(new_cache->x_centered.rows(), 0);
    EXPECT_EQ(new_cache->x_norm.rows(), 0);
    EXPECT_EQ(new_cache->batch_mean.size(), 0);
    EXPECT_EQ(new_cache->batch_var.size(), 0);
    EXPECT_EQ(new_cache->inv_std.size(), 0);
    EXPECT_FALSE(new_cache->training);
    EXPECT_FALSE(new_cache->is_valid());
    EXPECT_EQ(new_cache->get_type(), "BatchNormCache");
}

// ============================================================================
// Data Tests
// ============================================================================

TEST_F(BatchNormCacheTest, SetAndGetInput) {
    cache->mutable_input() = input;
    EXPECT_TRUE(cache->get_input().isApprox(input));
}

TEST_F(BatchNormCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output;
    EXPECT_TRUE(cache->get_output().isApprox(output));
}

TEST_F(BatchNormCacheTest, SetAndGetXCentered) {
    cache->mutable_x_centered() = x_centered;
    EXPECT_TRUE(cache->x_centered.isApprox(x_centered));
}

TEST_F(BatchNormCacheTest, SetAndGetXNorm) {
    cache->mutable_x_norm() = x_norm;
    EXPECT_TRUE(cache->x_norm.isApprox(x_norm));
}

TEST_F(BatchNormCacheTest, SetAndGetBatchMean) {
    cache->mutable_batch_mean() = batch_mean;
    EXPECT_TRUE(cache->batch_mean.isApprox(batch_mean));
}

TEST_F(BatchNormCacheTest, SetAndGetBatchVar) {
    cache->mutable_batch_var() = batch_var;
    EXPECT_TRUE(cache->batch_var.isApprox(batch_var));
}

TEST_F(BatchNormCacheTest, SetAndGetInvStd) {
    cache->mutable_inv_std() = inv_std;
    EXPECT_TRUE(cache->inv_std.isApprox(inv_std));
}

// ============================================================================
// Training Flag Tests
// ============================================================================

TEST_F(BatchNormCacheTest, TrainingFlag) {
    EXPECT_FALSE(cache->training);
    
    cache->training = true;
    EXPECT_TRUE(cache->training);
    
    cache->clear();
    EXPECT_FALSE(cache->training);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(BatchNormCacheTest, ClearResetsAllData) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    cache->mutable_x_centered() = x_centered;
    cache->mutable_x_norm() = x_norm;
    cache->mutable_batch_mean() = batch_mean;
    cache->mutable_batch_var() = batch_var;
    cache->mutable_inv_std() = inv_std;
    cache->training = true;
    
    EXPECT_TRUE(cache->is_valid());
    
    cache->clear();
    
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->x_centered.rows(), 0);
    EXPECT_EQ(cache->x_norm.rows(), 0);
    EXPECT_EQ(cache->batch_mean.size(), 0);
    EXPECT_EQ(cache->batch_var.size(), 0);
    EXPECT_EQ(cache->inv_std.size(), 0);
    EXPECT_FALSE(cache->training);
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(BatchNormCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->is_valid());
}