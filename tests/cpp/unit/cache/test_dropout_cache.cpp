// tests/cpp/unit/cache/test_dropout_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/dropout_cache.h"

using namespace layers;

class DropoutCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<DropoutCache>();
        
        input.resize(5, 4);
        output.resize(5, 4);
        mask.resize(5, 4);
        
        input.setRandom();
        output.setRandom();
        mask.setRandom();
    }
    
    std::shared_ptr<DropoutCache> cache;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd mask;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(DropoutCacheTest, DefaultConstruction) {
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->mutable_mask().rows(), 0);
    EXPECT_FALSE(cache->training);
    EXPECT_FALSE(cache->is_valid());
    EXPECT_EQ(cache->get_type(), "DropoutCache");
}

// ============================================================================
// Training Flag Tests
// ============================================================================

TEST_F(DropoutCacheTest, SetAndGetTraining) {
    EXPECT_FALSE(cache->training);
    
    cache->training = true;
    EXPECT_TRUE(cache->training);
    
    cache->training = false;
    EXPECT_FALSE(cache->training);
}

TEST_F(DropoutCacheTest, ClearResetsTrainingFlag) {
    cache->training = true;
    cache->clear();
    
    EXPECT_FALSE(cache->training);
}

// ============================================================================
// Mask Tests
// ============================================================================

TEST_F(DropoutCacheTest, SetAndGetMask) {
    cache->mutable_mask() = mask;
    
    EXPECT_TRUE(cache->mutable_mask().isApprox(mask));
}

TEST_F(DropoutCacheTest, ClearResetsMask) {
    cache->mutable_mask() = mask;
    cache->clear();
    
    EXPECT_EQ(cache->mutable_mask().rows(), 0);
}

// ============================================================================
// Data Tests
// ============================================================================

TEST_F(DropoutCacheTest, SetAndGetInput) {
    cache->mutable_input() = input;
    EXPECT_TRUE(cache->get_input().isApprox(input));
}

TEST_F(DropoutCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output;
    EXPECT_TRUE(cache->get_output().isApprox(output));
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(DropoutCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->is_valid());
}

TEST_F(DropoutCacheTest, IsValidReturnsFalseWhenInputEmpty) {
    cache->mutable_output() = output;
    
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(DropoutCacheTest, IsValidReturnsFalseWhenOutputEmpty) {
    cache->mutable_input() = input;
    
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(DropoutCacheTest, ClearResetsAllData) {
    // Populate all data
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    cache->mutable_mask() = mask;
    cache->training = true;
    
    EXPECT_TRUE(cache->is_valid());
    
    // Clear
    cache->clear();
    
    // Verify all data is reset
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->mutable_mask().rows(), 0);
    EXPECT_FALSE(cache->training);
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(DropoutCacheTest, MaskHasSameDimensionsAsInput) {
    cache->mutable_input() = input;
    cache->mutable_mask() = mask;
    
    EXPECT_EQ(cache->get_input().rows(), cache->mutable_mask().rows());
    EXPECT_EQ(cache->get_input().cols(), cache->mutable_mask().cols());
}

TEST_F(DropoutCacheTest, OutputHasSameDimensionsAsInput) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    
    EXPECT_EQ(cache->get_input().rows(), cache->get_output().rows());
    EXPECT_EQ(cache->get_input().cols(), cache->get_output().cols());
}