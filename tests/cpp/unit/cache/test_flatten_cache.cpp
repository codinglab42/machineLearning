// tests/cpp/unit/cache/test_flatten_cache.cpp
#include <gtest/gtest.h>
#include "components/cache/flatten_cache.h"

using namespace layers;

class FlattenCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<FlattenCache>();
        
        input.resize(4, 6);  // 4 samples, 6 features
        output.resize(4, 6);
        input.setRandom();
        output.setRandom();
        
        original_shape = {4, 2, 3};  // batch=4, height=2, width=3
    }
    
    std::shared_ptr<FlattenCache> cache;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    std::vector<int> original_shape;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(FlattenCacheTest, DefaultConstruction) {
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_TRUE(cache->original_shape.empty());
    EXPECT_FALSE(cache->is_valid());
    EXPECT_EQ(cache->get_type(), "FlattenCache");
}

// ============================================================================
// Original Shape Tests
// ============================================================================

TEST_F(FlattenCacheTest, SetAndGetOriginalShape) {
    cache->mutable_shape() = original_shape;
    
    const auto& retrieved = cache->original_shape;
    ASSERT_EQ(retrieved.size(), 3);
    EXPECT_EQ(retrieved[0], 4);
    EXPECT_EQ(retrieved[1], 2);
    EXPECT_EQ(retrieved[2], 3);
}

TEST_F(FlattenCacheTest, ClearResetsOriginalShape) {
    cache->mutable_shape() = original_shape;
    cache->clear();
    
    EXPECT_TRUE(cache->original_shape.empty());
}

// ============================================================================
// Data Tests
// ============================================================================

TEST_F(FlattenCacheTest, SetAndGetInput) {
    cache->mutable_input() = input;
    EXPECT_TRUE(cache->get_input().isApprox(input));
}

TEST_F(FlattenCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output;
    EXPECT_TRUE(cache->get_output().isApprox(output));
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(FlattenCacheTest, IsValidWhenPopulated) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->is_valid());
}