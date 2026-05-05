// tests/cpp/unit/cache/test_pooling_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/pooling_cache.h"

using namespace layers;

class PoolingCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<PoolingCache>();
        
        input.resize(2, 16);   // batch=2, 16 features (4x4 image, 1 channel)
        output.resize(2, 4);   // after 2x2 pooling
        input.setRandom();
        output.setRandom();
    }
    
    std::shared_ptr<PoolingCache> cache;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(PoolingCacheTest, DefaultConstruction) {
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_TRUE(cache->get_max_indices().empty());
    EXPECT_FALSE(cache->get_training());
    EXPECT_FALSE(cache->is_valid());
    EXPECT_EQ(cache->get_type(), "PoolingCache");
}

// ============================================================================
// Input/Output Tests
// ============================================================================

TEST_F(PoolingCacheTest, SetAndGetInput) {
    cache->set_input(input);
    EXPECT_TRUE(cache->get_input().isApprox(input));
}

TEST_F(PoolingCacheTest, SetAndGetOutput) {
    cache->set_output(output);
    EXPECT_TRUE(cache->get_output().isApprox(output));
}

TEST_F(PoolingCacheTest, MutableInputModifiesData) {
    cache->mutable_input() = input;
    cache->mutable_input()(0, 0) = 999.0;
    
    EXPECT_DOUBLE_EQ(cache->get_input()(0, 0), 999.0);
}

// ============================================================================
// Input Shape Tests
// ============================================================================

TEST_F(PoolingCacheTest, SetInputShape) {
    EXPECT_NO_THROW(cache->set_input_shape(32, 32, 3));
}

TEST_F(PoolingCacheTest, SetInputShapeMultipleTimes) {
    EXPECT_NO_THROW(cache->set_input_shape(28, 28, 1));
    EXPECT_NO_THROW(cache->set_input_shape(32, 32, 3));
    EXPECT_NO_THROW(cache->set_input_shape(64, 64, 3));
}

TEST_F(PoolingCacheTest, ClearResetsInputShapeViaSetInputShape) {
    cache->set_input_shape(32, 32, 3);
    cache->clear();
    
    EXPECT_NO_THROW(cache->set_input_shape(16, 16, 1));
}

// ============================================================================
// MaxIndex Tests
// ============================================================================

TEST_F(PoolingCacheTest, AddAndRetrieveMaxIndices) {
    cache->add_max_index(0, 0, 0, 0, 1, 2);
    cache->add_max_index(0, 1, 0, 0, 3, 4);
    cache->add_max_index(1, 0, 1, 1, 5, 6);
    cache->add_max_index(1, 1, 1, 1, 7, 8);
    
    const auto& indices = cache->get_max_indices();
    
    ASSERT_EQ(indices.size(), 4);
    
    EXPECT_EQ(indices[0].batch, 0);
    EXPECT_EQ(indices[0].channel, 0);
    EXPECT_EQ(indices[0].output_row, 0);
    EXPECT_EQ(indices[0].output_col, 0);
    EXPECT_EQ(indices[0].input_h, 1);
    EXPECT_EQ(indices[0].input_w, 2);
    
    EXPECT_EQ(indices[1].batch, 0);
    EXPECT_EQ(indices[1].channel, 1);
    EXPECT_EQ(indices[1].input_h, 3);
    EXPECT_EQ(indices[1].input_w, 4);
    
    EXPECT_EQ(indices[2].batch, 1);
    EXPECT_EQ(indices[2].channel, 0);
    EXPECT_EQ(indices[2].output_row, 1);
    EXPECT_EQ(indices[2].output_col, 1);
    EXPECT_EQ(indices[2].input_h, 5);
    EXPECT_EQ(indices[2].input_w, 6);
    
    EXPECT_EQ(indices[3].batch, 1);
    EXPECT_EQ(indices[3].channel, 1);
    EXPECT_EQ(indices[3].output_row, 1);
    EXPECT_EQ(indices[3].output_col, 1);
    EXPECT_EQ(indices[3].input_h, 7);
    EXPECT_EQ(indices[3].input_w, 8);
}

TEST_F(PoolingCacheTest, MutableMaxIndices) {
    cache->add_max_index(0, 0, 0, 0, 1, 2);
    
    auto& indices = cache->mutable_max_indices();
    indices[0].input_h = 10;
    indices[0].input_w = 20;
    
    const auto& retrieved = cache->get_max_indices();
    EXPECT_EQ(retrieved[0].input_h, 10);
    EXPECT_EQ(retrieved[0].input_w, 20);
}

TEST_F(PoolingCacheTest, ClearResetsIndices) {
    cache->add_max_index(0, 0, 0, 0, 1, 2);
    cache->add_max_index(0, 1, 0, 0, 3, 4);
    
    EXPECT_EQ(cache->get_max_indices().size(), 2);
    
    cache->clear();
    
    EXPECT_TRUE(cache->get_max_indices().empty());
}

// ============================================================================
// Training Flag Tests
// ============================================================================

TEST_F(PoolingCacheTest, SetAndGetTraining) {
    cache->set_training(true);
    EXPECT_TRUE(cache->get_training());
    
    cache->set_training(false);
    EXPECT_FALSE(cache->get_training());
}

TEST_F(PoolingCacheTest, ClearResetsTrainingFlag) {
    cache->set_training(true);
    cache->clear();
    
    EXPECT_FALSE(cache->get_training());
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(PoolingCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->set_input(input);
    cache->set_output(output);
    
    EXPECT_TRUE(cache->is_valid());
}

TEST_F(PoolingCacheTest, IsValidReturnsFalseWhenInputEmpty) {
    cache->set_output(output);
    
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(PoolingCacheTest, IsValidReturnsFalseWhenOutputEmpty) {
    cache->set_input(input);
    
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(PoolingCacheTest, ClearResetsAllData) {
    cache->set_input(input);
    cache->set_output(output);
    cache->add_max_index(0, 0, 0, 0, 1, 2);
    cache->set_training(true);
    cache->set_input_shape(32, 32, 3);
    
    EXPECT_TRUE(cache->is_valid());
    EXPECT_FALSE(cache->get_max_indices().empty());
    
    cache->clear();
    
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_TRUE(cache->get_max_indices().empty());
    EXPECT_FALSE(cache->get_training());
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(PoolingCacheTest, MultipleMaxIndicesForSamePosition) {
    cache->add_max_index(0, 0, 0, 0, 1, 2);
    cache->add_max_index(0, 0, 0, 0, 1, 3);
    
    const auto& indices = cache->get_max_indices();
    ASSERT_EQ(indices.size(), 2);
    EXPECT_EQ(indices[0].input_w, 2);
    EXPECT_EQ(indices[1].input_w, 3);
}

TEST_F(PoolingCacheTest, MaxIndexDefaultConstructor) {
    PoolingCache::MaxIndex default_idx;
    
    EXPECT_EQ(default_idx.batch, 0);
    EXPECT_EQ(default_idx.channel, 0);
    EXPECT_EQ(default_idx.output_row, 0);
    EXPECT_EQ(default_idx.output_col, 0);
    EXPECT_EQ(default_idx.input_h, 0);
    EXPECT_EQ(default_idx.input_w, 0);
}

TEST_F(PoolingCacheTest, MaxIndexParameterizedConstructor) {
    PoolingCache::MaxIndex idx(1, 2, 3, 4, 5, 6);
    
    EXPECT_EQ(idx.batch, 1);
    EXPECT_EQ(idx.channel, 2);
    EXPECT_EQ(idx.output_row, 3);
    EXPECT_EQ(idx.output_col, 4);
    EXPECT_EQ(idx.input_h, 5);
    EXPECT_EQ(idx.input_w, 6);
}

// ============================================================================
// Large Data Tests
// ============================================================================

TEST_F(PoolingCacheTest, AddManyMaxIndices) {
    const int num_indices = 1000;
    for (int i = 0; i < num_indices; ++i) {
        cache->add_max_index(i % 4, i % 8, i % 16, i % 16, i, i);
    }
    
    const auto& indices = cache->get_max_indices();
    EXPECT_EQ(indices.size(), static_cast<size_t>(num_indices));
    EXPECT_EQ(indices.back().input_h, num_indices - 1);
}

// ============================================================================
// Input/Output Dimension Consistency
// ============================================================================

TEST_F(PoolingCacheTest, InputOutputDimensionConsistency) {
    cache->set_input_shape(28, 28, 3);
    cache->set_input(input);
    cache->set_output(output);
    
    EXPECT_TRUE(cache->is_valid());
}