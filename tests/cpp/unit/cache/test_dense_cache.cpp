// tests/cpp/unit/cache/test_dense_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/dense_cache.h"

using namespace layers;

class DenseCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<DenseCache>();
        
        // Setup test matrices
        input.resize(3, 4);
        z.resize(3, 5);
        output.resize(3, 5);
        
        input.setRandom();
        z.setRandom();
        output.setRandom();
    }
    
    std::shared_ptr<DenseCache> cache;
    Eigen::MatrixXd input;
    Eigen::MatrixXd z;
    Eigen::MatrixXd output;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(DenseCacheTest, DefaultConstruction) {
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_input().cols(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->get_output().cols(), 0);
    EXPECT_FALSE(cache->is_valid());
    EXPECT_EQ(cache->get_type(), "DenseCache");
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(DenseCacheTest, ClearResetsState) {
    // Populate cache
    cache->mutable_input() = input;
    cache->mutable_z() = z;           // z_cache è accessibile via mutable_z()
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->is_valid());
    EXPECT_EQ(cache->get_input().rows(), 3);
    
    // Clear
    cache->clear();
    
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_input().cols(), 0);
    EXPECT_EQ(cache->mutable_z().rows(), 0);  // Usa mutable_z() invece di get_z()
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(DenseCacheTest, IsValidReturnsTrueWhenPopulated) {
    cache->mutable_input() = input;
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->is_valid());
}

TEST_F(DenseCacheTest, IsValidReturnsFalseWhenInputEmpty) {
    cache->mutable_output() = output;
    
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(DenseCacheTest, IsValidReturnsFalseWhenOutputEmpty) {
    cache->mutable_input() = input;
    
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Data Access Tests
// ============================================================================

TEST_F(DenseCacheTest, SetAndGetInput) {
    cache->mutable_input() = input;
    
    EXPECT_TRUE(cache->get_input().isApprox(input));
}

TEST_F(DenseCacheTest, SetAndGetZ) {
    cache->mutable_z() = z;
    
    // z_cache è accessibile via mutable_z(), che restituisce una reference
    EXPECT_TRUE(cache->mutable_z().isApprox(z));
}

TEST_F(DenseCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output;
    
    EXPECT_TRUE(cache->get_output().isApprox(output));
}

TEST_F(DenseCacheTest, MutableAccessorsModifyData) {
    cache->mutable_input() = input;
    cache->mutable_input()(0, 0) = 999.0;
    
    EXPECT_DOUBLE_EQ(cache->get_input()(0, 0), 999.0);
}

// ============================================================================
// Const Correctness Tests
// ============================================================================

TEST_F(DenseCacheTest, ConstGettersReturnReference) {
    cache->mutable_input() = input;
    const auto& const_cache = *cache;
    
    // This should compile - get_input is const
    const auto& input_ref = const_cache.get_input();
    EXPECT_EQ(input_ref.rows(), 3);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(DenseCacheTest, EmptyMatricesAreValid) {
    Eigen::MatrixXd empty_input(0, 0);
    Eigen::MatrixXd empty_output(0, 0);
    
    cache->mutable_input() = empty_input;
    cache->mutable_output() = empty_output;
    
    // is_valid checks size() > 0, so empty matrices return false
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(DenseCacheTest, NonMatchingDimensions) {
    cache->mutable_input() = input;      // 3x4
    cache->mutable_output() = output;    // 3x5 - different cols
    
    // is_valid doesn't check dimension consistency,
    // only that matrices are non-empty
    EXPECT_TRUE(cache->is_valid());
}