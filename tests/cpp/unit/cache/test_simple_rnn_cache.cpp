// tests/cpp/cache/test_simple_rnn_cache.cpp
#include <gtest/gtest.h>
#include "components/cache/simple_rnn_cache.h"

using namespace layers;

class SimpleRNNCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<SimpleRNNCache>();
        
        batch_size = 4;
        hidden_size = 64;
        timesteps = 10;
        
        // Setup z_values for each timestep
        for (int t = 0; t < timesteps; ++t) {
            z_values.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
        }
        
        // Setup base RNN data
        input_cache.resize(batch_size * timesteps, 32);
        output_cache.resize(batch_size * timesteps, hidden_size);
        input_cache.setRandom();
        output_cache.setRandom();
    }
    
    std::shared_ptr<SimpleRNNCache> cache;
    int batch_size, hidden_size, timesteps;
    std::vector<Eigen::MatrixXd> z_values;
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(SimpleRNNCacheTest, DefaultConstruction) {
    auto new_cache = std::make_shared<SimpleRNNCache>();
    
    EXPECT_TRUE(new_cache->z_values.empty());
    EXPECT_EQ(new_cache->get_type(), "SimpleRNNCache");
}

// ============================================================================
// Z Values Tests
// ============================================================================

TEST_F(SimpleRNNCacheTest, SetAndGetZValues) {
    cache->z_values = z_values;
    
    ASSERT_EQ(cache->z_values.size(), z_values.size());
    for (size_t i = 0; i < z_values.size(); ++i) {
        EXPECT_TRUE(cache->z_values[i].isApprox(z_values[i]));
    }
}

TEST_F(SimpleRNNCacheTest, AddZValue) {
    Eigen::MatrixXd new_z = Eigen::MatrixXd::Random(batch_size, hidden_size);
    
    cache->z_values.push_back(new_z);
    
    ASSERT_EQ(cache->z_values.size(), 1);
    EXPECT_TRUE(cache->z_values[0].isApprox(new_z));
}

// ============================================================================
// Inheritance Tests (from RNNCache)
// ============================================================================

TEST_F(SimpleRNNCacheTest, InheritsRNNCacheData) {
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    cache->timesteps = timesteps;
    cache->batch_size = batch_size;
    cache->hidden_size = hidden_size;
    cache->training = true;
    
    EXPECT_TRUE(cache->get_input().isApprox(input_cache));
    EXPECT_TRUE(cache->get_output().isApprox(output_cache));
    EXPECT_EQ(cache->timesteps, timesteps);
    EXPECT_EQ(cache->batch_size, batch_size);
    EXPECT_EQ(cache->hidden_size, hidden_size);
    EXPECT_TRUE(cache->training);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(SimpleRNNCacheTest, ClearResetsZValues) {
    cache->z_values = z_values;
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    
    EXPECT_FALSE(cache->z_values.empty());
    
    cache->clear();
    
    EXPECT_TRUE(cache->z_values.empty());
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
}