// tests/cpp/unit/cache/test_rnn_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/rnn_cache.h"

using namespace layers;

class RNNCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<RNNCache>();
        
        timesteps = 10;
        batch_size = 4;
        input_size = 32;
        hidden_size = 64;
        
        input_cache.resize(batch_size * timesteps, input_size);
        output_cache.resize(batch_size * timesteps, hidden_size);
        
        input_cache.setRandom();
        output_cache.setRandom();
        
        // Setup hidden states for each timestep
        for (int t = 0; t <= timesteps; ++t) {
            hidden_states.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
        }
        
        // Setup pre-activations for each timestep
        for (int t = 0; t < timesteps; ++t) {
            pre_activations.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
        }
    }
    
    std::shared_ptr<RNNCache> cache;
    int timesteps, batch_size, input_size, hidden_size;
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
    std::vector<Eigen::MatrixXd> hidden_states;
    std::vector<Eigen::MatrixXd> pre_activations;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(RNNCacheTest, DefaultConstruction) {
    auto new_cache = std::make_shared<RNNCache>();
    
    EXPECT_EQ(new_cache->get_input().rows(), 0);
    EXPECT_EQ(new_cache->get_output().rows(), 0);
    EXPECT_TRUE(new_cache->hidden_states.empty());
    EXPECT_TRUE(new_cache->pre_activations.empty());
    EXPECT_EQ(new_cache->timesteps, 0);
    EXPECT_EQ(new_cache->batch_size, 0);
    EXPECT_EQ(new_cache->input_size, 0);
    EXPECT_EQ(new_cache->hidden_size, 0);
    EXPECT_FALSE(new_cache->training);
    EXPECT_FALSE(new_cache->is_valid());
    EXPECT_EQ(new_cache->get_type(), "RNNCache");
}

// ============================================================================
// Data Tests
// ============================================================================

TEST_F(RNNCacheTest, SetAndGetInput) {
    cache->mutable_input() = input_cache;
    EXPECT_TRUE(cache->get_input().isApprox(input_cache));
}

TEST_F(RNNCacheTest, SetAndGetOutput) {
    cache->mutable_output() = output_cache;
    EXPECT_TRUE(cache->get_output().isApprox(output_cache));
}

TEST_F(RNNCacheTest, SetAndGetHiddenStates) {
    cache->hidden_states = hidden_states;
    
    const auto& retrieved = cache->hidden_states;
    ASSERT_EQ(retrieved.size(), hidden_states.size());
    for (size_t i = 0; i < hidden_states.size(); ++i) {
        EXPECT_TRUE(retrieved[i].isApprox(hidden_states[i]));
    }
}

TEST_F(RNNCacheTest, SetAndGetPreActivations) {
    cache->pre_activations = pre_activations;
    
    const auto& retrieved = cache->pre_activations;
    ASSERT_EQ(retrieved.size(), pre_activations.size());
    for (size_t i = 0; i < pre_activations.size(); ++i) {
        EXPECT_TRUE(retrieved[i].isApprox(pre_activations[i]));
    }
}

TEST_F(RNNCacheTest, MutableHiddenStates) {
    cache->hidden_states = hidden_states;
    cache->hidden_states[0](0, 0) = 999.0;
    
    EXPECT_DOUBLE_EQ(cache->hidden_states[0](0, 0), 999.0);
}

// ============================================================================
// Parameters Tests
// ============================================================================

TEST_F(RNNCacheTest, SetAndGetParameters) {
    cache->timesteps = timesteps;
    cache->batch_size = batch_size;
    cache->input_size = input_size;
    cache->hidden_size = hidden_size;
    cache->training = true;
    
    EXPECT_EQ(cache->timesteps, timesteps);
    EXPECT_EQ(cache->batch_size, batch_size);
    EXPECT_EQ(cache->input_size, input_size);
    EXPECT_EQ(cache->hidden_size, hidden_size);
    EXPECT_TRUE(cache->training);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(RNNCacheTest, ClearResetsAllData) {
    // Populate all data
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    cache->hidden_states = hidden_states;
    cache->pre_activations = pre_activations;
    cache->timesteps = timesteps;
    cache->batch_size = batch_size;
    cache->input_size = input_size;
    cache->hidden_size = hidden_size;
    cache->training = true;
    
    EXPECT_TRUE(cache->is_valid());
    
    cache->clear();
    
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_TRUE(cache->hidden_states.empty());
    EXPECT_TRUE(cache->pre_activations.empty());
    EXPECT_EQ(cache->timesteps, 0);
    EXPECT_EQ(cache->batch_size, 0);
    EXPECT_EQ(cache->input_size, 0);
    EXPECT_EQ(cache->hidden_size, 0);
    EXPECT_FALSE(cache->training);
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(RNNCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    
    EXPECT_TRUE(cache->is_valid());
}

TEST_F(RNNCacheTest, IsValidReturnsFalseWhenInputEmpty) {
    cache->mutable_output() = output_cache;
    
    EXPECT_FALSE(cache->is_valid());
}

TEST_F(RNNCacheTest, IsValidReturnsFalseWhenOutputEmpty) {
    cache->mutable_input() = input_cache;
    
    EXPECT_FALSE(cache->is_valid());
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(RNNCacheTest, EmptyHiddenStates) {
    cache->hidden_states.clear();
    EXPECT_TRUE(cache->hidden_states.empty());
}

TEST_F(RNNCacheTest, LargeHiddenStates) {
    const int large_size = 100;
    std::vector<Eigen::MatrixXd> large_hidden;
    for (int i = 0; i < large_size; ++i) {
        large_hidden.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
    }
    
    cache->hidden_states = large_hidden;
    EXPECT_EQ(cache->hidden_states.size(), static_cast<size_t>(large_size));
}