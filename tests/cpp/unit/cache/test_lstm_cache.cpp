// tests/cpp/unit/cache/test_lstm_cache.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "components/cache/lstm_cache.h"

using namespace layers;

class LSTMCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_shared<LSTMCache>();
        
        batch_size = 4;
        hidden_size = 64;
        timesteps = 10;
        
        // Setup LSTM-specific data
        for (int t = 0; t <= timesteps; ++t) {
            cell_states.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            hidden_states.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
        }
        
        for (int t = 0; t < timesteps; ++t) {
            input_gates.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            forget_gates.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            output_gates.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            cell_candidates.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            z_i.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            z_f.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            z_o.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
            z_c.push_back(Eigen::MatrixXd::Random(batch_size, hidden_size));
        }
        
        // Setup base RNN data
        input_cache.resize(batch_size * timesteps, 32);
        output_cache.resize(batch_size * timesteps, hidden_size);
        input_cache.setRandom();
        output_cache.setRandom();
    }
    
    std::shared_ptr<LSTMCache> cache;
    int batch_size, hidden_size, timesteps;
    
    std::vector<Eigen::MatrixXd> cell_states;
    std::vector<Eigen::MatrixXd> hidden_states;
    std::vector<Eigen::MatrixXd> input_gates;
    std::vector<Eigen::MatrixXd> forget_gates;
    std::vector<Eigen::MatrixXd> output_gates;
    std::vector<Eigen::MatrixXd> cell_candidates;
    std::vector<Eigen::MatrixXd> z_i;
    std::vector<Eigen::MatrixXd> z_f;
    std::vector<Eigen::MatrixXd> z_o;
    std::vector<Eigen::MatrixXd> z_c;
    
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd output_cache;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(LSTMCacheTest, DefaultConstruction) {
    auto new_cache = std::make_shared<LSTMCache>();
    
    EXPECT_TRUE(new_cache->cell_states.empty());
    EXPECT_TRUE(new_cache->input_gates.empty());
    EXPECT_TRUE(new_cache->forget_gates.empty());
    EXPECT_TRUE(new_cache->output_gates.empty());
    EXPECT_TRUE(new_cache->cell_candidates.empty());
    EXPECT_TRUE(new_cache->z_i.empty());
    EXPECT_TRUE(new_cache->z_f.empty());
    EXPECT_TRUE(new_cache->z_o.empty());
    EXPECT_TRUE(new_cache->z_c.empty());
    EXPECT_TRUE(new_cache->hidden_states.empty());
    EXPECT_EQ(new_cache->get_type(), "LSTMCache");
}

// ============================================================================
// Cell States Tests
// ============================================================================

TEST_F(LSTMCacheTest, SetAndGetCellStates) {
    cache->cell_states = cell_states;
    
    ASSERT_EQ(cache->cell_states.size(), cell_states.size());
    for (size_t i = 0; i < cell_states.size(); ++i) {
        EXPECT_TRUE(cache->cell_states[i].isApprox(cell_states[i]));
    }
}

// ============================================================================
// Gates Tests
// ============================================================================

TEST_F(LSTMCacheTest, SetAndGetInputGates) {
    cache->input_gates = input_gates;
    
    ASSERT_EQ(cache->input_gates.size(), input_gates.size());
    for (size_t i = 0; i < input_gates.size(); ++i) {
        EXPECT_TRUE(cache->input_gates[i].isApprox(input_gates[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetForgetGates) {
    cache->forget_gates = forget_gates;
    
    ASSERT_EQ(cache->forget_gates.size(), forget_gates.size());
    for (size_t i = 0; i < forget_gates.size(); ++i) {
        EXPECT_TRUE(cache->forget_gates[i].isApprox(forget_gates[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetOutputGates) {
    cache->output_gates = output_gates;
    
    ASSERT_EQ(cache->output_gates.size(), output_gates.size());
    for (size_t i = 0; i < output_gates.size(); ++i) {
        EXPECT_TRUE(cache->output_gates[i].isApprox(output_gates[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetCellCandidates) {
    cache->cell_candidates = cell_candidates;
    
    ASSERT_EQ(cache->cell_candidates.size(), cell_candidates.size());
    for (size_t i = 0; i < cell_candidates.size(); ++i) {
        EXPECT_TRUE(cache->cell_candidates[i].isApprox(cell_candidates[i]));
    }
}

// ============================================================================
// Z Values Tests
// ============================================================================

TEST_F(LSTMCacheTest, SetAndGetZI) {
    cache->z_i = z_i;
    
    ASSERT_EQ(cache->z_i.size(), z_i.size());
    for (size_t i = 0; i < z_i.size(); ++i) {
        EXPECT_TRUE(cache->z_i[i].isApprox(z_i[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetZF) {
    cache->z_f = z_f;
    
    ASSERT_EQ(cache->z_f.size(), z_f.size());
    for (size_t i = 0; i < z_f.size(); ++i) {
        EXPECT_TRUE(cache->z_f[i].isApprox(z_f[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetZO) {
    cache->z_o = z_o;
    
    ASSERT_EQ(cache->z_o.size(), z_o.size());
    for (size_t i = 0; i < z_o.size(); ++i) {
        EXPECT_TRUE(cache->z_o[i].isApprox(z_o[i]));
    }
}

TEST_F(LSTMCacheTest, SetAndGetZC) {
    cache->z_c = z_c;
    
    ASSERT_EQ(cache->z_c.size(), z_c.size());
    for (size_t i = 0; i < z_c.size(); ++i) {
        EXPECT_TRUE(cache->z_c[i].isApprox(z_c[i]));
    }
}

// ============================================================================
// Inheritance Tests (from RNNCache)
// ============================================================================

TEST_F(LSTMCacheTest, InheritsRNNCacheHiddenStates) {
    cache->hidden_states = hidden_states;
    
    const auto& retrieved = cache->hidden_states;
    ASSERT_EQ(retrieved.size(), hidden_states.size());
    for (size_t i = 0; i < hidden_states.size(); ++i) {
        EXPECT_TRUE(retrieved[i].isApprox(hidden_states[i]));
    }
}

TEST_F(LSTMCacheTest, InheritsRNNCacheInputOutput) {
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    
    EXPECT_TRUE(cache->get_input().isApprox(input_cache));
    EXPECT_TRUE(cache->get_output().isApprox(output_cache));
}

TEST_F(LSTMCacheTest, InheritsRNNCacheParameters) {
    cache->timesteps = timesteps;
    cache->batch_size = batch_size;
    cache->hidden_size = hidden_size;
    cache->training = true;
    
    EXPECT_EQ(cache->timesteps, timesteps);
    EXPECT_EQ(cache->batch_size, batch_size);
    EXPECT_EQ(cache->hidden_size, hidden_size);
    EXPECT_TRUE(cache->training);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST_F(LSTMCacheTest, ClearResetsAllData) {
    // Populate all data
    cache->cell_states = cell_states;
    cache->input_gates = input_gates;
    cache->forget_gates = forget_gates;
    cache->output_gates = output_gates;
    cache->cell_candidates = cell_candidates;
    cache->z_i = z_i;
    cache->z_f = z_f;
    cache->z_o = z_o;
    cache->z_c = z_c;
    cache->hidden_states = hidden_states;
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    cache->timesteps = timesteps;
    cache->training = true;
    
    cache->clear();
    
    EXPECT_TRUE(cache->cell_states.empty());
    EXPECT_TRUE(cache->input_gates.empty());
    EXPECT_TRUE(cache->forget_gates.empty());
    EXPECT_TRUE(cache->output_gates.empty());
    EXPECT_TRUE(cache->cell_candidates.empty());
    EXPECT_TRUE(cache->z_i.empty());
    EXPECT_TRUE(cache->z_f.empty());
    EXPECT_TRUE(cache->z_o.empty());
    EXPECT_TRUE(cache->z_c.empty());
    EXPECT_TRUE(cache->hidden_states.empty());
    EXPECT_EQ(cache->get_input().rows(), 0);
    EXPECT_EQ(cache->get_output().rows(), 0);
    EXPECT_EQ(cache->timesteps, 0);
    EXPECT_FALSE(cache->training);
}

// ============================================================================
// Validity Tests
// ============================================================================

TEST_F(LSTMCacheTest, IsValidWhenInputAndOutputPopulated) {
    cache->mutable_input() = input_cache;
    cache->mutable_output() = output_cache;
    
    EXPECT_TRUE(cache->is_valid());
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(LSTMCacheTest, EmptyGates) {
    EXPECT_TRUE(cache->input_gates.empty());
    EXPECT_TRUE(cache->forget_gates.empty());
    EXPECT_TRUE(cache->output_gates.empty());
}

TEST_F(LSTMCacheTest, EmptyZValues) {
    EXPECT_TRUE(cache->z_i.empty());
    EXPECT_TRUE(cache->z_f.empty());
    EXPECT_TRUE(cache->z_o.empty());
    EXPECT_TRUE(cache->z_c.empty());
}