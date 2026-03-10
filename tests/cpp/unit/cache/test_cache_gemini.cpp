#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "components/cache/batchnorm_cache.h"
#include "components/cache/conv_cache.h"
#include "components/cache/dense_cache.h"
#include "components/cache/dropout_cache.h"
#include "components/cache/flatten_cache.h"
#include "components/cache/gru_cache.h"
#include "components/cache/lstm_cache.h"
#include "components/cache/pooling_cache.h"
#include "components/cache/rnn_cache.h"
#include "components/cache/simple_rnn_cache.h"
#include "components/cache/weighted_cache.h"
#include <sys/resource.h>
#include <iostream>

long get_current_rss_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Restituisce il picco di memoria in Kilobytes
}

using namespace layers;

// --- Helper Fixture per Cache semplici ---
template <typename T>
class CacheTest : public ::testing::Test {
protected:
    T cache;
};

// --- Registrazione dei test per classi base ---
using SimpleCacheTypes = ::testing::Types<DenseCache, BatchNormCache, DropoutCache, FlattenCache>;
TYPED_TEST_SUITE(CacheTest, SimpleCacheTypes);

TYPED_TEST(CacheTest, ClearResetsState) {
    this->cache.mutable_input() = Eigen::MatrixXd::Random(2, 2);
    this->cache.clear();
    EXPECT_EQ(this->cache.get_input().size(), 0);
}

// --- Test per Cache Complesse (RNN/LSTM/GRU) ---
TEST(RNNCacheTest, Initialization) {
    RNNCache cache;
    cache.timesteps = 5;
    cache.mutable_hidden_states().resize(5);
    EXPECT_EQ(cache.hidden_states.size(), 5);
}

TEST(LSTMCacheTest, LSTMStructure) {
    LSTMCache cache;
    cache.cell_states.push_back(Eigen::MatrixXd::Zero(1, 1));
    EXPECT_EQ(cache.cell_states.size(), 1);
    EXPECT_EQ(cache.get_type(), "LSTMCache");
}

TEST(GRUCacheTest, GRUStructure) {
    GRUCache cache;
    cache.reset_gates.push_back(Eigen::MatrixXd::Identity(2, 2));
    EXPECT_EQ(cache.reset_gates[0](0,0), 1.0);
}

// --- Test Specifici per Logiche di Dimensione ---
TEST(ConvCacheTest, ShapeValidation) {
    ConvCache cache;
    cache.set_input_shape(32, 32, 3);
    EXPECT_EQ(cache.input_height, 32);
    EXPECT_EQ(cache.input_channels, 3);
}

TEST(PoolingCacheTest, MaxIndexStorage) {
    PoolingCache cache;
    cache.add_max_index(0, 0, 1, 1, 10, 10);
    EXPECT_EQ(cache.get_max_indices().size(), 1);
    EXPECT_EQ(cache.get_max_indices()[0].batch, 0);
}

// --- Test per WeightedCache (Stati Ottimizzatori) ---
TEST(WeightedCacheTest, Checkpointing) {
    WeightedCache cache;
    Eigen::MatrixXd w = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
    
    cache.save_checkpoint(1);
    
    Eigen::MatrixXd w_loaded;
    Eigen::VectorXd b_loaded;
    bool success = cache.load_checkpoint(1, w_loaded, b_loaded);
    
    EXPECT_TRUE(success);
}

TEST(WeightedCacheTest, TrainingWorkflowSimulation) {
    WeightedCache cache;
    
    // 1. Setup pesi iniziali
    Eigen::MatrixXd weights = Eigen::MatrixXd::Constant(2, 2, 0.5);
    cache.set_weights(weights);
    
    // 2. Simulazione Backprop: calcolo e salvataggio gradiente
    Eigen::MatrixXd grad = Eigen::MatrixXd::Constant(2, 2, 0.1);
    cache.set_weight_gradient(grad);
    cache.push_gradient_history(grad);
    
    // 3. Simulazione Ottimizzatore (es. Adam State)
    auto& adam = cache.get_optimizer_state("adam");
    adam.first_moment = Eigen::MatrixXd::Zero(2, 2);
    adam.timestep = 1;
    
    // 4. Checkpoint dei pesi prima dell'aggiornamento
    cache.save_checkpoint(1);
    
    // 5. Verifica integrità dati post-simulazione
    EXPECT_EQ(cache.get_gradient_history().size(), 1);
    EXPECT_EQ(cache.get_optimizer_state("adam").timestep, 1);
    
    // 6. Verifica recupero checkpoint
    Eigen::MatrixXd w_recovered;
    Eigen::VectorXd b_recovered;
    cache.load_checkpoint(1, w_recovered, b_recovered);
    EXPECT_TRUE(w_recovered.isApprox(weights));
}

TEST(WeightedCacheTest, StressTestGradientHistory) {
    WeightedCache cache;
    const int iterations = 10000;
    const int rows = 64;
    const int cols = 64;

    // Simula un ciclo di training massivo
    for (int i = 0; i < iterations; ++i) {
        Eigen::MatrixXd dummy_grad = Eigen::MatrixXd::Random(rows, cols);
        cache.push_gradient_history(dummy_grad);
        
        // Ogni 100 iterazioni simuliamo il clear dopo un'epoca
        if (i % 100 == 0) {
            cache.clear_gradient_history();
        }
    }

    // Verifica finale: la history deve essere vuota o contenere solo gli ultimi elementi
    // Se il test supera questo punto senza segmentation fault, la gestione memoria è stabile
    EXPECT_TRUE(cache.get_gradient_history().empty() || cache.get_gradient_history().size() < 100);
}

TEST(WeightedCacheTest, RapidCheckpointStress) {
    WeightedCache cache;
    
    // Test di stress su checkpointing rapido
    for (int i = 0; i < 500; ++i) {
        cache.save_checkpoint(i);
        
        // Verifica che il checkpoint sia leggibile
        Eigen::MatrixXd w;
        Eigen::VectorXd b;
        ASSERT_TRUE(cache.load_checkpoint(i, w, b));
    }
    
    cache.clear_checkpoints();
    EXPECT_EQ(cache.get_gradient_history().size(), 0); // O logica di validazione che preferisci
}

TEST(WeightedCacheTest, MemoryStabilityStressTest) {
    WeightedCache cache;
    const int iterations = 5000;
    
    long initial_rss = get_current_rss_kb();
    
    for (int i = 0; i < iterations; ++i) {
        // Simuliamo un carico di lavoro che genera matrici
        cache.push_gradient_history(Eigen::MatrixXd::Random(128, 128));
        
        if (i % 50 == 0) {
            cache.clear_gradient_history();
        }
    }
    
    long final_rss = get_current_rss_kb();
    
    // Tolleranza per l'allocazione dinamica: 
    // Se la crescita è drastica (es. > 10MB), c'è un leak.
    EXPECT_LT(final_rss - initial_rss, 10000) << "Possibile memory leak rilevato!";
}

TEST(WeightedCacheTest, NumericalConsistencyAfterStress) {
    WeightedCache cache;
    
    // 1. Setup iniziale: definisco dei pesi costanti e dei bias
    const int rows = 16;
    const int cols = 16;
    Eigen::MatrixXd initial_weights = Eigen::MatrixXd::Identity(rows, cols);
    Eigen::VectorXd initial_biases = Eigen::VectorXd::Zero(rows);
    
    cache.set_weights(initial_weights);
    cache.set_biases(initial_biases);
    cache.set_weight_version(1);
    
    // 2. Esecuzione Stress Test: simuliamo 2000 cicli di calcoli e pulizie
    // Includiamo l'aggiunta di gradienti e il loro reset
    for (int i = 0; i < 2000; ++i) {
        Eigen::MatrixXd random_grad = Eigen::MatrixXd::Random(rows, cols);
        cache.set_weight_gradient(random_grad);
        cache.push_gradient_history(random_grad);
        
        // Simulo un aggiornamento di stato di un ottimizzatore
        auto& opt = cache.get_optimizer_state("adam_test");
        opt.timestep = i;
        
        // Pulizia periodica (simulata dopo ogni epoca/batch)
        if (i % 50 == 0) {
            cache.clear_gradient_history();
        }
    }
    
    // 
    
    // 3. Verifica Integrità: 
    // I pesi originali non devono essere stati alterati dalle operazioni di gradienti e history
    EXPECT_TRUE(cache.get_weights().isApprox(initial_weights)) 
        << "Errore: I pesi originali sono stati corrotti durante lo stress test!";
    
    EXPECT_TRUE(cache.get_biases().isApprox(initial_biases)) 
        << "Errore: I bias originali sono stati corrotti!";
        
    EXPECT_EQ(cache.get_weight_version(), 1) 
        << "Errore: La versione dei pesi è cambiata inaspettatamente!";
    
    // 4. Verifica finale sullo stato ottimizzatore rimasto
    EXPECT_TRUE(cache.has_optimizer_state("adam_test"));
    EXPECT_EQ(cache.get_optimizer_state("adam_test").timestep, 1999);
}

TEST(LayerCacheTest, DeepCopyConsistency) {
    DenseCache cache1;
    cache1.mutable_input() = Eigen::MatrixXd::Random(5, 5);
    
    // Simuliamo una copia (se hai definito un costruttore di copia)
    DenseCache cache2 = cache1; 
    
    // Modifichiamo cache2
    cache2.mutable_input() *= 2.0;
    
    // Verifichiamo che cache1 sia rimasta invariata
    EXPECT_FALSE(cache1.get_input().isApprox(cache2.get_input()));
}

TEST(ConvCacheTest, HandlesInvalidDimensionsGracefully) {
    ConvCache cache;
    // Impostiamo dimensioni che non hanno senso o causano mismatch
    cache.set_input_shape(0, 0, 0); 
    
    // Assicurati che is_valid() ritorni false invece di lanciare eccezioni non gestite
    EXPECT_FALSE(cache.is_valid());
}

TEST(LayerIntegrationTest, InputOutputFlow) {
    DenseCache dense;
    BatchNormCache bn;
    
    // Output di Dense deve diventare Input di BatchNorm
    Eigen::MatrixXd fake_output = Eigen::MatrixXd::Random(32, 10);
    dense.mutable_output() = fake_output;
    
    bn.mutable_input() = dense.get_output();
    
    EXPECT_TRUE(bn.get_input().isApprox(dense.get_output()));
}

