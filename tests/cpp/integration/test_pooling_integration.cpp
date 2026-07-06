#include <gtest/gtest.h>
#include "components/layers/pooling_layer.h"
#include <Eigen/Dense>
#include <memory>

class PoolingIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ipotizziamo: 1 immagine, 2 canali, 4x4 pixel ciascuno
        // Dimensione totale input = 1 * 2 * 4 * 4 = 32 elementi
        batch_size = 1;
        channels = 2;
        input_height = 4;
        input_width = 4;
        input_size = channels * input_height * input_width;

        // Configurazione del pooling: pool=2, stride=2 -> l'output sarà 2x2 per canale
        // Dimensione totale output = 1 * 2 * 2 * 2 = 8 elementi
        pool_size = 2;
        stride = 2;
        
        output_height = (input_height - pool_size) / stride + 1;
        output_width = (input_width - pool_size) / stride + 1;
        output_size = channels * output_height * output_width;
    }

    int batch_size;
    int channels;
    int input_height;
    int input_width;
    int input_size;
    int pool_size;
    int stride;
    int output_height;
    int output_width;
    int output_size;
};

// ============================================================================
// TEST 1: INTEGRITÀ MAX POOLING (FORWARD + BACKWARD O(N) VERIFY)
// ============================================================================
TEST_F(PoolingIntegrationTest, MaxPoolingForwardBackwardPipeline) {
    auto pool_layer = std::make_unique<layers::PoolingLayer>(
        pool_size, stride, layers::PoolingLayer::MAX, channels
    );
    pool_layer->set_input_shape(input_size);

    // Creiamo un input controllato: un canale con valori crescenti, uno con picco isolato
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(batch_size, input_size);
    
    // Canale 0: riempito da 1 a 16
    for (int i = 0; i < 16; ++i) input(0, i) = static_cast<double>(i + 1);
    // Canale 1: tutto a zero tranne una posizione nota (es. indice spaziale 5, valore 99.0)
    input(0, 16 + 5) = 99.0;

    // 1. Eseguiamo il forward in modalità training
    Eigen::MatrixXd output = pool_layer->forward(input, true);

    // Verifica dimensioni output
    ASSERT_EQ(output.rows(), batch_size);
    ASSERT_EQ(output.cols(), output_size);

    // Verifica valori attesi nel Max Pooling
    // Quadrante in alto a sinistra del canale 0 (elementi 1,2,5,6) -> max è 6
    EXPECT_NEAR(output(0, 0), 6.0, 1e-5);
    // Canale 1 deve aver catturato il picco a 99.0 nel rispettivo quadrante
    EXPECT_NEAR(output(0, 4), 99.0, 1e-5);

    // 2. Creiamo un finto gradiente in ingresso dall'alto (tutti 1.0)
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Ones(batch_size, output_size);

    // Eseguiamo il backward
    Eigen::MatrixXd bottom_gradient = pool_layer->backward(top_gradient);

    // Verifica dimensioni gradiente propagato all'indietro
    ASSERT_EQ(bottom_gradient.rows(), batch_size);
    ASSERT_EQ(bottom_gradient.cols(), input_size);

    // Verifica la selettività del gradiente del Max Pooling: 
    // Il gradiente deve andare SOLO dove c'era il valore massimo!
    // Canale 0, cella dell'elemento '6' (riga 1, colonna 1 del canale 0 -> indice flat 5)
    EXPECT_NEAR(bottom_gradient(0, 5), 1.0, 1e-5);
    // Canale 0, cella dell'elemento '1' (indice flat 0) non era il massimo -> deve essere 0.0
    EXPECT_NEAR(bottom_gradient(0, 0), 0.0, 1e-5);

    // Canale 1, la cella col picco 99.0 (indice flat 16 + 5 = 21) deve ricevere il gradiente
    EXPECT_NEAR(bottom_gradient(0, 21), 1.0, 1e-5);
}

// ============================================================================
// TEST 2: INTEGRITÀ AVERAGE POOLING (DISTRIBUZIONE UNIFORME)
// ============================================================================
TEST_F(PoolingIntegrationTest, AveragePoolingGradDistribution) {
    auto pool_layer = std::make_unique<layers::PoolingLayer>(
        pool_size, stride, layers::PoolingLayer::AVG, channels
    );
    pool_layer->set_input_shape(input_size);

    Eigen::MatrixXd input = Eigen::MatrixXd::Constant(batch_size, input_size, 4.0);
    
    // Forward
    Eigen::MatrixXd output = pool_layer->forward(input, true);
    
    // Essendo la matrice costante a 4.0, la media di ogni patch deve essere 4.0
    for (int i = 0; i < output_size; ++i) {
        EXPECT_NEAR(output(0, i), 4.0, 1e-5);
    }

    // Finto gradiente a 4.0
    Eigen::MatrixXd top_gradient = Eigen::MatrixXd::Constant(batch_size, output_size, 4.0);
    Eigen::MatrixXd bottom_gradient = pool_layer->backward(top_gradient);

    // Nel pooling medio con patch 2x2 (4 elementi), il gradiente si distribuisce come: grad_val / count
    // Quindi 4.0 / 4 = 1.0 per ogni cella dell'input.
    for (int i = 0; i < input_size; ++i) {
        EXPECT_NEAR(bottom_gradient(0, i), 1.0, 1e-5);
    }
}