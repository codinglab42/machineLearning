#include <gtest/gtest.h>
#include "components/layers/conv2d_layer.h"
#include "components/layers/pooling_layer.h"
#include "components/layers/flatten_layer.h"
#include "components/layers/dense_layer.h"
#include "components/layers/batch_norm_layer.h"
#include "components/layers/dropout_layer.h"
#include "components/optimizers/sgd_optimizer.h" // Adatta il percorso se necessario
#include "components/loss/mean_squared_error_loss.h" // Adatta il percorso se necessario
#include "models/neural_network.h" // Adatta se hai una classe container o usa un ciclo manuale

#include <Eigen/Dense>
#include <memory>

TEST(HybridPipelineIntegrationTest, CNNToDenseForwardBackwardPipeline) {
    // Configurazione geometrica dell'input: 1 immagine, 1 canale, 6x6 pixel
    int batch_size = 1;
    int in_channels = 1;
    int in_height = 6;
    int in_width = 6;
    int input_size = in_channels * in_height * in_width; // 36

    // 1. Inizializzazione dei Layer della Catena
    // Conv2D: 2 filtri, kernel 3x3, stride 1, padding valid -> output: 2 canali, 4x4 pixel
    auto conv = std::make_shared<layers::Conv2DLayer>(2, 3, 1, "valid", "linear");
    conv->set_input_shape(input_size);
    conv->initialize_weights(); // <--- SVEGLIA I PESI DELLA CONVOLUZIONE

    // MaxPooling: pool 2x2, stride 2 -> output: 2 canali, 2x2 pixel
    auto pool = std::make_shared<layers::PoolingLayer>(2, 2, layers::PoolingLayer::MAX, 2);
    pool->set_input_shape(conv->get_output_size());

    // Flatten: trasforma i 2 canali 2x2 in un vettore lineare da 2 * 2 * 2 = 8 elementi
    auto flatten = std::make_shared<layers::FlattenLayer>();
    flatten->set_input_shape(pool->get_output_size());

    // Dense: prende gli 8 elementi in ingresso e mappa su 2 classi di output
    auto dense = std::make_shared<layers::DenseLayer>(2, "linear");
    dense->set_input_shape(flatten->get_output_size());
    dense->initialize_weights(); // <--- SVEGLIA I PESI DEL DENSE LAYER

    // 2. Generazione Input Controllato (Usa valori asimmetrici stabili)
    Eigen::MatrixXd input_data = Eigen::MatrixXd::Zero(batch_size, input_size);
    for (int i = 0; i < input_size; ++i) {
        input_data(0, i) = static_cast<double>(i + 1) * 0.1; 
    }

    // ==========================================
    // PIPELINE FORWARD
    // ==========================================
    Eigen::MatrixXd x1 = conv->forward(input_data, true);
    ASSERT_EQ(x1.cols(), conv->get_output_size());

    Eigen::MatrixXd x2 = pool->forward(x1, true);
    ASSERT_EQ(x2.cols(), pool->get_output_size());

    Eigen::MatrixXd x3 = flatten->forward(x2, true);
    ASSERT_EQ(x3.cols(), flatten->get_output_size());
    ASSERT_EQ(x3.cols(), 8); // Verifica geometrica di controllo flussi spaziali

    Eigen::MatrixXd final_output = dense->forward(x3, true);
    ASSERT_EQ(final_output.rows(), batch_size);
    ASSERT_EQ(final_output.cols(), 2);

    // ==========================================
    // PIPELINE BACKWARD (Propagazione del gradiente a ritroso)
    // ==========================================
    // Gradiente asimmetrico in arrivo dalla loss function
    Eigen::MatrixXd loss_gradient(batch_size, 2);
    loss_gradient(0, 0) = 0.25;
    loss_gradient(0, 1) = 0.75;

    Eigen::MatrixXd g3 = dense->backward(loss_gradient);
    ASSERT_EQ(g3.cols(), flatten->get_output_size());

    Eigen::MatrixXd g2 = flatten->backward(g3);
    ASSERT_EQ(g2.cols(), pool->get_output_size());

    Eigen::MatrixXd g1 = pool->backward(g2);
    ASSERT_EQ(g1.cols(), conv->get_output_size());

    Eigen::MatrixXd input_gradient = conv->backward(g1);
    
    // Il gradiente finale calcolato deve combaciare millimetricamente con la dimensione iniziale dell'input
    ASSERT_EQ(input_gradient.rows(), batch_size);
    ASSERT_EQ(input_gradient.cols(), input_size);

    // Verifica di integrità matematica: i gradienti ora DEVONO essere maggiori di zero
    EXPECT_GT(input_gradient.norm(), 0.0);
    EXPECT_GT(conv->get_weights_gradient().norm(), 0.0);
    EXPECT_GT(dense->get_weights_gradient().norm(), 0.0);
}

// ============================================================================
// TEST: VARIABILITÀ DINAMICA DEL BATCH SIZE
// ============================================================================
TEST(HybridPipelineIntegrationTest, DynamicBatchSizeHandling) {
    int input_size = 4;
    int units = 2;

    auto dense = std::make_shared<layers::DenseLayer>(units, "relu");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    // Passaggio 1: Forward/Backward con Batch Size = 3 (Es. Training)
    Eigen::MatrixXd input_batch3 = Eigen::MatrixXd::Random(3, input_size);
    Eigen::MatrixXd out_batch3 = dense->forward(input_batch3, true);
    ASSERT_EQ(out_batch3.rows(), 3);

    Eigen::MatrixXd grad_batch3 = Eigen::MatrixXd::Ones(3, units);
    Eigen::MatrixXd back_batch3 = dense->backward(grad_batch3);
    ASSERT_EQ(back_batch3.rows(), 3);

    // Passaggio 2: Cambiamo istantaneamente il Batch Size = 1 (Es. Inferenza o Singolo Predict)
    Eigen::MatrixXd input_batch1 = Eigen::MatrixXd::Random(1, input_size);
    Eigen::MatrixXd out_batch1 = dense->forward(input_batch1, false); // training = false
    ASSERT_EQ(out_batch1.rows(), 1);
    
    // Se Eigen non è andato in crash qui, il layer gestisce la memoria dinamicamente!
    SUCCEED();
}

// ============================================================================
// TEST: INTEGRATION CON DROPOUT E BATCH NORMALIZATION (Corretto)
// ============================================================================
TEST(HybridPipelineIntegrationTest, BatchNormAndDropoutPipeline) {
    int batch_size = 4;  // Aumentato per stabilizzare BatchNorm
    int input_size = 4;
    int units = 3;

    auto dense = std::make_shared<layers::DenseLayer>(units, "linear");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    auto bn = std::make_shared<layers::BatchNormLayer>();
    bn->set_input_shape(units);
    bn->initialize_weights(); // Se il tuo BatchNorm ha parametri gamma/beta da inizializzare

    auto dropout = std::make_shared<layers::DropoutLayer>(0.2); // Rate leggermente più basso
    dropout->set_input_shape(units);

    // Generazione input asimmetrico e progressivo per evitare medie identiche in BatchNorm
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(batch_size, input_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_size; ++i) {
            input(b, i) = static_cast<double>((b * input_size) + i + 1) * 0.25;
        }
    }
    
    // Flusso Forward
    Eigen::MatrixXd x1 = dense->forward(input, true);
    Eigen::MatrixXd x2 = bn->forward(x1, true);
    Eigen::MatrixXd final_output = dropout->forward(x2, true);

    ASSERT_EQ(final_output.rows(), batch_size);
    ASSERT_EQ(final_output.cols(), units);

    // Flusso Backward con gradiente asimmetrico riga per riga
    Eigen::MatrixXd loss_grad = Eigen::MatrixXd::Zero(batch_size, units);
    for (int b = 0; b < batch_size; ++b) {
        for (int u = 0; u < units; ++u) {
            loss_grad(b, u) = static_cast<double>(b + u + 1) * 0.5;
        }
    }
    
    Eigen::MatrixXd g3 = dropout->backward(loss_grad);
    Eigen::MatrixXd g2 = bn->backward(g3);
    Eigen::MatrixXd final_grad = dense->backward(g2);

    // Verifiche strutturali e matematiche
    ASSERT_EQ(final_grad.rows(), batch_size);
    ASSERT_EQ(final_grad.cols(), input_size);
    
    // Ora BatchNorm ha varianza e non azzererà il gradiente all'indietro
    EXPECT_GT(dense->get_weights_gradient().norm(), 0.0);
}

// ============================================================================
// TEST: INTEGRAZIONE OTTIMIZZATORE E AGGIORNAMENTO PESI
// ============================================================================
TEST(HybridPipelineIntegrationTest, OptimizerWeightUpdateAndConvergence) {
    int batch_size = 1;
    int input_size = 3;
    int units = 2;

    auto dense = std::make_shared<layers::DenseLayer>(units, "linear");
    dense->set_input_shape(input_size);
    dense->initialize_weights();

    // Salvia mo i pesi iniziali per verificare il cambiamento dopo l'update
    Eigen::MatrixXd weights_before = dense->get_weights();

    // Configura ottimizzatore (es. SGD con learning rate 0.1) e loss
    auto optimizer = std::make_shared<models::SGDOptimizer>(0.1);
    auto loss_fn = std::make_shared<loss::MeanSquaredErrorLoss>();

    // Input fisso e target per l'addestramento
    Eigen::MatrixXd input(batch_size, input_size);
    input << 1.0, 0.5, -0.5;
    
    Eigen::MatrixXd target(batch_size, units);
    target << 0.0, 1.0;

    // 1. Forward pass
    Eigen::MatrixXd output = dense->forward(input, true);

    // 2. Calcolo della loss e del gradiente della loss
    double loss_value = loss_fn->forward(output, target);
    Eigen::MatrixXd loss_grad = loss_fn->backward(output, target);

    // 3. Backward pass sul layer
    dense->backward(loss_grad);

    // 4. Aggiornamento dei parametri tramite l'ottimizzatore
    // Se la tua architettura usa net.update() o il layer accetta direttamente l'ottimizzatore:
    // Adatta questa riga al tuo pattern (es. optimizer->update(dense); )
    dense->update(optimizer.get()); 

    Eigen::MatrixXd weights_after = dense->get_weights();

    // Verifica 1: I pesi devono essere cambiati dopo l'aggiornamento dell'ottimizzatore
    double weight_diff = (weights_after - weights_before).norm();
    EXPECT_GT(weight_diff, 0.0);

    // Verifica 2: Se rieseguiamo il forward, l'output deve essersi avvicinato al target (loss diminuita)
    Eigen::MatrixXd new_output = dense->forward(input, false);
    double new_loss_value = loss_fn->forward(new_output, target);
    
    EXPECT_LT(new_loss_value, loss_value);
}

// ============================================================================
// TEST: INTEGRITÀ GEOMETRICA E NUMERICA POST-SERIALIZZAZIONE
// ============================================================================
TEST(HybridPipelineIntegrationTest, SerializationAndNumericalConsistency) {
    int batch_size = 1;
    int input_size = 4;
    int units = 2;

    // Rete Originale
    auto dense_original = std::make_shared<layers::DenseLayer>(units, "relu");
    dense_original->set_input_shape(input_size);
    dense_original->initialize_weights();

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(batch_size, input_size);
    Eigen::MatrixXd output_original = dense_original->forward(input, false);

    // Esporta lo stato della rete (adatta al tuo metodo: es. serialize(), save(), to_json())
    // Assumiamo che restituisca una stringa o scriva su un flusso stream
    std::string serialized_state = dense_original->serialize();

    // Rete Clona (Riorganizzata dallo stato serializzato)
    auto dense_cloned = std::make_shared<layers::DenseLayer>(units, "relu");
    dense_cloned->set_input_shape(input_size);
    
    // Ripristina lo stato
    dense_cloned->deserialize(serialized_state);

    // Forward sulla rete clonata con lo stesso identico input
    Eigen::MatrixXd output_cloned = dense_cloned->forward(input, false);

    // Verifica geometrica
    ASSERT_EQ(output_cloned.rows(), output_original.rows());
    ASSERT_EQ(output_cloned.cols(), output_original.cols());

    // Verifica numerica millimetrica: l'output non deve deviare a causa del salvataggio
    for (int i = 0; i < output_original.size(); ++i) {
        EXPECT_NEAR(output_cloned(i), output_original(i), 1e-6);
    }
}