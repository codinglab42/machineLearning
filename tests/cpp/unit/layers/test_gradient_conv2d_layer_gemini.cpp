#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>
#include "components/layers/conv2d_layer.h"

using namespace layers;
using namespace Eigen;

class Conv2DGradientTest : public ::testing::Test {
protected:
    double epsilon = 1e-6;
    double tolerance = 1e-5;
};

TEST_F(Conv2DGradientTest, CheckWeightAndBiasGradients) {
    // Configurazione: 2 filtri, kernel 3x3, input 5x5 (25 pixel)
    int filters = 2;
    int kernel_size = 3;
    auto layer = std::make_unique<Conv2DLayer>(filters, kernel_size, 1, "valid", "linear");
    
    // Inizializza shape (H=5, W=5)
    layer->set_input_shape(25);

    // 1. INPUT: Deve essere [batch_size, input_size] -> [1, 25]
    MatrixXd input = MatrixXd::Random(1, 25);
    
    // 2. FORWARD ANALITICO
    MatrixXd output = layer->forward(input);
    
    // 3. BACKWARD ANALITICO
    // dL/dOut deve essere [batch_size, output_size] -> [1, 18] 
    // (perché 3x3 spaziale * 2 filtri = 18)
    int out_size = layer->get_output_size();
    MatrixXd dL_dOut = MatrixXd::Ones(1, out_size); 
    layer->backward(dL_dOut);
    
    // La tua classe unifica dW e db in un'unica matrice se use_bias è true
    MatrixXd dW_analytical = layer->get_weights_gradient();

    // 4. GRADIENTE NUMERICO
    MatrixXd weights = layer->get_weights(); // [filters, kernel_elements + 1]
    MatrixXd dW_numerical = MatrixXd::Zero(weights.rows(), weights.cols());

    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            double original_val = weights(i, j);

            // W + eps
            weights(i, j) = original_val + epsilon;
            layer->set_weights(weights);
            double loss_plus = layer->forward(input).sum();

            // W - eps
            weights(i, j) = original_val - epsilon;
            layer->set_weights(weights);
            double loss_minus = layer->forward(input).sum();

            dW_numerical(i, j) = (loss_plus - loss_minus) / (2 * epsilon);
            
            // Reset peso
            weights(i, j) = original_val;
            layer->set_weights(weights);
        }
    }

    // 5. CONFRONTO
    // Calcoliamo l'errore relativo globale (include pesi e bias)
    double error = (dW_analytical - dW_numerical).norm() / 
                   (std::max(1e-10, dW_analytical.norm() + dW_numerical.norm()));
    
    std::cout << "Conv2D Total Gradient Error (Weights + Bias): " << error << std::endl;
    
    // Se la matematica di im2col/col2im è corretta, l'errore sarà < 1e-7
    EXPECT_LT(error, tolerance);
}