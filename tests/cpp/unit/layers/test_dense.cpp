#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <memory>
#include "components/layers/dense_layer.h"

using namespace Eigen;
using namespace std;

class DenseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Inizializzazioni comuni
    }
    
    // Funzione di loss: Mean Squared Error
    double mse_loss(const MatrixXd& pred, const MatrixXd& target) {
        return (pred - target).array().square().mean();
    }
    
    // Derivata della MSE loss rispetto a pred
    MatrixXd mse_gradient(const MatrixXd& pred, const MatrixXd& target) {
        return 2.0 * (pred - target) / pred.rows();
    }
    
    // Calcolo gradiente numerico con finite differences
    double numerical_gradient(layers::DenseLayer& layer, 
                              const MatrixXd& X, 
                              const MatrixXd& target,
                              int weight_row, int weight_col,
                              double epsilon = 1e-5) {
        
        MatrixXd weights = layer.get_weights();
        double original = weights(weight_row, weight_col);
        
        // +epsilon
        weights(weight_row, weight_col) = original + epsilon;
        layer.set_weights(weights);
        double loss_plus = mse_loss(layer.forward(X), target);
        
        // -epsilon
        weights(weight_row, weight_col) = original - epsilon;
        layer.set_weights(weights);
        double loss_minus = mse_loss(layer.forward(X), target);
        
        // Ripristina
        weights(weight_row, weight_col) = original;
        layer.set_weights(weights);
        
        return (loss_plus - loss_minus) / (2.0 * epsilon);
    }
};

// TEST 1: Verifica costruttore e dimensioni
TEST_F(DenseLayerTest, ConstructorAndDimensions) {
    int in_size = 5;
    int out_size = 3;
    
    layers::DenseLayer layer(in_size, out_size, "relu", 0.01, 0.001);
    
    EXPECT_EQ(layer.get_input_size(), in_size);
    EXPECT_EQ(layer.get_output_size(), out_size);
    EXPECT_EQ(layer.get_weights().rows(), out_size);
    EXPECT_EQ(layer.get_weights().cols(), in_size);
    EXPECT_EQ(layer.get_biases().size(), out_size);
    EXPECT_EQ(layer.get_type(), "Dense");
    EXPECT_TRUE(layer.has_weights());
    
    cout << "✓ Constructor and dimensions test passed" << endl;
}

// TEST 2: Forward pass verifica dimensioni output
TEST_F(DenseLayerTest, ForwardDimensions) {
    int in_size = 3;
    int out_size = 2;
    int batch_size = 4;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    MatrixXd X = MatrixXd::Random(batch_size, in_size);
    MatrixXd output = layer.forward(X);
    
    EXPECT_EQ(output.rows(), batch_size);
    EXPECT_EQ(output.cols(), out_size);
    
    cout << "✓ Forward dimensions test passed" << endl;
}

// TEST 3: Forward pass con pesi noti (verifica calcolo)
TEST_F(DenseLayerTest, ForwardComputation) {
    int in_size = 2;
    int out_size = 2;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    // Set weights noti
    MatrixXd test_weights(out_size, in_size);
    test_weights << 1, 2,
                    3, 4;
    layer.set_weights(test_weights);
    
    VectorXd test_biases(out_size);
    test_biases << 0.1, 0.2;
    layer.set_biases(test_biases);
    
    MatrixXd X(1, in_size);
    X << 5, 6;
    
    MatrixXd output = layer.forward(X);
    
    // Calcolo atteso: X * W^T + b
    MatrixXd expected = X * test_weights.transpose();
    expected.rowwise() += test_biases.transpose();
    
    EXPECT_TRUE(output.isApprox(expected, 1e-10));
    cout << "✓ Forward computation test passed" << endl;
}

// TEST 4: Forward con ReLU (verifica non negatività)
TEST_F(DenseLayerTest, ForwardReLU) {
    int in_size = 2;
    int out_size = 2;
    
    layers::DenseLayer layer(in_size, out_size, "relu");
    
    // Pesci che possono dare output negativi
    MatrixXd test_weights(out_size, in_size);
    test_weights << -1, -2,
                    -3, -4;
    layer.set_weights(test_weights);
    
    MatrixXd X(1, in_size);
    X << 1, 1;
    
    MatrixXd output = layer.forward(X);
    
    // Con ReLU, tutti gli output devono essere >= 0
    EXPECT_GE(output.minCoeff(), 0);
    cout << "✓ Forward ReLU test passed" << endl;
}

// TEST 5: Forward con Sigmoid (verifica range (0,1))
TEST_F(DenseLayerTest, ForwardSigmoid) {
    int in_size = 2;
    int out_size = 2;
    
    layers::DenseLayer layer(in_size, out_size, "sigmoid");
    
    MatrixXd X = MatrixXd::Random(3, in_size);
    MatrixXd output = layer.forward(X);
    
    EXPECT_GE(output.minCoeff(), 0);
    EXPECT_LE(output.maxCoeff(), 1);
    cout << "✓ Forward Sigmoid test passed" << endl;
}

// TEST 6: Gradient storage - verifica che i gradienti vengano calcolati
TEST_F(DenseLayerTest, GradientStorage) {
    layers::DenseLayer layer(3, 2, "identity");
    
    MatrixXd X = MatrixXd::Random(2, 3);
    MatrixXd target = MatrixXd::Random(2, 2);
    
    // Forward
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    // Backward
    layer.backward(dL_dout, 0.01);
    
    // Verifica che i gradienti siano stati calcolati (non zero)
    MatrixXd grad_w = layer.get_weight_gradients();
    VectorXd grad_b = layer.get_bias_gradients();
    
    EXPECT_GT(grad_w.array().abs().sum(), 0);
    EXPECT_GT(grad_b.array().abs().sum(), 0);
    
    cout << "✓ Gradient storage test passed" << endl;
}

// TEST 7: Gradient check con identity activation
TEST_F(DenseLayerTest, GradientCheckIdentity) {
    int in_size = 3;
    int out_size = 2;
    int batch_size = 2;
    double epsilon = 1e-5;
    double tolerance = 1e-4;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    MatrixXd X = MatrixXd::Random(batch_size, in_size);
    MatrixXd target = MatrixXd::Random(batch_size, out_size);
    
    // Forward
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    // Salva i pesi originali PRIMA del backward
    MatrixXd original_weights = layer.get_weights();
    
    // Backward (lr=0.0 per NON modificare i pesi!)
    layer.backward(dL_dout, 0.0);
    
    // Ottieni gradienti analitici
    MatrixXd analytical_grads = layer.get_weight_gradients();
    
    // Ripristina i pesi originali (anche se con lr=0.0 non dovrebbero essere cambiati)
    layer.set_weights(original_weights);
    
    int num_failures = 0;
    double max_diff = 0.0;
    
    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < in_size; ++j) {
            double num_grad = numerical_gradient(layer, X, target, i, j, epsilon);
            double ana_grad = analytical_grads(i, j);
            
            double diff = std::abs(ana_grad - num_grad);
            max_diff = std::max(max_diff, diff);
            
            if (diff > tolerance) {
                num_failures++;
                cout << "  Peso (" << i << "," << j << "): "
                     << "ana=" << ana_grad << ", num=" << num_grad 
                     << ", diff=" << diff << endl;
            }
            
            EXPECT_NEAR(ana_grad, num_grad, tolerance);
        }
    }
    
    if (num_failures == 0) {
        cout << "✓ Gradient check (identity) passed (max diff=" << max_diff << ")" << endl;
    }
}

// TEST 8: Gradient check con ReLU activation
TEST_F(DenseLayerTest, GradientCheckReLU) {
    int in_size = 3;
    int out_size = 2;
    int batch_size = 2;
    double epsilon = 1e-5;
    double tolerance = 1e-3;
    
    layers::DenseLayer layer(in_size, out_size, "relu");
    
    MatrixXd X = MatrixXd::Random(batch_size, in_size);
    MatrixXd target = MatrixXd::Random(batch_size, out_size);
    
    // Salva i pesi originali PRIMA di qualsiasi operazione
    MatrixXd original_weights = layer.get_weights();
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    // Backward con learning rate 0 per NON modificare i pesi!
    layer.backward(dL_dout, 0.0);
    MatrixXd analytical_grads = layer.get_weight_gradients();
    
    // Ripristina i pesi originali (anche se con lr=0.0 non dovrebbero essere cambiati)
    layer.set_weights(original_weights);
    
    int num_failures = 0;
    double max_diff = 0.0;
    
    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < in_size; ++j) {
            double num_grad = numerical_gradient(layer, X, target, i, j, epsilon);
            double ana_grad = analytical_grads(i, j);
            
            double diff = std::abs(ana_grad - num_grad);
            max_diff = std::max(max_diff, diff);
            
            if (diff > tolerance) {
                num_failures++;
                cout << "  Peso (" << i << "," << j << "): "
                     << "ana=" << ana_grad << ", num=" << num_grad 
                     << ", diff=" << diff << endl;
            }
            
            EXPECT_NEAR(ana_grad, num_grad, tolerance);
        }
    }
    
    if (num_failures == 0) {
        cout << "✓ Gradient check (ReLU) passed (max diff=" << max_diff << ")" << endl;
    }
}

// TEST 9: Batch gradient - verifica normalizzazione
TEST_F(DenseLayerTest, BatchGradient) {
    int in_size = 2;
    int out_size = 2;
    int batch_size = 3;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    // Pesi fissi per test
    MatrixXd test_weights(out_size, in_size);
    test_weights << 1, 2,
                    3, 4;
    layer.set_weights(test_weights);
    
    MatrixXd X(batch_size, in_size);
    X << 1, 1,
         2, 2,
         3, 3;
    
    MatrixXd target = MatrixXd::Zero(batch_size, out_size);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    layer.backward(dL_dout, 1.0);
    MatrixXd grads = layer.get_weight_gradients();
    
    // Calcola gradiente atteso manualmente
    MatrixXd expected_grads = MatrixXd::Zero(out_size, in_size);
    for (int b = 0; b < batch_size; ++b) {
        expected_grads += dL_dout.row(b).transpose() * X.row(b);
    }
    expected_grads /= batch_size;
    
    EXPECT_TRUE(grads.isApprox(expected_grads, 1e-10));
    cout << "✓ Batch gradient test passed" << endl;
}

// TEST 10: Regolarizzazione L2
TEST_F(DenseLayerTest, L2Regularization) {
    int in_size = 2;
    int out_size = 2;
    double lambda = 0.1;
    
    // Layer con regolarizzazione L2
    layers::DenseLayer layer(in_size, out_size, "identity", 0.0, lambda);
    
    MatrixXd test_weights(out_size, in_size);
    test_weights << 1, 2,
                    3, 4;
    layer.set_weights(test_weights);
    
    MatrixXd X = MatrixXd::Ones(1, in_size);
    MatrixXd target = MatrixXd::Zero(1, out_size);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    layer.backward(dL_dout, 1.0);
    MatrixXd grads = layer.get_weight_gradients();
    
    // Gradiente atteso: base + regolarizzazione
    MatrixXd base_grad = dL_dout.transpose() * X;
    MatrixXd expected_grads = base_grad + lambda * test_weights;
    
    EXPECT_TRUE(grads.isApprox(expected_grads, 1e-10));
    cout << "✓ L2 regularization test passed" << endl;
}

// TEST 11: Regolarizzazione L1
TEST_F(DenseLayerTest, L1Regularization) {
    int in_size = 2;
    int out_size = 2;
    double lambda = 0.1;
    
    // Layer con regolarizzazione L1
    layers::DenseLayer layer(in_size, out_size, "identity", lambda, 0.0);
    
    MatrixXd test_weights(out_size, in_size);
    test_weights << 1, -2,
                    3, -4;
    layer.set_weights(test_weights);
    
    MatrixXd X = MatrixXd::Ones(1, in_size);
    MatrixXd target = MatrixXd::Zero(1, out_size);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    layer.backward(dL_dout, 1.0);
    MatrixXd grads = layer.get_weight_gradients();
    
    // Gradiente L1 atteso: lambda * sign(w)
    MatrixXd l1_grad = lambda * test_weights.unaryExpr([](double w) {
        return w > 0 ? 1.0 : (w < 0 ? -1.0 : 0.0);
    });
    
    // Gradiente atteso: base + regolarizzazione
    MatrixXd base_grad = dL_dout.transpose() * X;
    MatrixXd expected_grads = base_grad + l1_grad;
    
    EXPECT_TRUE(grads.isApprox(expected_grads, 1e-10));
    cout << "✓ L1 regularization test passed" << endl;
}

// TEST 12: Clear gradients
TEST_F(DenseLayerTest, ClearGradients) {
    layers::DenseLayer layer(3, 2, "identity");
    
    MatrixXd X = MatrixXd::Random(2, 3);
    MatrixXd target = MatrixXd::Random(2, 2);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    layer.backward(dL_dout, 0.01);
    
    // Verifica che i gradienti non siano zero
    EXPECT_GT(layer.get_weight_gradients().array().abs().sum(), 0);
    
    // Pulisci
    layer.clear_gradients();
    
    // Verifica che siano zero
    EXPECT_EQ(layer.get_weight_gradients().array().abs().sum(), 0);
    EXPECT_EQ(layer.get_bias_gradients().array().abs().sum(), 0);
    
    cout << "✓ Clear gradients test passed" << endl;
}

// TEST 13: Set weights resetta gradienti
TEST_F(DenseLayerTest, SetWeightsResetsGradients) {
    layers::DenseLayer layer(3, 2, "identity");
    
    // Calcola gradienti
    MatrixXd X = MatrixXd::Random(2, 3);
    MatrixXd target = MatrixXd::Random(2, 2);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    layer.backward(dL_dout, 0.01);
    
    // Verifica gradienti non zero
    EXPECT_GT(layer.get_weight_gradients().array().abs().sum(), 0);
    
    // Cambia pesi
    MatrixXd new_weights = MatrixXd::Random(2, 3);
    layer.set_weights(new_weights);
    
    // Verifica gradienti resettati
    EXPECT_EQ(layer.get_weight_gradients().array().abs().sum(), 0);
    
    cout << "✓ Set weights resets gradients test passed" << endl;
}

// TEST 14: Propagazione del gradiente al layer precedente
TEST_F(DenseLayerTest, GradientPropagation) {
    int in_size = 3;
    int out_size = 2;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    MatrixXd X = MatrixXd::Random(1, in_size);
    MatrixXd target = MatrixXd::Random(1, out_size);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    MatrixXd dA_prev = layer.backward(dL_dout, 0.1);
    
    EXPECT_EQ(dA_prev.rows(), X.rows());
    EXPECT_EQ(dA_prev.cols(), X.cols());
    
    cout << "✓ Gradient propagation test passed" << endl;
}

// TEST 15: Weight update con learning rate
TEST_F(DenseLayerTest, WeightUpdate) {
    int in_size = 2;
    int out_size = 2;
    double lr = 0.01;
    
    layers::DenseLayer layer(in_size, out_size, "identity");
    
    MatrixXd weights_before = layer.get_weights();
    VectorXd biases_before = layer.get_biases();
    
    MatrixXd X = MatrixXd::Random(1, in_size);
    MatrixXd target = MatrixXd::Random(1, out_size);
    
    MatrixXd output = layer.forward(X);
    MatrixXd dL_dout = mse_gradient(output, target);
    
    layer.backward(dL_dout, lr);
    
    MatrixXd grads = layer.get_weight_gradients();
    MatrixXd weights_after = layer.get_weights();
    
    MatrixXd expected_weights = weights_before - lr * grads;
    
    EXPECT_TRUE(weights_after.isApprox(expected_weights, 1e-10));
    cout << "✓ Weight update test passed" << endl;
}

// TEST 16: Serializzazione (test base)
TEST_F(DenseLayerTest, Serialization) {
    layers::DenseLayer layer1(3, 2, "relu", 0.01, 0.001);
    
    // Imposta pesi noti per test
    MatrixXd test_weights(2, 3);
    test_weights << 1, 2, 3,
                    4, 5, 6;
    VectorXd test_biases(2);
    test_biases << 0.1, 0.2;
    
    layer1.set_weights(test_weights);
    layer1.set_biases(test_biases);
    
    // Serializza su string stream
    stringstream ss;
    layer1.serialize(ss);
    
    // Deserializza in nuovo layer
    layers::DenseLayer layer2(1, 1, "identity");  // costruttore dummy
    layer2.deserialize(ss);
    
    // Verifica che i pesi siano uguali
    EXPECT_TRUE(layer2.get_weights().isApprox(layer1.get_weights()));
    EXPECT_TRUE(layer2.get_biases().isApprox(layer1.get_biases()));
    EXPECT_EQ(layer2.get_type(), layer1.get_type());
    
    cout << "✓ Serialization test passed" << endl;
}

// TEST 17: Eccezioni - input dimensioni errate
TEST_F(DenseLayerTest, InvalidInputDimensions) {
    layers::DenseLayer layer(3, 2, "identity");
    
    MatrixXd X_wrong = MatrixXd::Random(1, 4);  // 4 features invece di 3
    
    EXPECT_THROW(layer.forward(X_wrong), ml_exception::DimensionMismatchException);
    cout << "✓ Invalid input dimensions test passed" << endl;
}

// TEST 18: Eccezioni - cache non inizializzata
TEST_F(DenseLayerTest, UninitializedCache) {
    layers::DenseLayer layer(3, 2, "identity");
    
    MatrixXd dummy_grad = MatrixXd::Random(1, 2);
    
    EXPECT_THROW(layer.backward(dummy_grad, 0.1), ml_exception::InvalidConfigurationException);
    cout << "✓ Uninitialized cache test passed" << endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}