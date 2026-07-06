#include <memory>
#include <Eigen/Dense>
#include "components/activation/activation.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace Eigen;
using namespace activation;

// Factory per creare funzioni di attivazione
std::unique_ptr<Activation> activation::create_activation(const std::string& type) {
    if (type == "relu") {
        return std::make_unique<ReLU>();
    } else if (type == "sigmoid") {
        return std::make_unique<Sigmoid>();
    } else if (type == "tanh") {
        return std::make_unique<Tanh>();
    } else if (type == "softmax") {
        return std::make_unique<Softmax>();
    } else if (type == "leaky_relu") {
        return std::make_unique<LeakyReLU>();
    } else if (type == "linear" || type == "identity") {
        return std::make_unique<Linear>();
    }
    return nullptr;
}

// ========================================================================
// ReLU foeward/Backward
// ========================================================================

MatrixXd ReLU::forward(const MatrixXd& z) {
    return z.unaryExpr([](double v) { return std::max(0.0, v); });
}

MatrixXd ReLU::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd dZ = (z.array() > 0.0).cast<double>().matrix();
    return dA.array() * dZ.array();
}

// ========================================================================
// Sigmoid Forward/backward
// ========================================================================
MatrixXd Sigmoid::forward(const MatrixXd& z) {
    return z.unaryExpr([](double v) {
        // Manteniamo la tua ottima implementazione stabile a due rami,
        // aggiungendo un leggero clip per evitare overflow in casi limite
        double v_clipped = std::max(-50.0, std::min(50.0, v));
        if (v_clipped >= 0) {
            return 1.0 / (1.0 + std::exp(-v_clipped));
        } else {
            double exp_v = std::exp(v_clipped);
            return exp_v / (1.0 + exp_v);
        }
    });
}

MatrixXd Sigmoid::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd s = forward(z);
    MatrixXd dZ = s.array() * (1.0 - s.array());
    return dA.array() * dZ.array();
}

// Tanh
MatrixXd Tanh::forward(const MatrixXd& z) {
    return z.unaryExpr([](double v) { return std::tanh(v); });
}

MatrixXd Tanh::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd tanh_z = forward(z);
    MatrixXd dZ = 1.0 - tanh_z.array().square();
    return dA.array() * dZ.array();
}

// ========================================================================
// Softmax Forward/backward
// ========================================================================
MatrixXd Softmax::forward(const MatrixXd& z) {
    // Sottrai il massimo coefficiente di ogni riga per stabilità numerica
    Eigen::VectorXd row_max = z.rowwise().maxCoeff();
    MatrixXd exp_z = (z.colwise() - row_max).array().exp().matrix();
    
    // Calcola la somma delle righe e aggiungi un epsilon per evitare divisioni per zero
    Eigen::VectorXd sum_exp = exp_z.rowwise().sum().array() + 1e-15;
    
    // Divide ogni colonna di exp_z per il vettore colonna delle somme (Broadcasting)
    return (exp_z.array().colwise() / sum_exp.array()).matrix();
}

MatrixXd Softmax::backward(const MatrixXd& dA, const MatrixXd& z) {
    // Per softmax combinato con cross-entropy, il gradiente è semplice
    // dZ = y_pred - y_true (se usato con cross-entropy)
    // Qui restituiamo dA come placeholder, la logica completa è nella loss
    return dA;
}

// Leaky ReLU
MatrixXd LeakyReLU::forward(const MatrixXd& z) {
    return z.unaryExpr([this](double v) { 
        return (v > 0) ? v : alpha_ * v; 
    });
}

MatrixXd LeakyReLU::backward(const MatrixXd& dA, const MatrixXd& z) {
    MatrixXd dZ = z.unaryExpr([this](double v) { 
        return (v > 0) ? 1.0 : alpha_; 
    });
    return dA.array() * dZ.array();
}