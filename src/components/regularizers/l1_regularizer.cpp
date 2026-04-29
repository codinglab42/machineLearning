#include <memory>
#include <Eigen/Dense>
#include "components/regularizers/l1_regularizer.h"
#include <cmath>

namespace models {

    L1Regularizer::L1Regularizer(double strength)
        : Regularizer(strength) {}

    double L1Regularizer::compute_loss(const Eigen::MatrixXd& weights) const {
        // L1 loss = lambda * sum(|w|)
        return strength_ * weights.array().abs().sum();
    }

    double L1Regularizer::compute_loss(const Eigen::VectorXd& bias) const {
        return strength_ * bias.array().abs().sum();
    }

    Eigen::MatrixXd L1Regularizer::compute_gradient(const Eigen::MatrixXd& weights) const {
        // Gradiente L1 = lambda * sign(w)
        Eigen::MatrixXd grad(weights.rows(), weights.cols());
        for (int i = 0; i < weights.size(); ++i) {
            grad(i) = (weights(i) > 0) ? strength_ : 
                      (weights(i) < 0) ? -strength_ : 0.0;
        }
        return grad;
    }

    Eigen::VectorXd L1Regularizer::compute_gradient(const Eigen::VectorXd& bias) const {
        Eigen::VectorXd grad(bias.size());
        for (int i = 0; i < bias.size(); ++i) {
            grad(i) = (bias(i) > 0) ? strength_ : 
                      (bias(i) < 0) ? -strength_ : 0.0;
        }
        return grad;
    }

    std::unique_ptr<Regularizer> L1Regularizer::clone() const {
        return std::make_unique<L1Regularizer>(strength_);
    }

} // namespace models

