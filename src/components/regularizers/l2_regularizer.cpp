#include <memory>
#include <Eigen/Dense>
#include "components/regularizers/l2_regularizer.h"

namespace models {

    L2Regularizer::L2Regularizer(double strength)
        : Regularizer(strength) {}

    double L2Regularizer::compute_loss(const Eigen::MatrixXd& weights) const {
        // L2 loss = (lambda/2) * sum(w^2)
        return 0.5 * strength_ * weights.array().square().sum();
    }

    double L2Regularizer::compute_loss(const Eigen::VectorXd& bias) const {
        return 0.5 * strength_ * bias.array().square().sum();
    }

    Eigen::MatrixXd L2Regularizer::compute_gradient(const Eigen::MatrixXd& weights) const {
        // Gradiente L2 = lambda * w
        return strength_ * weights;
    }

    Eigen::VectorXd L2Regularizer::compute_gradient(const Eigen::VectorXd& bias) const {
        return strength_ * bias;
    }

    std::unique_ptr<Regularizer> L2Regularizer::clone() const {
        return std::make_unique<L2Regularizer>(strength_);
    }

} // namespace models

