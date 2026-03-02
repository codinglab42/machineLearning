#include "components/optimizers/sgd_optimizer.h"

namespace models {

    SGDOptimizer::SGDOptimizer(double learning_rate, double decay)
        : Optimizer(learning_rate, decay) {}

    void SGDOptimizer::update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) {
        double lr = get_current_learning_rate();
        weights -= lr * gradient;
        iterations_++;
    }

    void SGDOptimizer::update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) {
        double lr = get_current_learning_rate();
        bias -= lr * gradient;
    }

    void SGDOptimizer::reset() {
        iterations_ = 0;
    }

    std::unique_ptr<Optimizer> SGDOptimizer::clone() const {
        return std::make_unique<SGDOptimizer>(learning_rate_, decay_);
    }

} // namespace models
