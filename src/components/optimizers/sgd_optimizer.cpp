#include <memory>
#include <Eigen/Dense>
#include "components/optimizers/sgd_optimizer.h"
#include "utils/serializable.h"

namespace models {

    SGDOptimizer::SGDOptimizer(double learning_rate, double decay)
        : Optimizer(learning_rate, decay) {}

    void SGDOptimizer::update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) {
        double lr = get_current_learning_rate();
        iterations_++;
        
        // SGD semplice: w = w - lr * gradient
        weights -= lr * gradient;
    }

    void SGDOptimizer::update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) {
        double lr = get_current_learning_rate();
        
        bias -= lr * gradient;
    }

    void SGDOptimizer::reset() {
        iterations_ = 0;
        // Nessuno stato da resettare per SGD
    }

    std::unique_ptr<Optimizer> SGDOptimizer::clone() const {
        auto clone = std::make_unique<SGDOptimizer>(learning_rate_, decay_);
        clone->iterations_ = iterations_;
        return clone;
    }

} // namespace models