#include "exceptions/exception_macros.h"
#include "components/optimizers/adagrad_optimizer.h"
#include "utils/serializable.h"

namespace models {

    AdagradOptimizer::AdagradOptimizer(double learning_rate, double epsilon, double decay)
        : Optimizer(learning_rate, decay), epsilon_(epsilon) {
        if (learning_rate <= 0.0) {
            ML_THROW_PARAMETER_ERROR("learning_rate", "must be > 0", "AdagradOptimizer");
        }
        if (epsilon <= 0.0) {
            ML_THROW_PARAMETER_ERROR("epsilon", "must be > 0", "AdagradOptimizer");
        }
        if (decay < 0.0) {
            ML_THROW_PARAMETER_ERROR("decay", "must be >= 0", "AdagradOptimizer");
        }
    }

    void AdagradOptimizer::initialize_if_needed(int rows, int cols) {
        if (accumulated_squared_gradients_w_.rows() != rows ||
            accumulated_squared_gradients_w_.cols() != cols) {
            accumulated_squared_gradients_w_ = Eigen::MatrixXd::Zero(rows, cols);
        }
    }

    void AdagradOptimizer::initialize_if_needed(int size) {
        if (accumulated_squared_gradients_b_.size() != size) {
            accumulated_squared_gradients_b_ = Eigen::VectorXd::Zero(size);
        }
    }

    void AdagradOptimizer::update_weights(
        Eigen::Ref<Eigen::MatrixXd> weights,
        const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
        const double learning_rate = get_current_learning_rate();
        initialize_if_needed(weights.rows(), weights.cols());
        ++iterations_;

        accumulated_squared_gradients_w_.array() += gradient.array().square();
        weights.array() -= learning_rate * gradient.array() /
                           (accumulated_squared_gradients_w_.array().sqrt() + epsilon_);
    }

    void AdagradOptimizer::update_bias(
        Eigen::Ref<Eigen::VectorXd> bias,
        const Eigen::Ref<const Eigen::VectorXd>& gradient) {
        const double learning_rate = get_current_learning_rate();
        initialize_if_needed(bias.size());

        accumulated_squared_gradients_b_.array() += gradient.array().square();
        bias.array() -= learning_rate * gradient.array() /
                        (accumulated_squared_gradients_b_.array().sqrt() + epsilon_);
    }

    void AdagradOptimizer::reset() {
        iterations_ = 0;
        accumulated_squared_gradients_w_.resize(0, 0);
        accumulated_squared_gradients_b_.resize(0);
    }

    void AdagradOptimizer::serialize(std::ostream& out) const {
        Optimizer::serialize(out);
        utils::write_scalar(out, epsilon_);
        utils::write_eigen_matrix(out, accumulated_squared_gradients_w_);
        utils::write_eigen_vector(out, accumulated_squared_gradients_b_);
    }

    void AdagradOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        utils::read_scalar(in, epsilon_);
        utils::read_eigen_matrix(in, accumulated_squared_gradients_w_);
        utils::read_eigen_vector(in, accumulated_squared_gradients_b_);
    }

    std::unique_ptr<Optimizer> AdagradOptimizer::clone() const {
        auto copy = std::make_unique<AdagradOptimizer>(learning_rate_, epsilon_, decay_);
        copy->accumulated_squared_gradients_w_ = accumulated_squared_gradients_w_;
        copy->accumulated_squared_gradients_b_ = accumulated_squared_gradients_b_;
        copy->iterations_ = iterations_;
        return copy;
    }

} // namespace models
