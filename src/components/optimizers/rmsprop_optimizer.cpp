#include "exceptions/exception_macros.h"
#include "components/optimizers/rmsprop_optimizer.h"
#include "utils/serializable.h"

namespace models {

    RMSpropOptimizer::RMSpropOptimizer(double learning_rate,
                                       double rho,
                                       double epsilon,
                                       double decay,
                                       double momentum)
        : Optimizer(learning_rate, decay),
          rho_(rho),
          epsilon_(epsilon),
          momentum_(momentum) {
        if (learning_rate <= 0.0) {
            ML_THROW_PARAMETER_ERROR("learning_rate", "must be > 0", "RMSpropOptimizer");
        }
        if (rho <= 0.0 || rho >= 1.0) {
            ML_THROW_PARAMETER_ERROR("rho", "must be in (0, 1)", "RMSpropOptimizer");
        }
        if (epsilon <= 0.0) {
            ML_THROW_PARAMETER_ERROR("epsilon", "must be > 0", "RMSpropOptimizer");
        }
        if (decay < 0.0) {
            ML_THROW_PARAMETER_ERROR("decay", "must be >= 0", "RMSpropOptimizer");
        }
        if (momentum < 0.0 || momentum >= 1.0) {
            ML_THROW_PARAMETER_ERROR("momentum", "must be in [0, 1)", "RMSpropOptimizer");
        }
    }

    void RMSpropOptimizer::initialize_if_needed(int rows, int cols) {
        if (mean_squared_gradients_w_.rows() != rows ||
            mean_squared_gradients_w_.cols() != cols) {
            mean_squared_gradients_w_ = Eigen::MatrixXd::Zero(rows, cols);
            velocity_w_ = Eigen::MatrixXd::Zero(rows, cols);
        }
    }

    void RMSpropOptimizer::initialize_if_needed(int size) {
        if (mean_squared_gradients_b_.size() != size) {
            mean_squared_gradients_b_ = Eigen::VectorXd::Zero(size);
            velocity_b_ = Eigen::VectorXd::Zero(size);
        }
    }

    void RMSpropOptimizer::update_weights(
        Eigen::Ref<Eigen::MatrixXd> weights,
        const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
        const double learning_rate = get_current_learning_rate();
        initialize_if_needed(weights.rows(), weights.cols());
        ++iterations_;

        mean_squared_gradients_w_.array() =
            rho_ * mean_squared_gradients_w_.array() +
            (1.0 - rho_) * gradient.array().square();
        const Eigen::ArrayXXd step =
            learning_rate * gradient.array() /
            (mean_squared_gradients_w_.array().sqrt() + epsilon_);

        if (momentum_ > 0.0) {
            velocity_w_ = momentum_ * velocity_w_ + step.matrix();
            weights -= velocity_w_;
        } else {
            weights -= step.matrix();
        }
    }

    void RMSpropOptimizer::update_bias(
        Eigen::Ref<Eigen::VectorXd> bias,
        const Eigen::Ref<const Eigen::VectorXd>& gradient) {
        const double learning_rate = get_current_learning_rate();
        initialize_if_needed(bias.size());

        mean_squared_gradients_b_.array() =
            rho_ * mean_squared_gradients_b_.array() +
            (1.0 - rho_) * gradient.array().square();
        const Eigen::ArrayXd step =
            learning_rate * gradient.array() /
            (mean_squared_gradients_b_.array().sqrt() + epsilon_);

        if (momentum_ > 0.0) {
            velocity_b_ = momentum_ * velocity_b_ + step.matrix();
            bias -= velocity_b_;
        } else {
            bias -= step.matrix();
        }
    }

    void RMSpropOptimizer::reset() {
        iterations_ = 0;
        mean_squared_gradients_w_.resize(0, 0);
        mean_squared_gradients_b_.resize(0);
        velocity_w_.resize(0, 0);
        velocity_b_.resize(0);
    }

    void RMSpropOptimizer::serialize(std::ostream& out) const {
        Optimizer::serialize(out);
        utils::write_scalar(out, rho_);
        utils::write_scalar(out, epsilon_);
        utils::write_scalar(out, momentum_);
        utils::write_eigen_matrix(out, mean_squared_gradients_w_);
        utils::write_eigen_vector(out, mean_squared_gradients_b_);
        utils::write_eigen_matrix(out, velocity_w_);
        utils::write_eigen_vector(out, velocity_b_);
    }

    void RMSpropOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        utils::read_scalar(in, rho_);
        utils::read_scalar(in, epsilon_);
        utils::read_scalar(in, momentum_);
        utils::read_eigen_matrix(in, mean_squared_gradients_w_);
        utils::read_eigen_vector(in, mean_squared_gradients_b_);
        utils::read_eigen_matrix(in, velocity_w_);
        utils::read_eigen_vector(in, velocity_b_);
    }

    std::unique_ptr<Optimizer> RMSpropOptimizer::clone() const {
        auto copy = std::make_unique<RMSpropOptimizer>(
            learning_rate_, rho_, epsilon_, decay_, momentum_);
        copy->mean_squared_gradients_w_ = mean_squared_gradients_w_;
        copy->mean_squared_gradients_b_ = mean_squared_gradients_b_;
        copy->velocity_w_ = velocity_w_;
        copy->velocity_b_ = velocity_b_;
        copy->iterations_ = iterations_;
        return copy;
    }

} // namespace models
