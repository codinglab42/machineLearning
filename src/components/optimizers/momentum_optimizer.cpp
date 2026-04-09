#include "components/optimizers/momentum_optimizer.h"
#include "utils/serializable.h"
#include <cmath>

namespace models {

    MomentumOptimizer::MomentumOptimizer(double learning_rate, 
                                         double momentum,
                                         double decay,
                                         bool nesterov)
        : Optimizer(learning_rate, decay),
          momentum_(momentum),
          nesterov_(nesterov) {}

    void MomentumOptimizer::initialize_if_needed(int rows, int cols) {
        if (velocity_w_.rows() != rows || velocity_w_.cols() != cols) {
            velocity_w_ = Eigen::MatrixXd::Zero(rows, cols);
        }
    }

    void MomentumOptimizer::initialize_if_needed(int size) {
        if (velocity_b_.size() != size) {
            velocity_b_ = Eigen::VectorXd::Zero(size);
        }
    }

    void MomentumOptimizer::update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) {
        double lr = get_current_learning_rate();
        initialize_if_needed(weights.rows(), weights.cols());
        
        iterations_++;
        
        // Aggiorna velocità
        velocity_w_ = momentum_ * velocity_w_ - lr * gradient;
        
        if (nesterov_) {
            // Nesterov momentum
            weights += momentum_ * velocity_w_ - lr * gradient;
        } else {
            // Standard momentum
            weights += velocity_w_;
        }
    }

    void MomentumOptimizer::update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) {
        double lr = get_current_learning_rate();
        initialize_if_needed(bias.size());
        
        velocity_b_ = momentum_ * velocity_b_ - lr * gradient;
        
        if (nesterov_) {
            bias += momentum_ * velocity_b_ - lr * gradient;
        } else {
            bias += velocity_b_;
        }
    }

    void MomentumOptimizer::reset() {
        iterations_ = 0;
        velocity_w_.resize(0, 0);
        velocity_b_.resize(0);
    }

    void MomentumOptimizer::serialize(std::ostream& out) const {
        Optimizer::serialize(out);
        
        utils::write_scalar(out, momentum_);
        utils::write_scalar(out, nesterov_);
        
        // Usa le nuove funzioni
        utils::write_eigen_matrix(out, velocity_w_);
        utils::write_eigen_vector(out, velocity_b_);
    }

    void MomentumOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        
        utils::read_scalar(in, momentum_);
        utils::read_scalar(in, nesterov_);
        
        // Usa le nuove funzioni
        utils::read_eigen_matrix(in, velocity_w_);
        utils::read_eigen_vector(in, velocity_b_);
    }
    std::unique_ptr<Optimizer> MomentumOptimizer::clone() const {
        auto clone = std::make_unique<MomentumOptimizer>(learning_rate_, momentum_, decay_, nesterov_);
        clone->velocity_w_ = velocity_w_;
        clone->velocity_b_ = velocity_b_;
        clone->iterations_ = iterations_;
        return clone;
    }

} // namespace models

