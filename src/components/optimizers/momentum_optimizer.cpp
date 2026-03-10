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
        
        out.write(reinterpret_cast<const char*>(&momentum_), sizeof(double));
        
        bool nesterov = nesterov_;
        out.write(reinterpret_cast<const char*>(&nesterov), sizeof(bool));
        
        utils::eigen_utils::serialize_eigen(velocity_w_, out);
        utils::eigen_utils::serialize_eigen_vector(velocity_b_, out);
    }

    void MomentumOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        
        in.read(reinterpret_cast<char*>(&momentum_), sizeof(double));
        
        bool nesterov;
        in.read(reinterpret_cast<char*>(&nesterov), sizeof(bool));
        nesterov_ = nesterov;
        
        utils::eigen_utils::deserialize_eigen(velocity_w_, in);
        utils::eigen_utils::deserialize_eigen_vector(velocity_b_, in);
    }

    std::unique_ptr<Optimizer> MomentumOptimizer::clone() const {
        auto clone = std::make_unique<MomentumOptimizer>(learning_rate_, momentum_, decay_, nesterov_);
        clone->velocity_w_ = velocity_w_;
        clone->velocity_b_ = velocity_b_;
        clone->iterations_ = iterations_;
        return clone;
    }

} // namespace models

