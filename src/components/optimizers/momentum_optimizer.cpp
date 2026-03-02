#include "components/optimizers/momentum_optimizer.h"
#include "utils/serializable.h"

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
        
        // Aggiorna velocità
        velocity_w_ = momentum_ * velocity_w_ - lr * gradient;
        
        if (nesterov_) {
            // Nesterov: anticipa la correzione
            weights += momentum_ * velocity_w_ - lr * gradient;
        } else {
            // Momentum standard
            weights += velocity_w_;
        }
        
        iterations_++;
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
        
        using namespace utils;
        out.write(reinterpret_cast<const char*>(&momentum_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&nesterov_), sizeof(bool));
        
        serialize_matrix(out, velocity_w_);
        serialize_vector(out, velocity_b_);
    }

    void MomentumOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        
        using namespace utils;
        in.read(reinterpret_cast<char*>(&momentum_), sizeof(double));
        in.read(reinterpret_cast<char*>(&nesterov_), sizeof(bool));
        
        velocity_w_ = deserialize_matrix(in);
        velocity_b_ = deserialize_vector(in);
    }

    std::unique_ptr<Optimizer> MomentumOptimizer::clone() const {
        auto clone = std::make_unique<MomentumOptimizer>(learning_rate_, momentum_, decay_, nesterov_);
        clone->velocity_w_ = velocity_w_;
        clone->velocity_b_ = velocity_b_;
        return clone;
    }

} // namespace models
