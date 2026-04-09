#include "components/optimizers/adam_optimizer.h"
#include "utils/serializable.h"
#include <cmath>

namespace models {

    AdamOptimizer::AdamOptimizer(double learning_rate,
                             double beta1,
                             double beta2,
                             double epsilon,
                             double decay)
        : Optimizer(learning_rate, decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon) {
        
        // Validazione learning rate (se non già fatto da Optimizer)
        if (learning_rate <= 0.0) {
            ML_THROW_PARAMETER_ERROR("learning_rate", "must be > 0", "AdamOptimizer");
        }
        
        // Validazione decay (opzionale, può essere 0)
        if (decay < 0.0) {
            ML_THROW_PARAMETER_ERROR("decay", "must be >= 0", "AdamOptimizer");
        }
        
        // Validazione parametri specifici di Adam
        if (beta1 <= 0.0 || beta1 >= 1.0) {
            ML_THROW_PARAMETER_ERROR("beta1", "must be in (0, 1)", "AdamOptimizer");
        }
        
        if (beta2 <= 0.0 || beta2 >= 1.0) {
            ML_THROW_PARAMETER_ERROR("beta2", "must be in (0, 1)", "AdamOptimizer");
        }
        
        if (epsilon <= 0.0) {
            ML_THROW_PARAMETER_ERROR("epsilon", "must be > 0", "AdamOptimizer");
        }
    }
    void AdamOptimizer::initialize_if_needed(int rows, int cols) {
        if (m_w_.rows() != rows || m_w_.cols() != cols) {
            m_w_ = Eigen::MatrixXd::Zero(rows, cols);
            v_w_ = Eigen::MatrixXd::Zero(rows, cols);
        }
    }

    void AdamOptimizer::initialize_if_needed(int size) {
        if (m_b_.size() != size) {
            m_b_ = Eigen::VectorXd::Zero(size);
            v_b_ = Eigen::VectorXd::Zero(size);
        }
    }

    void AdamOptimizer::update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) {
        double lr = get_current_learning_rate();
        initialize_if_needed(weights.rows(), weights.cols());
        
        iterations_++;
        
        // Aggiorna momenti
        m_w_ = beta1_ * m_w_ + (1.0 - beta1_) * gradient;
        v_w_ = beta2_ * v_w_ + (1.0 - beta2_) * gradient.array().square().matrix();
        
        // Correzione bias
        double m_correction = 1.0 / (1.0 - std::pow(beta1_, iterations_));
        double v_correction = 1.0 / (1.0 - std::pow(beta2_, iterations_));
        
        // Aggiorna pesi - usando array() per operazioni elemento-per-elemento
        Eigen::ArrayXXd m_corrected = m_w_.array() * m_correction;
        Eigen::ArrayXXd v_corrected = v_w_.array() * v_correction;
        Eigen::ArrayXXd denom = v_corrected.sqrt() + epsilon_;
        
        weights -= lr * (m_corrected / denom).matrix();
    }

    void AdamOptimizer::update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) {
        double lr = get_current_learning_rate();
        initialize_if_needed(bias.size());
        
        m_b_ = beta1_ * m_b_ + (1.0 - beta1_) * gradient;
        v_b_ = beta2_ * v_b_ + (1.0 - beta2_) * gradient.array().square().matrix();
        
        double m_correction = 1.0 / (1.0 - std::pow(beta1_, iterations_));
        double v_correction = 1.0 / (1.0 - std::pow(beta2_, iterations_));
        
        Eigen::ArrayXd m_corrected = m_b_.array() * m_correction;
        Eigen::ArrayXd v_corrected = v_b_.array() * v_correction;
        Eigen::ArrayXd denom = v_corrected.sqrt() + epsilon_;
        
        bias -= lr * (m_corrected / denom).matrix();
    }

    void AdamOptimizer::reset() {
        iterations_ = 0;
        m_w_.resize(0, 0);
        v_w_.resize(0, 0);
        m_b_.resize(0);
        v_b_.resize(0);
    }

    void AdamOptimizer::serialize(std::ostream& out) const {
        Optimizer::serialize(out);
        
        utils::write_scalar(out, beta1_);
        utils::write_scalar(out, beta2_);
        utils::write_scalar(out, epsilon_);
        
        // Usa le nuove funzioni
        utils::write_eigen_matrix(out, m_w_);
        utils::write_eigen_matrix(out, v_w_);
        utils::write_eigen_vector(out, m_b_);
        utils::write_eigen_vector(out, v_b_);
    }

    void AdamOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        
        utils::read_scalar(in, beta1_);
        utils::read_scalar(in, beta2_);
        utils::read_scalar(in, epsilon_);
        
        // Usa le nuove funzioni
        utils::read_eigen_matrix(in, m_w_);
        utils::read_eigen_matrix(in, v_w_);
        utils::read_eigen_vector(in, m_b_);
        utils::read_eigen_vector(in, v_b_);
    }

    std::unique_ptr<Optimizer> AdamOptimizer::clone() const {
        auto clone = std::make_unique<AdamOptimizer>(learning_rate_, beta1_, beta2_, epsilon_, decay_);
        clone->m_w_ = m_w_;
        clone->v_w_ = v_w_;
        clone->m_b_ = m_b_;
        clone->v_b_ = v_b_;
        clone->iterations_ = iterations_;
        return clone;
    }

} // namespace models