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
          epsilon_(epsilon) {}

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
        
        // Aggiorna pesi
        Eigen::MatrixXd denom = (v_w_.array().sqrt() / std::sqrt(v_correction)).matrix();
        denom.array() += epsilon_;
        
        weights -= lr * (m_w_ * m_correction).array() / denom.array();
    }

    void AdamOptimizer::update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) {
        double lr = get_current_learning_rate();
        initialize_if_needed(bias.size());
        
        m_b_ = beta1_ * m_b_ + (1.0 - beta1_) * gradient;
        v_b_ = beta2_ * v_b_ + (1.0 - beta2_) * gradient.array().square().matrix();
        
        double m_correction = 1.0 / (1.0 - std::pow(beta1_, iterations_));
        double v_correction = 1.0 / (1.0 - std::pow(beta2_, iterations_));
        
        Eigen::VectorXd denom = (v_b_.array().sqrt() / std::sqrt(v_correction)).matrix();
        denom.array() += epsilon_;
        
        bias -= lr * (m_b_ * m_correction).array() / denom.array();
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
        
        using namespace utils;
        out.write(reinterpret_cast<const char*>(&beta1_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&beta2_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&epsilon_), sizeof(double));
        
        serialize_matrix(out, m_w_);
        serialize_matrix(out, v_w_);
        serialize_vector(out, m_b_);
        serialize_vector(out, v_b_);
    }

    void AdamOptimizer::deserialize(std::istream& in) {
        Optimizer::deserialize(in);
        
        using namespace utils;
        in.read(reinterpret_cast<char*>(&beta1_), sizeof(double));
        in.read(reinterpret_cast<char*>(&beta2_), sizeof(double));
        in.read(reinterpret_cast<char*>(&epsilon_), sizeof(double));
        
        m_w_ = deserialize_matrix(in);
        v_w_ = deserialize_matrix(in);
        m_b_ = deserialize_vector(in);
        v_b_ = deserialize_vector(in);
    }

    std::unique_ptr<Optimizer> AdamOptimizer::clone() const {
        auto clone = std::make_unique<AdamOptimizer>(learning_rate_, beta1_, beta2_, epsilon_, decay_);
        clone->m_w_ = m_w_;
        clone->v_w_ = v_w_;
        clone->m_b_ = m_b_;
        clone->v_b_ = v_b_;
        return clone;
    }

} // namespace models
