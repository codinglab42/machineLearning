#include "components/loss/huber_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

// RIMUOVI QUESTO - è già definito nell'header:
// HuberLoss::HuberLoss(double delta) : delta_(delta) {}

double HuberLoss::compute(const Eigen::VectorXd& y_true,
                          const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "HuberLoss");
    
    Eigen::ArrayXd residual = (y_pred - y_true).array().abs();
    double loss = 0.0;
    
    for (int i = 0; i < residual.size(); ++i) {
        if (residual(i) <= delta_) {
            loss += 0.5 * residual(i) * residual(i);
        } else {
            loss += delta_ * (residual(i) - 0.5 * delta_);
        }
    }
    
    loss /= residual.size();
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("huber", "loss is NaN/Inf");
    }
    
    return loss;
}

double HuberLoss::compute(const Eigen::MatrixXd& y_true,
                          const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "HuberLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "HuberLoss");
    
    Eigen::ArrayXd residual = (y_pred - y_true).array().abs();
    double loss = 0.0;
    
    for (int i = 0; i < residual.rows(); ++i) {
        for (int j = 0; j < residual.cols(); ++j) {
            int idx = i * residual.cols() + j;
            if (residual(idx) <= delta_) {
                loss += 0.5 * residual(idx) * residual(idx);
            } else {
                loss += delta_ * (residual(idx) - 0.5 * delta_);
            }
        }
    }
    
    loss /= (residual.rows() * residual.cols());
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("huber", "loss is NaN/Inf");
    }
    
    return loss;
}

Eigen::MatrixXd HuberLoss::gradient(const Eigen::MatrixXd& y_true,
                                     const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "HuberLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "HuberLoss");
    
    Eigen::MatrixXd residual = y_pred - y_true;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(residual.rows(), residual.cols());
    
    for (int i = 0; i < residual.rows(); ++i) {
        for (int j = 0; j < residual.cols(); ++j) {
            double r = residual(i, j);
            if (std::abs(r) <= delta_) {
                grad(i, j) = r;
            } else {
                grad(i, j) = delta_ * (r > 0 ? 1.0 : -1.0);
            }
        }
    }
    
    grad /= (y_true.rows() * y_true.cols());
    
    ML_CHECK_NO_NAN(grad, "HuberLoss", "gradient");
    ML_CHECK_NO_INF(grad, "HuberLoss", "gradient");
    
    return grad;
}

} // namespace loss