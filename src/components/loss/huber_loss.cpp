#include <Eigen/Dense>
#include "components/loss/huber_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double HuberLoss::compute(const Eigen::VectorXd& y_true,
                          const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "HuberLoss");
    
    // Convertiamo esplicitamente i riferimenti a VectorXd in MatrixXd.
    // In Eigen è un'operazione a costo zero (no-copy conversion).
    return compute(static_cast<const Eigen::MatrixXd&>(y_true), 
                   static_cast<const Eigen::MatrixXd&>(y_pred));
}

double HuberLoss::compute(const Eigen::MatrixXd& y_true,
                          const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "HuberLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "HuberLoss");
    
    // Calcolo del residuo assoluto coefficiente per coefficiente
    Eigen::MatrixXd residual = (y_pred - y_true).cwiseAbs();
    
    double loss = 0.0;
    int total_elements = residual.size();
    
    // Accesso lineare sicuro sul buffer dei dati nativi
    for (int i = 0; i < total_elements; ++i) {
        double r = residual.data()[i];
        if (r <= delta_) {
            loss += 0.5 * r * r;
        } else {
            loss += delta_ * (r - 0.5 * delta_);
        }
    }
    
    loss /= total_elements;
    
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
    Eigen::MatrixXd grad(residual.rows(), residual.cols());
    
    int total_elements = residual.size();
    
    for (int i = 0; i < total_elements; ++i) {
        double r = residual.data()[i];
        if (std::abs(r) <= delta_) {
            grad.data()[i] = r;
        } else {
            grad.data()[i] = delta_ * ((r > 0.0) ? 1.0 : -1.0);
        }
    }
    
    grad /= total_elements;
    
    ML_CHECK_NO_NAN(grad, "HuberLoss", "gradient");
    ML_CHECK_NO_INF(grad, "HuberLoss", "gradient");
    
    return grad;
}

} // namespace loss