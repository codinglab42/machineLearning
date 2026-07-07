#include <Eigen/Dense>
#include "components/loss/mean_absolute_error_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double MeanAbsoluteErrorLoss::compute(const Eigen::VectorXd& y_true,
                                       const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "MeanAbsoluteErrorLoss");
    
    double loss = (y_pred - y_true).array().abs().mean();
    
    ML_CHECK_FINITE(loss, "MeanAbsoluteErrorLoss", "compute");
    return loss;
}

double MeanAbsoluteErrorLoss::compute(const Eigen::MatrixXd& y_true,
                                       const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "MeanAbsoluteErrorLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "MeanAbsoluteErrorLoss");
    
    double loss = (y_pred - y_true).array().abs().mean();
    
    ML_CHECK_FINITE(loss, "MeanAbsoluteErrorLoss", "compute");
    return loss;
}

Eigen::MatrixXd MeanAbsoluteErrorLoss::gradient(const Eigen::MatrixXd& y_true,
                                                 const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "MeanAbsoluteErrorLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "MeanAbsoluteErrorLoss");
    
    // d/dx |y_pred - y_true| = sign(y_pred - y_true)
    // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    Eigen::MatrixXd grad = (y_pred - y_true).array().sign().matrix() / y_true.size();
    
    ML_CHECK_NO_NAN(grad, "MeanAbsoluteErrorLoss", "gradient");
    ML_CHECK_NO_INF(grad, "MeanAbsoluteErrorLoss", "gradient");
    
    return grad;
}

} // namespace loss