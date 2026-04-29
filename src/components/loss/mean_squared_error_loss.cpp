#include <Eigen/Dense>
#include "components/loss/mean_squared_error_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double MeanSquaredErrorLoss::compute(const Eigen::VectorXd& y_true,
                                      const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "MeanSquaredErrorLoss");
    
    double loss = (y_pred - y_true).array().square().mean();
    
    if (loss < 0 && loss > -1e-10) {
        loss = 0.0;
    }
    
    ML_CHECK_FINITE(loss, "MeanSquaredErrorLoss", "compute");
    return loss;
}

double MeanSquaredErrorLoss::compute(const Eigen::MatrixXd& y_true,
                                      const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "MeanSquaredErrorLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "MeanSquaredErrorLoss");
    
    double loss = (y_pred - y_true).array().square().mean();
    
    if (loss < 0 && loss > -1e-10) {
        loss = 0.0;
    }
    
    ML_CHECK_FINITE(loss, "MeanSquaredErrorLoss", "compute");
    return loss;
}

Eigen::MatrixXd MeanSquaredErrorLoss::gradient(const Eigen::MatrixXd& y_true,
                                                const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "MeanSquaredErrorLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "MeanSquaredErrorLoss");
    
    Eigen::MatrixXd grad = 2.0 * (y_pred - y_true) / y_true.rows();
    
    ML_CHECK_NO_NAN(grad, "MeanSquaredErrorLoss", "gradient");
    ML_CHECK_NO_INF(grad, "MeanSquaredErrorLoss", "gradient");
    
    return grad;
}

} // namespace loss