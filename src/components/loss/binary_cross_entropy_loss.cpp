#include "components/loss/binary_cross_entropy_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double BinaryCrossEntropyLoss::compute(const Eigen::VectorXd& y_true,
                                        const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "BinaryCrossEntropyLoss");
    
    Eigen::ArrayXd pred = y_pred.array().max(1e-7).min(1.0 - 1e-7);
    Eigen::ArrayXd true_val = y_true.array();
    
    double loss = -(true_val * pred.log() + (1 - true_val) * (1 - pred).log()).mean();
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("binary_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

double BinaryCrossEntropyLoss::compute(const Eigen::MatrixXd& y_true,
                                        const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "BinaryCrossEntropyLoss");
    ML_CHECK_PARAM(y_true.cols() == 1 && y_pred.cols() == 1, "y_true/y_pred",
                  "must be column vectors", "BinaryCrossEntropyLoss");
    
    Eigen::ArrayXd pred = y_pred.col(0).array().max(1e-7).min(1.0 - 1e-7);
    Eigen::ArrayXd true_val = y_true.col(0).array();
    
    double loss = -(true_val * pred.log() + (1 - true_val) * (1 - pred).log()).mean();
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("binary_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

Eigen::MatrixXd BinaryCrossEntropyLoss::gradient(const Eigen::MatrixXd& y_true,
                                                  const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "BinaryCrossEntropyLoss");
    
    Eigen::MatrixXd clipped_pred = y_pred.array().max(1e-7).min(1.0 - 1e-7);
    Eigen::MatrixXd grad = clipped_pred - y_true;
    
    ML_CHECK_NO_NAN(grad, "BinaryCrossEntropyLoss", "gradient");
    ML_CHECK_NO_INF(grad, "BinaryCrossEntropyLoss", "gradient");
    
    return grad;
}

} // namespace loss