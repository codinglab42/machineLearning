#include <Eigen/Dense>
#include "components/loss/categorical_cross_entropy_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double CategoricalCrossEntropyLoss::compute(const Eigen::VectorXd& y_true,
                                             const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "CategoricalCrossEntropyLoss");
    
    Eigen::ArrayXd pred = y_pred.array().max(1e-7);
    Eigen::ArrayXd true_val = y_true.array();
    
    double loss = -(true_val * pred.log()).sum() / y_true.size();
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("categorical_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

double CategoricalCrossEntropyLoss::compute(const Eigen::MatrixXd& y_true,
                                             const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "CategoricalCrossEntropyLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "CategoricalCrossEntropyLoss");
    
    Eigen::ArrayXd pred = y_pred.array().max(1e-7);
    Eigen::ArrayXd true_val = y_true.array();
    
    double loss = -(true_val * pred.log()).sum() / y_true.rows();
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("categorical_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

Eigen::MatrixXd CategoricalCrossEntropyLoss::gradient(const Eigen::MatrixXd& y_true,
                                                       const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "CategoricalCrossEntropyLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "CategoricalCrossEntropyLoss");
    
    Eigen::MatrixXd clipped_pred = y_pred.array().max(1e-7);

    int n = y_true.rows();
    Eigen::MatrixXd grad = (clipped_pred - y_true)/n;
    
    ML_CHECK_NO_NAN(grad, "CategoricalCrossEntropyLoss", "gradient");
    ML_CHECK_NO_INF(grad, "CategoricalCrossEntropyLoss", "gradient");
    
    return grad;
}

} // namespace loss