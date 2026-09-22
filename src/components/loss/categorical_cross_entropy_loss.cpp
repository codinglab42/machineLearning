#include <Eigen/Dense>
#include "components/loss/categorical_cross_entropy_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double CategoricalCrossEntropyLoss::compute(const Eigen::VectorXd& y_true,
                                             const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "CategoricalCrossEntropyLoss");
    
    // Per vettori 1D usare ArrayXd
    Eigen::ArrayXd pred = y_pred.array().cwiseMax(1e-7).cwiseMin(1.0 - 1e-7);
    Eigen::ArrayXd true_val = y_true.array();
    
    double loss = -(true_val * pred.log()).sum() / static_cast<double>(y_true.size());
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("categorical_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

double CategoricalCrossEntropyLoss::compute(const Eigen::MatrixXd& y_true,
                                             const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "CategoricalCrossEntropyLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "CategoricalCrossEntropyLoss");
    
    // CORREZIONE: Usare ArrayXXd per matrici 2D (N x C)
    Eigen::ArrayXXd pred = y_pred.array().cwiseMax(1e-7).cwiseMin(1.0 - 1e-7);
    Eigen::ArrayXXd true_val = y_true.array();
    
    double loss = -(true_val * pred.log()).sum() / static_cast<double>(y_true.rows());
    
    if (std::isnan(loss) || std::isinf(loss)) {
        ML_THROW_MATH_ERROR("categorical_crossentropy", "loss is NaN/Inf");
    }
    
    return loss;
}

Eigen::MatrixXd CategoricalCrossEntropyLoss::gradient(const Eigen::MatrixXd& y_true,
                                                       const Eigen::MatrixXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.rows(), y_pred.rows(), "CategoricalCrossEntropyLoss");
    ML_CHECK_XY_SIZE(y_true.cols(), y_pred.cols(), "CategoricalCrossEntropyLoss");
    
    // CORREZIONE: clipping 2D e calcolo corretto del gradiente - (y_true / y_pred) / N
    Eigen::ArrayXXd clipped_pred = y_pred.array().cwiseMax(1e-7).cwiseMin(1.0 - 1e-7);
    double n = static_cast<double>(y_true.rows());

    Eigen::MatrixXd grad = (-(y_true.array() / clipped_pred) / n).matrix();
    
    ML_CHECK_NO_NAN(grad, "CategoricalCrossEntropyLoss", "gradient");
    ML_CHECK_NO_INF(grad, "CategoricalCrossEntropyLoss", "gradient");
    
    return grad;
}

} // namespace loss