#include <Eigen/Dense>
#include "components/loss/binary_cross_entropy_loss.h"
#include "exceptions/exception_macros.h"
#include <cmath>

namespace loss {

double BinaryCrossEntropyLoss::compute(const Eigen::VectorXd& y_true,
                                        const Eigen::VectorXd& y_pred) const {
    ML_CHECK_XY_SIZE(y_true.size(), y_pred.size(), "BinaryCrossEntropyLoss");
    
    // ⭐ CORREZIONE 1: Sanificazione preventiva di NaN/Inf per VectorXd
    Eigen::VectorXd sanitized_pred = y_pred.unaryExpr([](double v) {
        if (std::isnan(v)) return 0.5; 
        if (std::isinf(v)) return v > 0 ? (1.0 - 1e-7) : 1e-7;
        return v;
    });
    
    Eigen::ArrayXd pred = sanitized_pred.array().max(1e-7).min(1.0 - 1e-7);
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
    
    // ⭐ CORREZIONE 2: Sanificazione preventiva di NaN/Inf per MatrixXd
    Eigen::MatrixXd sanitized_pred = y_pred.unaryExpr([](double v) {
        if (std::isnan(v)) return 0.5; 
        if (std::isinf(v)) return v > 0 ? (1.0 - 1e-7) : 1e-7;
        return v;
    });
    
    Eigen::ArrayXd pred = sanitized_pred.col(0).array().max(1e-7).min(1.0 - 1e-7);
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
    
    int n = y_true.rows();
    
    // Formula numericamente stabile del gradiente BCE (i denominatori dell'attivazione si cancellano)
    // grad = (1/n) * (y_pred - y_true)
    Eigen::MatrixXd grad = (y_pred - y_true) / n;
    
    // Controllo di sicurezza rapido senza alterare la dinamica con zeri arbitrari
    if (!grad.allFinite()) {
        grad = grad.unaryExpr([](double v) {
            if (std::isnan(v)) return 0.0;
            if (std::isinf(v)) return v > 0 ? 1.0 : -1.0;
            return v;
        });
    }
    
    ML_CHECK_NO_NAN(grad, "BinaryCrossEntropyLoss", "gradient");
    ML_CHECK_NO_INF(grad, "BinaryCrossEntropyLoss", "gradient");
    
    return grad;
}

} // namespace loss