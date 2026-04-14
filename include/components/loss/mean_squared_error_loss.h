#ifndef MEAN_SQUARED_ERROR_LOSS_H
#define MEAN_SQUARED_ERROR_LOSS_H

#include "components/loss/loss.h"

namespace loss {

class MeanSquaredErrorLoss : public Loss {
public:
    double compute(const Eigen::VectorXd& y_true, 
                  const Eigen::VectorXd& y_pred) const override;
    
    double compute(const Eigen::MatrixXd& y_true,
                  const Eigen::MatrixXd& y_pred) const override;
    
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                            const Eigen::MatrixXd& y_pred) const override;
    
    std::string name() const override { return "mse"; }
};

} // namespace loss

#endif // MEAN_SQUARED_ERROR_LOSS_H