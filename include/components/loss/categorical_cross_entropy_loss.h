#ifndef CATEGORICAL_CROSS_ENTROPY_LOSS_H
#define CATEGORICAL_CROSS_ENTROPY_LOSS_H

#include "components/loss/loss.h"

namespace loss {

class CategoricalCrossEntropyLoss : public Loss {
public:
    double compute(const Eigen::VectorXd& y_true, 
                  const Eigen::VectorXd& y_pred) const override;
    
    double compute(const Eigen::MatrixXd& y_true,
                  const Eigen::MatrixXd& y_pred) const override;
    
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                            const Eigen::MatrixXd& y_pred) const override;
    
    std::string name() const override { return "categorical_crossentropy"; }
};

} // namespace loss

#endif // CATEGORICAL_CROSS_ENTROPY_LOSS_H