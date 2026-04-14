#ifndef BINARY_CROSS_ENTROPY_LOSS_H
#define BINARY_CROSS_ENTROPY_LOSS_H

#include "components/loss/loss.h"

namespace loss {

class BinaryCrossEntropyLoss : public Loss {
public:
    double compute(const Eigen::VectorXd& y_true, 
                  const Eigen::VectorXd& y_pred) const override;
    
    double compute(const Eigen::MatrixXd& y_true,
                  const Eigen::MatrixXd& y_pred) const override;
    
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                            const Eigen::MatrixXd& y_pred) const override;
    
    std::string name() const override { return "binary_crossentropy"; }
};

} // namespace loss

#endif // BINARY_CROSS_ENTROPY_LOSS_H