#ifndef HUBER_LOSS_H
#define HUBER_LOSS_H

#include "components/loss/loss.h"

namespace loss {

class HuberLoss : public Loss {
public:
    explicit HuberLoss(double delta = 1.0) : delta_(delta) {}
    
    double compute(const Eigen::VectorXd& y_true, 
                  const Eigen::VectorXd& y_pred) const override;
    
    double compute(const Eigen::MatrixXd& y_true,
                  const Eigen::MatrixXd& y_pred) const override;
    
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                            const Eigen::MatrixXd& y_pred) const override;
    
    std::string name() const override { return "huber"; }
    
    void set_delta(double delta) { delta_ = delta; }
    double get_delta() const { return delta_; }
    
private:
    double delta_;
};

} // namespace loss

#endif // HUBER_LOSS_H