#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>
#include <string>
#include <memory>
#include "exceptions/exception_macros.h"

namespace loss {

class Loss {
public:
    virtual ~Loss() = default;
    
    virtual double compute(const Eigen::VectorXd& y_true, 
                          const Eigen::VectorXd& y_pred) const = 0;
    
    virtual double compute(const Eigen::MatrixXd& y_true, 
                          const Eigen::MatrixXd& y_pred) const = 0;
    
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_true,
                                     const Eigen::MatrixXd& y_pred) const = 0;
    
    virtual std::string name() const = 0;
};

} // namespace loss

#endif // LOSS_H