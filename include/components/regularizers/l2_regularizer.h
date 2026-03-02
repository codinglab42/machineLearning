#ifndef L2_REGULARIZER_H
#define L2_REGULARIZER_H

#include "regularizer.h"

namespace models {

    class L2Regularizer : public Regularizer {
    public:
        L2Regularizer(double strength = 0.01);
        ~L2Regularizer() override = default;
        
        double compute_loss(const Eigen::MatrixXd& weights) const override;
        double compute_loss(const Eigen::VectorXd& bias) const override;
        
        Eigen::MatrixXd compute_gradient(const Eigen::MatrixXd& weights) const override;
        Eigen::VectorXd compute_gradient(const Eigen::VectorXd& bias) const override;
        
        RegularizerType get_type() const override { return RegularizerType::L2; }
        std::string get_type_str() const override { return "l2"; }
        
        std::unique_ptr<Regularizer> clone() const override;
    };

} // namespace models

#endif

