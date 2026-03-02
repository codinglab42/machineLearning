#ifndef L1_REGULARIZER_H
#define L1_REGULARIZER_H

#include "regularizer.h"

namespace models {

    class L1Regularizer : public Regularizer {
    public:
        L1Regularizer(double strength = 0.01);
        ~L1Regularizer() override = default;
        
        double compute_loss(const Eigen::MatrixXd& weights) const override;
        double compute_loss(const Eigen::VectorXd& bias) const override;
        
        Eigen::MatrixXd compute_gradient(const Eigen::MatrixXd& weights) const override;
        Eigen::VectorXd compute_gradient(const Eigen::VectorXd& bias) const override;
        
        RegularizerType get_type() const override { return RegularizerType::L1; }
        std::string get_type_str() const override { return "l1"; }
        
        std::unique_ptr<Regularizer> clone() const override;
    };

} // namespace models

#endif

