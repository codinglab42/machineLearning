#ifndef ELASTIC_NET_REGULARIZER_H
#define ELASTIC_NET_REGULARIZER_H

#include "regularizer.h"

namespace models {

    class ElasticNetRegularizer : public Regularizer {
    public:
        ElasticNetRegularizer(double strength = 0.01, double l1_ratio = 0.5);
        ~ElasticNetRegularizer() override = default;
        
        double compute_loss(const Eigen::MatrixXd& weights) const override;
        double compute_loss(const Eigen::VectorXd& bias) const override;
        
        Eigen::MatrixXd compute_gradient(const Eigen::MatrixXd& weights) const override;
        Eigen::VectorXd compute_gradient(const Eigen::VectorXd& bias) const override;
        
        RegularizerType get_type() const override { return RegularizerType::ELASTIC_NET; }
        std::string get_type_str() const override { return "elastic_net"; }
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::unique_ptr<Regularizer> clone() const override;
        
        // Getter specifici
        double get_l1_ratio() const { return l1_ratio_; }
        void set_l1_ratio(double ratio) { l1_ratio_ = ratio; }
        
    private:
        double l1_ratio_;  // bilanciamento tra L1 e L2 (0 = solo L2, 1 = solo L1)
    };

} // namespace models

#endif

