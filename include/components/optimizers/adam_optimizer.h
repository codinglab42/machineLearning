#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "optimizer.h"
#include <Eigen/Dense>

namespace models {

    class AdamOptimizer : public Optimizer {
    public:
        AdamOptimizer(double learning_rate = 0.001,
                     double beta1 = 0.9,
                     double beta2 = 0.999,
                     double epsilon = 1e-8,
                     double decay = 0.0);
        ~AdamOptimizer() override = default;
        
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) override;
        void update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) override;
        
        void reset() override;
        
        OptimizerType get_type() const override { return OptimizerType::ADAM; }
        std::string get_type_str() const override { return "adam"; }
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::unique_ptr<Optimizer> clone() const override;
        
    private:
        double beta1_;
        double beta2_;
        double epsilon_;
        
        Eigen::MatrixXd m_w_;  // primo momento (media)
        Eigen::MatrixXd v_w_;  // secondo momento (varianza non centrata)
        Eigen::VectorXd m_b_;
        Eigen::VectorXd v_b_;
        
        void initialize_if_needed(int rows, int cols);
        void initialize_if_needed(int size);
    };

} // namespace models

#endif
