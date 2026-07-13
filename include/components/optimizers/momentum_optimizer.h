#ifndef MOMENTUM_OPTIMIZER_H
#define MOMENTUM_OPTIMIZER_H

#include "optimizer.h"
#include <Eigen/Dense>

namespace models {

    class MomentumOptimizer : public Optimizer {
    public:
        MomentumOptimizer(double learning_rate = 0.01, 
                         double momentum = 0.9,
                         double decay = 0.0,
                         bool nesterov = false);
        ~MomentumOptimizer() override = default;
        
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) override;
        void update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) override;
        
        void reset() override;
        
        OptimizerType get_type() const override { return OptimizerType::MOMENTUM; }
        std::string get_type_str() const override { return "momentum"; }
        
        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;
        
        std::unique_ptr<Optimizer> clone() const override;

        double get_momentum() const { return momentum_; }
        bool get_nesterov() const { return nesterov_; }
        
    private:
        double momentum_;
        bool nesterov_;
        Eigen::MatrixXd velocity_w_;
        Eigen::VectorXd velocity_b_;
        
        void initialize_if_needed(int rows, int cols);
        void initialize_if_needed(int size);
    };

} // namespace models

#endif