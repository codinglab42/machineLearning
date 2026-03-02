#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "optimizer.h"

namespace models {

    class SGDOptimizer : public Optimizer {
    public:
        SGDOptimizer(double learning_rate = 0.01, double decay = 0.0);
        ~SGDOptimizer() override = default;
        
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) override;
        void update(Eigen::VectorXd& bias, const Eigen::VectorXd& gradient) override;
        
        void reset() override;
        
        OptimizerType get_type() const override { return OptimizerType::SGD; }
        std::string get_type_str() const override { return "sgd"; }
        
        std::unique_ptr<Optimizer> clone() const override;
    };

} // namespace models

#endif

