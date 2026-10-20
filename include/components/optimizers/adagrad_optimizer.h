#ifndef ADAGRAD_OPTIMIZER_H
#define ADAGRAD_OPTIMIZER_H

#include "optimizer.h"

namespace models {

    class AdagradOptimizer : public Optimizer {
    public:
        AdagradOptimizer(double learning_rate = 0.01,
                         double epsilon = 1e-8,
                         double decay = 0.0);
        ~AdagradOptimizer() override = default;

        void update_weights(Eigen::Ref<Eigen::MatrixXd> weights,
                            const Eigen::Ref<const Eigen::MatrixXd>& gradient) override;
        void update_bias(Eigen::Ref<Eigen::VectorXd> bias,
                         const Eigen::Ref<const Eigen::VectorXd>& gradient) override;

        void reset() override;

        OptimizerType get_type() const override { return OptimizerType::ADAGRAD; }
        std::string get_type_str() const override { return "adagrad"; }

        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;

        std::unique_ptr<Optimizer> clone() const override;

        double get_epsilon() const { return epsilon_; }

    private:
        double epsilon_;
        Eigen::MatrixXd accumulated_squared_gradients_w_;
        Eigen::VectorXd accumulated_squared_gradients_b_;

        void initialize_if_needed(int rows, int cols);
        void initialize_if_needed(int size);
    };

} // namespace models

#endif
