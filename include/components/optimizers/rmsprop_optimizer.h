#ifndef RMSPROP_OPTIMIZER_H
#define RMSPROP_OPTIMIZER_H

#include "optimizer.h"

namespace models {

    class RMSpropOptimizer : public Optimizer {
    public:
        RMSpropOptimizer(double learning_rate = 0.001,
                         double rho = 0.9,
                         double epsilon = 1e-8,
                         double decay = 0.0,
                         double momentum = 0.0);
        ~RMSpropOptimizer() override = default;

        void update_weights(Eigen::Ref<Eigen::MatrixXd> weights,
                            const Eigen::Ref<const Eigen::MatrixXd>& gradient) override;
        void update_bias(Eigen::Ref<Eigen::VectorXd> bias,
                         const Eigen::Ref<const Eigen::VectorXd>& gradient) override;

        void reset() override;

        OptimizerType get_type() const override { return OptimizerType::RMSPROP; }
        std::string get_type_str() const override { return "rmsprop"; }

        void serialize(std::ostream& out) const override;
        void deserialize(std::istream& in) override;

        std::unique_ptr<Optimizer> clone() const override;

        double get_rho() const { return rho_; }
        double get_epsilon() const { return epsilon_; }
        double get_momentum() const { return momentum_; }

    private:
        double rho_;
        double epsilon_;
        double momentum_;
        Eigen::MatrixXd mean_squared_gradients_w_;
        Eigen::VectorXd mean_squared_gradients_b_;
        Eigen::MatrixXd velocity_w_;
        Eigen::VectorXd velocity_b_;

        void initialize_if_needed(int rows, int cols);
        void initialize_if_needed(int size);
    };

} // namespace models

#endif
