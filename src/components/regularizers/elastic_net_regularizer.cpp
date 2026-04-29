#include <memory>
#include <Eigen/Dense>
#include "components/regularizers/elastic_net_regularizer.h"
#include "utils/serializable.h"
#include <cmath>

namespace models {

    ElasticNetRegularizer::ElasticNetRegularizer(double strength, double l1_ratio)
        : Regularizer(strength), l1_ratio_(l1_ratio) {
        // l1_ratio deve essere tra 0 e 1
        if (l1_ratio < 0.0 || l1_ratio > 1.0) {
            throw std::invalid_argument("l1_ratio must be between 0 and 1");
        }
    }

    double ElasticNetRegularizer::compute_loss(const Eigen::MatrixXd& weights) const {
        // Elastic Net = lambda * [l1_ratio * L1 + (1-l1_ratio) * 0.5 * L2]
        double l1_term = l1_ratio_ * weights.array().abs().sum();
        double l2_term = (1.0 - l1_ratio_) * 0.5 * weights.array().square().sum();
        return strength_ * (l1_term + l2_term);
    }

    double ElasticNetRegularizer::compute_loss(const Eigen::VectorXd& bias) const {
        double l1_term = l1_ratio_ * bias.array().abs().sum();
        double l2_term = (1.0 - l1_ratio_) * 0.5 * bias.array().square().sum();
        return strength_ * (l1_term + l2_term);
    }

    Eigen::MatrixXd ElasticNetRegularizer::compute_gradient(const Eigen::MatrixXd& weights) const {
        Eigen::MatrixXd grad(weights.rows(), weights.cols());
        
        for (int i = 0; i < weights.size(); ++i) {
            // Gradiente L1
            double l1_grad = (weights(i) > 0) ? 1.0 : (weights(i) < 0) ? -1.0 : 0.0;
            
            // Gradiente combinato
            grad(i) = strength_ * (l1_ratio_ * l1_grad + (1.0 - l1_ratio_) * weights(i));
        }
        
        return grad;
    }

    Eigen::VectorXd ElasticNetRegularizer::compute_gradient(const Eigen::VectorXd& bias) const {
        Eigen::VectorXd grad(bias.size());
        
        for (int i = 0; i < bias.size(); ++i) {
            double l1_grad = (bias(i) > 0) ? 1.0 : (bias(i) < 0) ? -1.0 : 0.0;
            grad(i) = strength_ * (l1_ratio_ * l1_grad + (1.0 - l1_ratio_) * bias(i));
        }
        
        return grad;
    }

    void ElasticNetRegularizer::serialize(std::ostream& out) const {
        Regularizer::serialize(out);
        using namespace utils;
        out.write(reinterpret_cast<const char*>(&l1_ratio_), sizeof(double));
    }

    void ElasticNetRegularizer::deserialize(std::istream& in) {
        Regularizer::deserialize(in);
        using namespace utils;
        in.read(reinterpret_cast<char*>(&l1_ratio_), sizeof(double));
    }

    std::unique_ptr<Regularizer> ElasticNetRegularizer::clone() const {
        return std::make_unique<ElasticNetRegularizer>(strength_, l1_ratio_);
    }

} // namespace models

