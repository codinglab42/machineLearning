#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <Eigen/Dense>
#include <string>
#include "components/optimizers/optimizer.h"
#include "components/optimizers/sgd.h"
#include "components/optimizers/adam.h"
#include "utils/serializable.h"

namespace models {

    
    enum class OptimizerType {

        SGD,
        ADAM
    };


    class Estimator : public utils::SerializableModel {
    public:
        virtual ~Estimator() = default;

        // Metodi virtuali puri che ogni modello deve implementare
        virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
        virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;
        virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const = 0;

        // Metodi comuni per optimizer
        void Estimator::set_learning_rate(double lr) {
            // Il learning rate deve essere positivo e ragionevole
            ML_CHECK_PARAM(lr > 0, "learning_rate", "must be positive", get_model_type());
            ML_CHECK_PARAM(lr < 10.0, "learning_rate", "should be < 10.0 (suggested: 0.001-1.0)", get_model_type());
            
            if (optimizer_) {
                optimizer_->set_learning_rate(lr);
            }
        }

        
        double get_learning_rate() const {
            return optimizer_ ? optimizer_->get_learning_rate() : 0.0;
        }
        
        void Estimator::set_optimizer(OptimizerType type, double learning_rate) {
            // Controlli sul learning rate
            ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be positive", get_model_type());
            ML_CHECK_PARAM(learning_rate < 10.0, "learning_rate", "should be < 10.0", get_model_type());
            
            // Controllo sul tipo optimizer (già garantito dall'enum, ma per sicurezza)
            ML_CHECK_PARAM(type == OptimizerType::SGD || type == OptimizerType::ADAM,
                        "optimizer_type", "must be SGD or ADAM", get_model_type());
            
            switch(type) {
                case OptimizerType::SGD:
                    optimizer_ = std::make_unique<optimizers::SGD>(learning_rate);
                    break;
                case OptimizerType::ADAM:
                    optimizer_ = std::make_unique<optimizers::Adam>(learning_rate);
                    break;
            }
        }
        
        // Metodi ereditati da SerializableModel
        std::string to_string() const override = 0;
        void serialize_binary(std::ostream& out) const override = 0;
        void deserialize_binary(std::istream& in) override = 0;
        std::string get_model_type() const override = 0;
        
        // Manteniamo save/load per compatibilità (già implementati in SerializableModel)
        using utils::SerializableModel::save;
        using utils::SerializableModel::load;

    protected:
        std::unique_ptr<optimizers::Optimizer> optimizer_;
    };

} // namespace regression

#endif