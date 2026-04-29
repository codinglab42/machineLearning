// src/models/estimator.cpp
#include "exceptions/exception_macros.h"
#include <memory>
#include "models/estimator.h"
#include "components/optimizers/momentum_optimizer.h"  // AGGIUNGI QUESTO
#include "components/optimizers/sgd_optimizer.h"
#include "components/optimizers/adam_optimizer.h"
namespace models {

void Estimator::set_optimizer(OptimizerType type, double learning_rate) {
    ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be positive", get_model_type());
    ML_CHECK_PARAM(learning_rate < 10.0, "learning_rate", "should be < 10.0", get_model_type());
    
    switch(type) {
        case OptimizerType::SGD:
            optimizer_ = std::make_unique<SGDOptimizer>(learning_rate);
            break;
        case OptimizerType::MOMENTUM:
            optimizer_ = std::make_unique<MomentumOptimizer>(learning_rate);
            break;
        case OptimizerType::ADAM:
            optimizer_ = std::make_unique<AdamOptimizer>(learning_rate);
            break;
        case OptimizerType::RMSPROP:
        case OptimizerType::ADAGRAD:
            // TODO: implementare quando disponibili
            ML_THROW_NOT_IMPLEMENTED_ERROR("Optimizer type", get_model_type());
            break;
        default:
            ML_THROW_PARAMETER_ERROR("optimizer type", "unknown optimizer type", get_model_type());
            break;
    }
}

void Estimator::set_learning_rate(double lr) {
    ML_CHECK_PARAM(lr > 0, "learning_rate", "must be positive", get_model_type());
    if (optimizer_) {
        optimizer_->set_learning_rate(lr);
    }
}

double Estimator::get_learning_rate() const {
    return optimizer_ ? optimizer_->get_learning_rate() : 0.0;
}

} // namespace models