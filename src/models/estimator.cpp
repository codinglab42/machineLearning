#include "models/estimator.h"
#include "components/optimizers/sgd.h"
#include "components/optimizers/adam.h"
#include "exceptions/exception_macros.h"
#include <iostream>

namespace models {

void Estimator::set_optimizer(OptimizerType type, double learning_rate) {
    std::cout << "🔍 Estimator::set_optimizer: type=" 
              << (type == OptimizerType::ADAM ? "ADAM" : "SGD")
              << ", lr=" << learning_rate << std::endl;
    
    ML_CHECK_PARAM(learning_rate > 0, "learning_rate", "must be positive", get_model_type());
    ML_CHECK_PARAM(learning_rate < 10.0, "learning_rate", "should be < 10.0", get_model_type());
    
    switch(type) {
        case OptimizerType::SGD:
            optimizer_ = std::make_unique<optimizers::SGD>(learning_rate);
            std::cout << "🔍 Creato SGD con successo" << std::endl;
            break;
        case OptimizerType::ADAM:
            optimizer_ = std::make_unique<optimizers::Adam>(learning_rate);
            std::cout << "🔍 Creato ADAM con successo" << std::endl;
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