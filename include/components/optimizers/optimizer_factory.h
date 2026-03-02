#ifndef OPTIMIZER_FACTORY_H
#define OPTIMIZER_FACTORY_H

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include "components/optimizers/optimizer.h"
#include "components/optimizers/sgd_optimizer.h"
#include "components/optimizers/momentum_optimizer.h"
#include "components/optimizers/adam_optimizer.h"
//#include "components/optimizers/rmsprop_optimizer.h"
//#include "components/optimizers/adagrad_optimizer.h"
#include "exceptions/exception_macros.h"

namespace models {

    class OptimizerFactory {
    public:
        // Crea un ottimizzatore basato sul tipo
        static std::unique_ptr<Optimizer> create(OptimizerType type, 
                                                 double learning_rate = 0.01,
                                                 const std::unordered_map<std::string, double>& params = {}) {
            switch (type) {
                case OptimizerType::SGD:
                    return create_sgd(learning_rate, params);
                case OptimizerType::MOMENTUM:
                    return create_momentum(learning_rate, params);
                case OptimizerType::ADAM:
                    return create_adam(learning_rate, params);
                /*
                case OptimizerType::RMSPROP:
                    return create_rmsprop(learning_rate, params);
                case OptimizerType::ADAGRAD:
                    return create_adagrad(learning_rate, params);
                */
                default:
                    ML_THROW_PARAMETER_ERROR("optimizer type", "unknown optimizer type", "OptimizerFactory");
                    return nullptr;
            }
        }
        
        // Crea ottimizzatore da stringa (per convenienza)
        static std::unique_ptr<Optimizer> create(const std::string& type_str,
                                                 double learning_rate = 0.01,
                                                 const std::unordered_map<std::string, double>& params = {}) {
            OptimizerType type = string_to_type(type_str);
            return create(type, learning_rate, params);
        }
        
        // Converte stringa in enum
        static OptimizerType string_to_type(const std::string& type_str) {
            static const std::unordered_map<std::string, OptimizerType> type_map = {
                {"sgd", OptimizerType::SGD},
                {"momentum", OptimizerType::MOMENTUM},
                {"adam", OptimizerType::ADAM},
                {"rmsprop", OptimizerType::RMSPROP},
                {"adagrad", OptimizerType::ADAGRAD}
            };
            
            auto it = type_map.find(type_str);
            if (it == type_map.end()) {
                ML_THROW_PARAMETER_ERROR("optimizer type", 
                                        "must be one of: sgd, momentum, adam, rmsprop, adagrad", 
                                        "OptimizerFactory");
            }
            return it->second;
        }
        
        // Converte enum in stringa
        static std::string type_to_string(OptimizerType type) {
            switch (type) {
                case OptimizerType::SGD: return "sgd";
                case OptimizerType::MOMENTUM: return "momentum";
                case OptimizerType::ADAM: return "adam";
                case OptimizerType::RMSPROP: return "rmsprop";
                case OptimizerType::ADAGRAD: return "adagrad";
                default: return "unknown";
            }
        }
        
    private:
        // Factory methods per ogni tipo
        static std::unique_ptr<Optimizer> create_sgd(double lr, 
                                                     const std::unordered_map<std::string, double>& params) {
            double momentum = params.count("momentum") ? params.at("momentum") : 0.0;
            double decay = params.count("decay") ? params.at("decay") : 0.0;
            bool nesterov = params.count("nesterov") ? static_cast<bool>(params.at("nesterov")) : false;
            
            if (momentum > 0.0) {
                return std::make_unique<MomentumOptimizer>(lr, momentum, decay, nesterov);
            } else {
                return std::make_unique<SGDOptimizer>(lr, decay);
            }
        }
        
        static std::unique_ptr<Optimizer> create_momentum(double lr, 
                                                          const std::unordered_map<std::string, double>& params) {
            double momentum = params.count("momentum") ? params.at("momentum") : 0.9;
            double decay = params.count("decay") ? params.at("decay") : 0.0;
            bool nesterov = params.count("nesterov") ? static_cast<bool>(params.at("nesterov")) : false;
            
            return std::make_unique<MomentumOptimizer>(lr, momentum, decay, nesterov);
        }
        
        static std::unique_ptr<Optimizer> create_adam(double lr, 
                                                      const std::unordered_map<std::string, double>& params) {
            double beta1 = params.count("beta1") ? params.at("beta1") : 0.9;
            double beta2 = params.count("beta2") ? params.at("beta2") : 0.999;
            double epsilon = params.count("epsilon") ? params.at("epsilon") : 1e-8;
            double decay = params.count("decay") ? params.at("decay") : 0.0;
            
            return std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon, decay);
        }
        
        /*
        static std::unique_ptr<Optimizer> create_rmsprop(double lr, 
                                                         const std::unordered_map<std::string, double>& params) {
            double rho = params.count("rho") ? params.at("rho") : 0.9;
            double epsilon = params.count("epsilon") ? params.at("epsilon") : 1e-8;
            double decay = params.count("decay") ? params.at("decay") : 0.0;
            double momentum = params.count("momentum") ? params.at("momentum") : 0.0;
            
            return std::make_unique<RMSpropOptimizer>(lr, rho, epsilon, decay, momentum);
        }
        
        static std::unique_ptr<Optimizer> create_adagrad(double lr, 
                                                         const std::unordered_map<std::string, double>& params) {
            double epsilon = params.count("epsilon") ? params.at("epsilon") : 1e-8;
            double decay = params.count("decay") ? params.at("decay") : 0.0;
            
            return std::make_unique<AdagradOptimizer>(lr, epsilon, decay);
        }
            */
    };

} // namespace models

#endif
