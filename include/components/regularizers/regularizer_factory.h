#ifndef REGULARIZER_FACTORY_H
#define REGULARIZER_FACTORY_H

#include <memory>
#include <string>
#include <unordered_map>
#include "regularizer.h"
#include "l1_regularizer.h"
#include "l2_regularizer.h"
#include "elastic_net_regularizer.h"
#include "exceptions/exception_macros.h"

namespace models {

    class RegularizerFactory {
    public:
        // Crea un regolarizzatore basato sul tipo
        static std::unique_ptr<Regularizer> create(RegularizerType type, 
                                                   double strength = 0.01,
                                                   const std::unordered_map<std::string, double>& params = {}) {
            switch (type) {
                case RegularizerType::NONE:
                    return nullptr;
                case RegularizerType::L1:
                    return std::make_unique<L1Regularizer>(strength);
                case RegularizerType::L2:
                    return std::make_unique<L2Regularizer>(strength);
                case RegularizerType::ELASTIC_NET: {
                    double l1_ratio = params.count("l1_ratio") ? params.at("l1_ratio") : 0.5;
                    return std::make_unique<ElasticNetRegularizer>(strength, l1_ratio);
                }
                default:
                    ML_THROW_PARAMETER_ERROR("regularizer type", "unknown regularizer type", "RegularizerFactory");
                    return nullptr;
            }
        }
        
        // Crea regolarizzatore da stringa
        static std::unique_ptr<Regularizer> create(const std::string& type_str,
                                                   double strength = 0.01,
                                                   const std::unordered_map<std::string, double>& params = {}) {
            RegularizerType type = string_to_type(type_str);
            return create(type, strength, params);
        }
        
        // Converte stringa in enum
        static RegularizerType string_to_type(const std::string& type_str) {
            static const std::unordered_map<std::string, RegularizerType> type_map = {
                {"none", RegularizerType::NONE},
                {"l1", RegularizerType::L1},
                {"l2", RegularizerType::L2},
                {"elastic_net", RegularizerType::ELASTIC_NET}
            };
            
            auto it = type_map.find(type_str);
            if (it == type_map.end()) {
                ML_THROW_PARAMETER_ERROR("regularizer type", 
                                        "must be one of: none, l1, l2, elastic_net", 
                                        "RegularizerFactory");
            }
            return it->second;
        }
        
        // Converte enum in stringa
        static std::string type_to_string(RegularizerType type) {
            switch (type) {
                case RegularizerType::NONE: return "none";
                case RegularizerType::L1: return "l1";
                case RegularizerType::L2: return "l2";
                case RegularizerType::ELASTIC_NET: return "elastic_net";
                default: return "unknown";
            }
        }
    };

} // namespace models

#endif

