// include/utils/scaler_factory.h
#pragma once
#include "scaler.h"
#include "standard_scaler.h"
#include "minmax_scaler.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace ml {

class ScalerFactory {
public:
    enum ScalerType {
        STANDARD,
        MINMAX
    };
    
    static std::unique_ptr<Scaler> create_scaler(ScalerType type) {
        switch (type) {
            case STANDARD:
                return std::make_unique<StandardScaler>();
            case MINMAX:
                return std::make_unique<MinMaxScaler>();
            default:
                throw MLException("Unknown scaler type");
        }
    }
    
    static std::unique_ptr<Scaler> create_scaler(const std::string& type_name) {
        static const std::unordered_map<std::string, ScalerType> type_map = {
            {"standard", STANDARD},
            {"minmax", MINMAX},
            {"min_max", MINMAX}
        };
        
        auto it = type_map.find(type_name);
        if (it != type_map.end()) {
            return create_scaler(it->second);
        }
        throw MLException("Unknown scaler type: " + type_name);
    }
};

} // namespace ml