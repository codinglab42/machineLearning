// include/utils/scaler.h
#pragma once
#include <vector>
#include <memory>

namespace ml {
    
class Scaler {
public:
    virtual ~Scaler() = default;
    
    // Fit su dati di training
    virtual void fit(const std::vector<std::vector<double>>& X) = 0;
    
    // Trasforma i dati
    virtual std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& X) const = 0;
    
    // Fit + Transform in una volta
    std::vector<std::vector<double>> fit_transform(
        std::vector<std::vector<double>>& X) {
        fit(X);
        return transform(X);
    }
    
    // Inversa (da scaled a originali)
    virtual std::vector<std::vector<double>> inverse_transform(
        const std::vector<std::vector<double>>& X_scaled) const = 0;
    
    virtual std::string get_type() const = 0;
};

} // namespace ml