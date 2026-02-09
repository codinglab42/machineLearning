// include/utils/standard_scaler.h
#pragma once
#include "exceptions/ml_exception.h"
#include "scaler.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace ml {

    using namespace ml_exception;

    class StandardScaler : public Scaler {
    private:
        std::vector<double> mean_;
        std::vector<double> std_;
        bool fitted_ = false;
        double epsilon_ = 1e-8;
        
    public:
        StandardScaler() = default;
        explicit StandardScaler(double epsilon) : epsilon_(epsilon) {}
        
        void fit(const std::vector<std::vector<double>>& X) override {
            if (X.empty() || X[0].empty()) {
                throw MLException("Cannot fit scaler on empty data");
            }
            
            size_t n_samples = X.size();
            size_t n_features = X[0].size();
            
            mean_.resize(n_features, 0.0);
            std_.resize(n_features, 0.0);
            
            // Calcola la media per ogni feature
            for (const auto& sample : X) {
                if (sample.size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in fit data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    mean_[j] += sample[j];
                }
            }
            
            for (size_t j = 0; j < n_features; ++j) {
                mean_[j] /= n_samples;
            }
            
            // Calcola la deviazione standard
            for (const auto& sample : X) {
                for (size_t j = 0; j < n_features; ++j) {
                    double diff = sample[j] - mean_[j];
                    std_[j] += diff * diff;
                }
            }
            
            for (size_t j = 0; j < n_features; ++j) {
                std_[j] = std::sqrt(std_[j] / n_samples + epsilon_);
            }
            
            fitted_ = true;
        }
        
        std::vector<std::vector<double>> transform(
            const std::vector<std::vector<double>>& X) const override {
            
            if (!fitted_) {
                throw MLException("Scaler must be fitted before transform");
            }
            
            if (X.empty()) return {};
            
            size_t n_samples = X.size();
            size_t n_features = X[0].size();
            
            if (n_features != mean_.size()) {
                throw MLException("Feature dimension mismatch in transform");
            }
            
            std::vector<std::vector<double>> X_scaled(n_samples, 
                std::vector<double>(n_features));
            
            for (size_t i = 0; i < n_samples; ++i) {
                if (X[i].size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in transform data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    if (std_[j] < epsilon_) {
                        X_scaled[i][j] = 0.0;
                    } else {
                        X_scaled[i][j] = (X[i][j] - mean_[j]) / std_[j];
                    }
                }
            }
            
            return X_scaled;
        }
        
        std::vector<std::vector<double>> inverse_transform(
            const std::vector<std::vector<double>>& X_scaled) const override {
            
            if (!fitted_) {
                throw MLException("Scaler must be fitted before inverse_transform");
            }
            
            if (X_scaled.empty()) return {};
            
            size_t n_samples = X_scaled.size();
            size_t n_features = X_scaled[0].size();
            
            if (n_features != mean_.size()) {
                throw MLException("Feature dimension mismatch in inverse_transform");
            }
            
            std::vector<std::vector<double>> X_original(n_samples, 
                std::vector<double>(n_features));
            
            for (size_t i = 0; i < n_samples; ++i) {
                if (X_scaled[i].size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in inverse_transform data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    X_original[i][j] = X_scaled[i][j] * std_[j] + mean_[j];
                }
            }
            
            return X_original;
        }
        
        std::string get_type() const override { return "StandardScaler"; }
        
        // Getter per parametri
        const std::vector<double>& get_mean() const { return mean_; }
        const std::vector<double>& get_std() const { return std_; }
        
        // Setter (utile per caricare parametri pre-trained)
        void set_params(const std::vector<double>& mean, 
                        const std::vector<double>& std) {
            if (mean.size() != std.size()) {
                throw MLException("Mean and std must have same dimensions");
            }
            mean_ = mean;
            std_ = std;
            fitted_ = true;
        }
    };

} // namespace ml