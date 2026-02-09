// include/utils/minmax_scaler.h
#pragma once
#include "exceptions/ml_exception.h"
#include "scaler.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace ml {

    using namespace ml_exception;

    class MinMaxScaler : public Scaler {
    private:
        std::vector<double> min_;
        std::vector<double> max_;
        double feature_range_min_;
        double feature_range_max_;
        bool fitted_ = false;
        
    public:
        MinMaxScaler(double range_min = 0.0, double range_max = 1.0) 
            : feature_range_min_(range_min), feature_range_max_(range_max) {
            if (range_min >= range_max) {
                throw MLException("feature_range_min must be less than feature_range_max");
            }
        }
        
        void fit(const std::vector<std::vector<double>>& X) override {
            if (X.empty() || X[0].empty()) {
                throw MLException("Cannot fit scaler on empty data");
            }
            
            size_t n_samples = X.size();
            size_t n_features = X[0].size();
            
            min_.resize(n_features, std::numeric_limits<double>::max());
            max_.resize(n_features, std::numeric_limits<double>::lowest());
            
            for (const auto& sample : X) {
                if (sample.size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in fit data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    if (sample[j] < min_[j]) min_[j] = sample[j];
                    if (sample[j] > max_[j]) max_[j] = sample[j];
                }
            }
            
            // Controlla che min != max per ogni feature
            for (size_t j = 0; j < n_features; ++j) {
                if (std::abs(max_[j] - min_[j]) < 1e-12) {
                    // Feature costante, imposta range fisso
                    max_[j] = min_[j] + 1.0;
                }
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
            
            if (n_features != min_.size()) {
                throw MLException("Feature dimension mismatch in transform");
            }
            
            std::vector<std::vector<double>> X_scaled(n_samples, 
                std::vector<double>(n_features));
            
            double range_scale = feature_range_max_ - feature_range_min_;
            
            for (size_t i = 0; i < n_samples; ++i) {
                if (X[i].size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in transform data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    double scale = max_[j] - min_[j];
                    X_scaled[i][j] = feature_range_min_ + 
                        ((X[i][j] - min_[j]) / scale) * range_scale;
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
            
            if (n_features != min_.size()) {
                throw MLException("Feature dimension mismatch in inverse_transform");
            }
            
            std::vector<std::vector<double>> X_original(n_samples, 
                std::vector<double>(n_features));
            
            double range_scale = feature_range_max_ - feature_range_min_;
            
            for (size_t i = 0; i < n_samples; ++i) {
                if (X_scaled[i].size() != n_features) {
                    throw MLException("Inconsistent feature dimensions in inverse_transform data");
                }
                for (size_t j = 0; j < n_features; ++j) {
                    double scale = max_[j] - min_[j];
                    X_original[i][j] = min_[j] + 
                        ((X_scaled[i][j] - feature_range_min_) / range_scale) * scale;
                }
            }
            
            return X_original;
        }
        
        std::string get_type() const override { return "MinMaxScaler"; }
        
        // Getter e setter
        const std::vector<double>& get_min() const { return min_; }
        const std::vector<double>& get_max() const { return max_; }
        
        void set_params(const std::vector<double>& min, 
                        const std::vector<double>& max) {
            if (min.size() != max.size()) {
                throw MLException("Min and max must have same dimensions");
            }
            min_ = min;
            max_ = max;
            fitted_ = true;
        }
    };

} // namespace ml