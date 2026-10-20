/**
 * @file example_energy_forecast.cpp
 * @brief Forecast next-day electricity demand using linear regression.
 *
 * The program generates a reproducible synthetic series, trains on its first
 * 80%, and compares future-period forecasts with a persistence baseline.
 */

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "models/linear_regression.h"

namespace {

constexpr int kDays = 4 * 365;
constexpr int kHistoryDays = 7;
constexpr double kPi = 3.14159265358979323846;

struct ForecastData {
    Eigen::MatrixXd features;
    Eigen::VectorXd targets;
    Eigen::VectorXd persistence;
};

ForecastData generate_data() {
    std::mt19937 generator(42);
    std::normal_distribution<double> temperature_noise(0.0, 3.0);
    std::normal_distribution<double> forecast_noise(0.0, 1.5);
    std::normal_distribution<double> demand_noise(0.0, 5.0);

    std::vector<double> temperature(kDays);
    std::vector<double> forecast_temperature(kDays);
    std::vector<double> demand(kDays);
    demand[0] = 220.0;

    for (int day = 0; day < kDays; ++day) {
        const double annual_phase = 2.0 * kPi * day / 365.0;
        temperature[day] =
            12.0 + 11.0 * std::sin(annual_phase - 0.8) + temperature_noise(generator);
        forecast_temperature[day] = temperature[day] + forecast_noise(generator);

        if (day == 0) {
            continue;
        }

        const int weekday = day % 7;
        const double weekend_effect = weekday >= 5 ? -18.0 : 0.0;
        const double heating = 2.2 * std::max(0.0, 16.0 - temperature[day]);
        const double cooling = 1.7 * std::max(0.0, temperature[day] - 23.0);
        const double baseline = 220.0 + weekend_effect + heating + cooling;
        const double seasonal =
            12.0 * std::sin(annual_phase + 0.3) +
            5.0 * std::sin(2.0 * kPi * day / 7.0 - 0.5);

        demand[day] = 0.62 * demand[day - 1] + 0.38 * baseline +
                      seasonal + demand_noise(generator);
    }

    const int samples = kDays - kHistoryDays;
    ForecastData data{
        Eigen::MatrixXd(samples, 8),
        Eigen::VectorXd(samples),
        Eigen::VectorXd(samples),
    };

    for (int sample = 0; sample < samples; ++sample) {
        const int day = sample + kHistoryDays;
        double weekly_average = 0.0;
        for (int lag = 1; lag <= kHistoryDays; ++lag) {
            weekly_average += demand[day - lag];
        }
        weekly_average /= kHistoryDays;

        data.features(sample, 0) = demand[day - 1];
        data.features(sample, 1) = demand[day - kHistoryDays];
        data.features(sample, 2) = weekly_average;
        data.features(sample, 3) = forecast_temperature[day];
        data.features(sample, 4) = std::sin(2.0 * kPi * day / 7.0);
        data.features(sample, 5) = std::cos(2.0 * kPi * day / 7.0);
        data.features(sample, 6) = std::sin(2.0 * kPi * day / 365.0);
        data.features(sample, 7) = std::cos(2.0 * kPi * day / 365.0);
        data.targets(sample) = demand[day];
        data.persistence(sample) = demand[day - 1];
    }

    return data;
}

void standardize_features(Eigen::MatrixXd& train, Eigen::MatrixXd& test) {
    const Eigen::RowVectorXd means = train.colwise().mean();
    Eigen::RowVectorXd scales(train.cols());
    for (Eigen::Index column = 0; column < train.cols(); ++column) {
        const double variance =
            (train.col(column).array() - means(column)).square().mean();
        scales(column) = std::sqrt(variance);
        if (scales(column) == 0.0) {
            scales(column) = 1.0;
        }
    }

    train.rowwise() -= means;
    test.rowwise() -= means;
    train.array().rowwise() /= scales.array();
    test.array().rowwise() /= scales.array();
}

double mean_absolute_error(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& predicted) {
    return (actual - predicted).cwiseAbs().mean();
}

double root_mean_squared_error(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& predicted) {
    return std::sqrt((actual - predicted).squaredNorm() / actual.size());
}

}  // namespace

int main() {
    ForecastData data = generate_data();
    const int split = static_cast<int>(data.targets.size() * 0.8);
    const int test_size = static_cast<int>(data.targets.size()) - split;

    Eigen::MatrixXd train_features = data.features.topRows(split);
    Eigen::MatrixXd test_features = data.features.bottomRows(test_size);
    const Eigen::VectorXd train_targets = data.targets.head(split);
    const Eigen::VectorXd test_targets = data.targets.tail(test_size);
    const Eigen::VectorXd baseline = data.persistence.tail(test_size);

    standardize_features(train_features, test_features);

    models::LinearRegression model(
        0.01,
        2000,
        0.01,
        models::LinearRegression::GRADIENT_DESCENT);
    model.fit(train_features, train_targets);

    const Eigen::VectorXd predictions = model.predict(test_features);
    const double model_mae = mean_absolute_error(test_targets, predictions);
    const double model_rmse = root_mean_squared_error(test_targets, predictions);
    const double baseline_mae = mean_absolute_error(test_targets, baseline);
    const double baseline_rmse = root_mean_squared_error(test_targets, baseline);
    const double r_squared = model.score(test_features, test_targets);

    std::cout << "Synthetic next-day electricity demand forecast\n"
              << "Observations: " << kDays << " daily values\n"
              << "Chronological split: " << split << " training samples, "
              << test_size << " test samples\n"
              << "Features: recent demand, 7-day average, calendar cycles, "
                 "forecast temperature\n"
              << std::fixed << std::setprecision(2)
              << "Test mean demand: " << test_targets.mean()
              << " synthetic units/day\n"
              << "C++ LinearRegression: MAE=" << model_mae
              << ", RMSE=" << model_rmse
              << ", R2=" << std::setprecision(3) << r_squared << '\n'
              << std::setprecision(2)
              << "Persistence baseline (tomorrow = today): MAE=" << baseline_mae
              << ", RMSE=" << baseline_rmse << "\n\n"
              << "Sample forecasts (actual vs predicted, test period):\n";

    for (int index = 0; index < std::min(7, test_size); ++index) {
        std::cout << "  actual " << std::setw(7) << test_targets(index)
                  << " | forecast " << std::setw(7) << predictions(index) << '\n';
    }

    std::cout << "\nSynthetic data only; these scores do not estimate "
                 "real-world performance.\n";
    return 0;
}
