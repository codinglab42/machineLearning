#include "exceptions/exception_macros.h"
#include <Eigen/Dense>
#include "models/logistic_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "utils/serializable.h"

using namespace Eigen;
using namespace models;
using namespace utils;

// Costruttori
LogisticRegression::LogisticRegression()
    : learning_rate_(0.1), max_iter_(1000), lambda_(0.0), tolerance_(1e-4),
      verbose_(false), n_features_(0), n_iter_(0) {}

LogisticRegression::LogisticRegression(double learning_rate, int max_iter, 
                                     double lambda, double tolerance, bool verbose)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda), 
      tolerance_(tolerance), verbose_(verbose), n_features_(0), n_iter_(0) {
    
    ML_CHECK_PARAM(learning_rate_ > 0, "learning_rate", "must be > 0", get_model_type());
    ML_CHECK_PARAM(max_iter_ > 0, "max_iter", "must be > 0", get_model_type());
    ML_CHECK_PARAM(lambda_ >= 0, "lambda", "must be >= 0", get_model_type());
    ML_CHECK_PARAM(tolerance_ > 0, "tolerance", "must be > 0", get_model_type());
}

// Setters con validazione
void LogisticRegression::set_max_iterations(int max_iter) {
    ML_CHECK_PARAM(max_iter > 0, "max_iter", "must be > 0", get_model_type());
    max_iter_ = max_iter;
}

void LogisticRegression::set_lambda(double lambda) {
    ML_CHECK_PARAM(lambda >= 0, "lambda", "must be >= 0", get_model_type());
    lambda_ = lambda;
}

void LogisticRegression::set_tolerance(double tolerance) {
    ML_CHECK_PARAM(tolerance > 0, "tolerance", "must be > 0", get_model_type());
    tolerance_ = tolerance;
}

void LogisticRegression::set_verbose(bool verbose) {
    verbose_ = verbose;
}

// Metodo fit principale
void LogisticRegression::fit(const MatrixXd& X, const Eigen::VectorXd& y) {
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_NOT_EMPTY(y, "y", get_model_type());
    ML_CHECK_XY_SIZE(X.rows(), y.size(), get_model_type());
    
    // Verifica che y sia binario (0 o 1)
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    if (y_min < 0 || y_max > 1) {
        throw ml_exception::InvalidParameterException(
            "y", "must contain only 0 and 1 values", get_model_type());
    }
    
    // Salva il numero di feature ORIGINALI (senza bias)
    Eigen::Index cols = X.cols();
    if (cols > std::numeric_limits<int>::max()) {
        throw ml_exception::DimensionMismatchException(
            "Number of features", 
            std::numeric_limits<int>::max(), 1,
            static_cast<int>(cols), 1,
            get_model_type()
        );
    }
    n_features_ = static_cast<int>(cols);
    
    // Aggiunge intercetta UNA SOLA VOLTA per il training
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    
    gradient_descent(X_with_bias, y);
}

void LogisticRegression::gradient_descent(const MatrixXd& X_with_bias, const VectorXd& y) {
    double m = static_cast<double>(X_with_bias.rows());
    theta_ = VectorXd::Zero(X_with_bias.cols());  // Dimensione = n_features_ + 1
    cost_history_.clear();
    accuracy_history_.clear();
    cost_history_.reserve(max_iter_);
    accuracy_history_.reserve(max_iter_);

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        // Forward pass
        VectorXd z = X_with_bias * theta_;
        VectorXd h = MathUtils::sigmoid_vec(z);
        
        // Calcolo gradienti
        VectorXd error = h - y;
        VectorXd gradient = (X_with_bias.transpose() * error) / m;
        
        // Regularizzazione L2 (non regolarizzare l'intercetta)
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / m) * theta_;
            reg(0) = 0; // Non regolarizzare l'intercetta
            gradient += reg;
        }
        
        // Aggiornamento parametri
        theta_ -= learning_rate_ * gradient;
        
        // Calcolo metriche (passa X_with_bias che ha già il bias)
        cost_history_.push_back(compute_cost(X_with_bias, y));
        accuracy_history_.push_back(compute_accuracy(X_with_bias, y));
        
        // Early stopping
        if (n_iter_ > 10 && cost_history_.size() > 10) {
            double improvement = cost_history_[cost_history_.size() - 11] 
                               - cost_history_.back();
            if (std::abs(improvement) < tolerance_) {
                if (verbose_) {
                    std::cout << "Early stopping at iteration " << n_iter_ 
                              << ", cost: " << cost_history_.back() 
                              << ", accuracy: " << accuracy_history_.back() 
                              << std::endl;
                }
                break;
            }
        }
        
        // Log progresso
        if (verbose_ && n_iter_ % 100 == 0) {
            std::cout << "Iteration " << n_iter_ 
                      << ", Cost: " << cost_history_.back()
                      << ", Accuracy: " << accuracy_history_.back() 
                      << std::endl;
        }
    }
    
    if (n_iter_ >= max_iter_ && verbose_) {
        std::cout << "Reached maximum iterations: " << max_iter_ 
                  << ", Final cost: " << cost_history_.back()
                  << ", Final accuracy: " << accuracy_history_.back()
                  << std::endl;
    }
}

double LogisticRegression::compute_cost(const MatrixXd& X_with_bias, const VectorXd& y) const {
    double m = static_cast<double>(X_with_bias.rows());
    
    VectorXd z = X_with_bias * theta_;
    VectorXd h = MathUtils::sigmoid_vec(z);
    
    // Log loss con stabilità numerica
    VectorXd log_h = MathUtils::safe_log(h);
    VectorXd log_1_minus_h = MathUtils::safe_log(VectorXd::Ones(h.size()) - h);
    
    double J = -(y.dot(log_h) + (VectorXd::Ones(y.size()) - y).dot(log_1_minus_h)) / m;
    
    // Regularizzazione L2 (escludi intercetta)
    if (lambda_ > 0 && theta_.size() > 1) {
        J += (lambda_ / (2.0 * m)) * theta_.tail(theta_.size() - 1).squaredNorm();
    }
    
    return J;
}

double LogisticRegression::compute_accuracy(const MatrixXd& X_with_bias, const VectorXd& y) const {
    VectorXi y_pred_class = predict_class_from_features(X_with_bias, 0.5);
    int correct = 0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (y_pred_class(i) == static_cast<int>(y(i))) {
            correct++;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(y.size());
}

// Metodo interno per predire su X che ha GIA' il bias
VectorXd LogisticRegression::predict_from_features(const MatrixXd& X_with_bias) const {
    ML_CHECK_FITTED(theta_.size() > 0, get_model_type());
    ML_CHECK_NOT_EMPTY(X_with_bias, "X", get_model_type());
    ML_CHECK_FEATURE_DIMENSIONS(X_with_bias.cols(), theta_.size(), get_model_type());
    
    VectorXd z = X_with_bias * theta_;
    return MathUtils::sigmoid_vec(z);
}

// Metodo interno per classificare su X che ha GIA' il bias
VectorXi LogisticRegression::predict_class_from_features(const MatrixXd& X_with_bias, double threshold) const {
    ML_CHECK_PARAM(threshold >= 0 && threshold <= 1, "threshold", 
                  "must be between 0 and 1", get_model_type());
    
    VectorXd probabilities = predict_from_features(X_with_bias);
    VectorXi classes(probabilities.size());
    
    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        classes(i) = (probabilities(i) >= threshold) ? 1 : 0;
    }
    
    return classes;
}

// Metodi predict pubblici - accettano X SENZA bias
VectorXd LogisticRegression::predict(const MatrixXd& X) const {
    ML_CHECK_FITTED(theta_.size() > 0, get_model_type());
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, get_model_type());
    
    // Aggiunge bias automaticamente per l'utente
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    return predict_from_features(X_with_bias);
}

VectorXi LogisticRegression::predict_class(const MatrixXd& X, double threshold) const {
    ML_CHECK_PARAM(threshold >= 0 && threshold <= 1, "threshold", 
                  "must be between 0 and 1", get_model_type());
    
    // Aggiunge bias automaticamente per l'utente
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    return predict_class_from_features(X_with_bias, threshold);
}

// Metodi score
double LogisticRegression::score(const MatrixXd& X, const VectorXd& y) const {
    ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, get_model_type());
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    return compute_accuracy(X_with_bias, y);
}

Vector3d LogisticRegression::precision_recall_f1(const MatrixXd& X, 
                                               const VectorXd& y, 
                                               double threshold) const {
    ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, get_model_type());
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    Eigen::MatrixXd cm = confusion_matrix_from_features(X_with_bias, y, threshold);
    
    double tp = cm(1, 1);
    double fp = cm(0, 1);
    double fn = cm(1, 0);
    
    double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    
    return Vector3d(precision, recall, f1);
}

MatrixXd LogisticRegression::confusion_matrix(const MatrixXd& X, 
                                            const VectorXd& y, 
                                            double threshold) const {
    ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, get_model_type());
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    return confusion_matrix_from_features(X_with_bias, y, threshold);
}

MatrixXd LogisticRegression::confusion_matrix_from_features(const MatrixXd& X_with_bias, 
                                                          const VectorXd& y, 
                                                          double threshold) const {
    VectorXi y_pred = predict_class_from_features(X_with_bias, threshold);
    MatrixXd cm = MatrixXd::Zero(2, 2);
    
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        int actual = static_cast<int>(y(i));
        int predicted = y_pred(i);
        if (actual >= 0 && actual < 2 && predicted >= 0 && predicted < 2) {
            cm(actual, predicted) += 1.0;
        }
    }
    
    return cm;
}

// Serializzazione
void LogisticRegression::serialize_binary(std::ostream& out) const {
    using namespace utils;
    
    // Serializza parametri
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&max_iter_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&lambda_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&tolerance_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&verbose_), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_iter_), sizeof(int));
    
    // Serializza theta
    utils::write_eigen_vector(out, theta_);
    
    // Serializza history
    size_t cost_size = cost_history_.size();
    out.write(reinterpret_cast<const char*>(&cost_size), sizeof(size_t));
    if (cost_size > 0) {
        out.write(reinterpret_cast<const char*>(cost_history_.data()), 
                 cost_size * sizeof(double));
    }
    
    size_t acc_size = accuracy_history_.size();
    out.write(reinterpret_cast<const char*>(&acc_size), sizeof(size_t));
    if (acc_size > 0) {
        out.write(reinterpret_cast<const char*>(accuracy_history_.data()), 
                 acc_size * sizeof(double));
    }
}

void LogisticRegression::deserialize_binary(std::istream& in) {
    using namespace utils;
    
    // Deserializza parametri
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&max_iter_), sizeof(int));
    in.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
    in.read(reinterpret_cast<char*>(&tolerance_), sizeof(double));
    in.read(reinterpret_cast<char*>(&verbose_), sizeof(bool));
    in.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_iter_), sizeof(int));
    
    // Deserializza theta
    utils::read_eigen_vector(in, theta_);
    
    // Deserializza history
    size_t cost_size;
    in.read(reinterpret_cast<char*>(&cost_size), sizeof(size_t));
    cost_history_.resize(cost_size);
    if (cost_size > 0) {
        in.read(reinterpret_cast<char*>(cost_history_.data()), 
               cost_size * sizeof(double));
    }
    
    size_t acc_size;
    in.read(reinterpret_cast<char*>(&acc_size), sizeof(size_t));
    accuracy_history_.resize(acc_size);
    if (acc_size > 0) {
        in.read(reinterpret_cast<char*>(accuracy_history_.data()), 
               acc_size * sizeof(double));
    }
}

std::string LogisticRegression::to_string() const {
    std::ostringstream oss;
    oss << "LogisticRegression [" 
        << "Features: " << n_features_
        << ", Iterations: " << n_iter_
        << ", λ: " << std::fixed << std::setprecision(4) << lambda_
        << ", LR: " << learning_rate_
        << ", Tolerance: " << tolerance_
        << "]";
    return oss.str();
}