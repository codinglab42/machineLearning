#include "models/linear_regression.h"
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
LinearRegression::LinearRegression()
    : learning_rate_(0.01), max_iter_(1000), lambda_(0.0), tolerance_(1e-4),
      verbose_(false), solver_(GRADIENT_DESCENT), n_features_(0), n_iter_(0) {}

LinearRegression::LinearRegression(double learning_rate, int max_iter, 
                                 double lambda, Solver solver)
    : learning_rate_(learning_rate), max_iter_(max_iter), lambda_(lambda), 
      tolerance_(1e-4), verbose_(false), solver_(solver), 
      n_features_(0), n_iter_(0) {
    
    ML_CHECK_PARAM(learning_rate_ > 0, "learning_rate", "must be > 0", get_model_type());
    ML_CHECK_PARAM(max_iter_ > 0, "max_iter", "must be > 0", get_model_type());
    ML_CHECK_PARAM(lambda_ >= 0, "lambda", "must be >= 0", get_model_type());
}

// Setters con validazione
void LinearRegression::set_learning_rate(double rate) {
    ML_CHECK_PARAM(rate > 0, "learning_rate", "must be > 0", get_model_type());
    learning_rate_ = rate;
}

void LinearRegression::set_max_iterations(int max_iter) {
    ML_CHECK_PARAM(max_iter > 0, "max_iter", "must be > 0", get_model_type());
    max_iter_ = max_iter;
}

void LinearRegression::set_lambda(double lambda) {
    ML_CHECK_PARAM(lambda >= 0, "lambda", "must be >= 0", get_model_type());
    lambda_ = lambda;
}

// Metodo fit principale - CORRETTO
void LinearRegression::fit(const MatrixXd& X, const Eigen::VectorXd& y) {
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_NOT_EMPTY(y, "y", get_model_type());
    ML_CHECK_XY_SIZE(X.rows(), y.size(), get_model_type());
    
    // Salva numero feature ORIGINALI (senza bias)
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
    
    // Aggiungi bias UNA SOLA VOLTA
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    
    // Chiama il solver appropriato UNA SOLA VOLTA
    if (solver_ == GRADIENT_DESCENT) {
        gradient_descent(X_with_bias, y);
    } else if (solver_ == NORMAL_EQUATION) {
        normal_equation(X_with_bias, y);
    } else {
        svd_solve(X_with_bias, y);
    }
    
    // Debug (opzionale)
    if (verbose_) {
        std::cout << "[DEBUG] Model fitted. Features: " << n_features_ 
                  << ", Theta size: " << theta_.size() 
                  << " (includes bias)" << std::endl;
    }
}

void LinearRegression::normal_equation(const MatrixXd& X_with_bias, const VectorXd& y) {
    // X_with_bias ha già la colonna di 1
    MatrixXd XtX = X_with_bias.transpose() * X_with_bias;
    VectorXd Xty = X_with_bias.transpose() * y;

    if (lambda_ > 0) {
        MatrixXd I = MatrixXd::Identity(XtX.rows(), XtX.cols());
        I(0, 0) = 0; // Non regolarizzare intercetta
        XtX += lambda_ * I;
    }
    
    theta_ = XtX.ldlt().solve(Xty);
    
    if (verbose_) {
        std::cout << "Normal equation solved. Theta size: " << theta_.size() << std::endl;
    }
}

void LinearRegression::svd_solve(const MatrixXd& X_with_bias, const VectorXd& y) {
    // X_with_bias ha già la colonna di 1
    theta_ = X_with_bias.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
    
    if (verbose_) {
        std::cout << "SVD solved. Theta size: " << theta_.size() << std::endl;
    }
}

void LinearRegression::gradient_descent(const MatrixXd& X_with_bias, const VectorXd& y) {
    double m = static_cast<double>(X_with_bias.rows());
    theta_ = VectorXd::Zero(X_with_bias.cols());  // Dimensione = n_features_ + 1
    
    if (verbose_) {
        std::cout << "[DEBUG] gradient_descent: theta_ size = " << theta_.size() 
                  << " (includes bias)" << std::endl;
    }

    cost_history_.clear();
    cost_history_.reserve(max_iter_);

    for (n_iter_ = 0; n_iter_ < max_iter_; ++n_iter_) {
        VectorXd error = (X_with_bias * theta_) - y;
        VectorXd gradient = (X_with_bias.transpose() * error) / m;
        
        if (lambda_ > 0) {
            VectorXd reg = (lambda_ / m) * theta_;
            reg(0) = 0; // Non regolarizzare intercetta
            gradient += reg;
        }

        theta_ -= learning_rate_ * gradient;
        cost_history_.push_back(compute_cost(X_with_bias, y));
        
        // Early stopping
        if (n_iter_ > 10 && cost_history_.size() > 10) {
            double improvement = cost_history_[cost_history_.size() - 11] 
                               - cost_history_.back();
            if (std::abs(improvement) < tolerance_) {
                if (verbose_) {
                    std::cout << "Early stopping at iteration " << n_iter_ << std::endl;
                }
                break;
            }
        }
    }
    
    if (n_iter_ >= max_iter_ && verbose_) {
        std::cout << "Reached maximum iterations: " << max_iter_ << std::endl;
    }
}

double LinearRegression::compute_cost(const MatrixXd& X_with_bias, const VectorXd& y) const {
    double m = static_cast<double>(X_with_bias.rows());
    double J = (X_with_bias * theta_ - y).squaredNorm() / (2.0 * m);
    
    if (lambda_ > 0 && theta_.size() > 1) {
        J += (lambda_ / (2.0 * m)) * theta_.tail(theta_.size() - 1).squaredNorm();
    }
    return J;
}

// Metodi predict - ACCETTANO X SENZA BIAS
VectorXd LinearRegression::predict(const MatrixXd& X) const {
    ML_CHECK_FITTED(theta_.size() > 0, get_model_type());
    ML_CHECK_NOT_EMPTY(X, "X", get_model_type());
    ML_CHECK_FEATURE_DIMENSIONS(X.cols(), n_features_, get_model_type());
    
    // Aggiungi bias automaticamente per l'utente
    MatrixXd X_with_bias = MathUtils::add_intercept(X);
    return X_with_bias * theta_;
}

double LinearRegression::predict(const VectorXd& x) const {
    ML_CHECK_FITTED(theta_.size() > 0, get_model_type());
    ML_CHECK_FEATURE_DIMENSIONS(x.size(), n_features_, get_model_type());
    
    MatrixXd X_row(1, n_features_);
    X_row.row(0) = x;
    return predict(X_row)(0);
}

// Metodi score
double LinearRegression::score(const MatrixXd& X, const VectorXd& y) const {
    return r2_score(X, y);
}

double LinearRegression::r2_score(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    double ss_res = (y - y_pred).squaredNorm();
    double ss_tot = (y.array() - y.mean()).square().sum();
    
    if (std::abs(ss_tot) < 1e-12) return 1.0;
    return 1.0 - (ss_res / ss_tot);
}

double LinearRegression::mse(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).squaredNorm() / static_cast<double>(X.rows());
}

double LinearRegression::mae(const MatrixXd& X, const VectorXd& y) const {
    VectorXd y_pred = predict(X);
    return (y - y_pred).cwiseAbs().sum() / static_cast<double>(X.rows());
}

// Cross validation
VectorXd LinearRegression::cross_val_score(const MatrixXd& X, const VectorXd& y, 
                                         int cv, Solver solver) {
    ML_CHECK_PARAM(cv > 1, "cv", "must be > 1", "LinearRegression");
    ML_CHECK_NOT_EMPTY(X, "X", "LinearRegression");
    ML_CHECK_NOT_EMPTY(y, "y", "LinearRegression");
    ML_CHECK_XY_SIZE(X.rows(), y.size(), "LinearRegression");
    
    Eigen::Index n_samples = X.rows();
    Eigen::Index fold_size = n_samples / static_cast<Eigen::Index>(cv);
    VectorXd scores(cv);
    
    for (int i = 0; i < cv; ++i) {
        Eigen::Index start = static_cast<Eigen::Index>(i) * fold_size;
        Eigen::Index end = (i == cv-1) ? n_samples : static_cast<Eigen::Index>(i+1) * fold_size;
        
        // Indici per training e test
        std::vector<Eigen::Index> train_indices, test_indices;
        for (Eigen::Index j = 0; j < n_samples; ++j) {
            if (j >= start && j < end) {
                test_indices.push_back(j);
            } else {
                train_indices.push_back(j);
            }
        }
        
        // Costruisci matrici
        MatrixXd X_train(train_indices.size(), X.cols());
        VectorXd y_train(train_indices.size());
        MatrixXd X_test(test_indices.size(), X.cols());
        VectorXd y_test(test_indices.size());
        
        for (size_t j = 0; j < train_indices.size(); ++j) {
            X_train.row(static_cast<Eigen::Index>(j)) = X.row(train_indices[j]);
            y_train(static_cast<Eigen::Index>(j)) = y(train_indices[j]);
        }
        
        for (size_t j = 0; j < test_indices.size(); ++j) {
            X_test.row(static_cast<Eigen::Index>(j)) = X.row(test_indices[j]);
            y_test(static_cast<Eigen::Index>(j)) = y(test_indices[j]);
        }
        
        LinearRegression lr(0.01, 1000, 0.0, solver);
        lr.fit(X_train, y_train);
        scores(i) = lr.score(X_test, y_test);
    }
    
    return scores;
}

// Serializzazione
void LinearRegression::serialize_binary(std::ostream& out) const {
    using namespace utils;
    
    out.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&max_iter_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&lambda_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&tolerance_), sizeof(double));
    out.write(reinterpret_cast<const char*>(&verbose_), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&solver_), sizeof(Solver));
    out.write(reinterpret_cast<const char*>(&n_features_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_iter_), sizeof(int));
    
    eigen_utils::serialize_eigen_vector(theta_, out);
    
    size_t cost_size = cost_history_.size();
    out.write(reinterpret_cast<const char*>(&cost_size), sizeof(size_t));
    if (cost_size > 0) {
        out.write(reinterpret_cast<const char*>(cost_history_.data()), 
                 cost_size * sizeof(double));
    }
}

void LinearRegression::deserialize_binary(std::istream& in) {
    using namespace utils;
    
    in.read(reinterpret_cast<char*>(&learning_rate_), sizeof(double));
    in.read(reinterpret_cast<char*>(&max_iter_), sizeof(int));
    in.read(reinterpret_cast<char*>(&lambda_), sizeof(double));
    in.read(reinterpret_cast<char*>(&tolerance_), sizeof(double));
    in.read(reinterpret_cast<char*>(&verbose_), sizeof(bool));
    in.read(reinterpret_cast<char*>(&solver_), sizeof(Solver));
    in.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
    in.read(reinterpret_cast<char*>(&n_iter_), sizeof(int));
    
    eigen_utils::deserialize_eigen_vector(theta_, in);
    
    size_t cost_size;
    in.read(reinterpret_cast<char*>(&cost_size), sizeof(size_t));
    cost_history_.resize(cost_size);
    if (cost_size > 0) {
        in.read(reinterpret_cast<char*>(cost_history_.data()), 
               cost_size * sizeof(double));
    }
}

// Utility
std::string LinearRegression::solver_to_string(Solver solver) {
    switch(solver) {
        case GRADIENT_DESCENT: return "Gradient Descent";
        case NORMAL_EQUATION: return "Normal Equation";
        case SVD: return "SVD";
        default: return "Unknown";
    }
}

std::string LinearRegression::to_string() const {
    std::ostringstream oss;
    oss << "LinearRegression [" 
        << "Solver: " << solver_to_string(solver_)
        << ", Features: " << n_features_
        << ", Iterations: " << n_iter_
        << ", λ: " << std::fixed << std::setprecision(4) << lambda_
        << ", LR: " << learning_rate_
        << "]";
    return oss.str();
}