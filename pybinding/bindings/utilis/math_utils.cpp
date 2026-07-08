#include "math_utils.h"
#include "utils/math_utils.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_math_utils(py::module_& m) {
    py::class_<utils::MathUtils>(m, "MathUtils")
        // Sigmoid functions
        .def_static("sigmoid",
                    py::overload_cast<double>(&utils::MathUtils::sigmoid),
                    py::arg("z"),
                    "Compute sigmoid function for scalar")
        .def_static("sigmoid_vec",
                    &utils::MathUtils::sigmoid_vec,
                    py::arg("z"),
                    "Compute sigmoid for vector")
        .def_static("sigmoid_derivative",
                    py::overload_cast<double>(&utils::MathUtils::sigmoid_derivative),
                    py::arg("z"),
                    "Compute sigmoid derivative for scalar")
        .def_static("sigmoid_derivative_vec",
                    &utils::MathUtils::sigmoid_derivative_vec,
                    py::arg("z"),
                    "Compute sigmoid derivative for vector")
        
        // Matrix operations
        .def_static("add_intercept",
                    &utils::MathUtils::add_intercept,
                    py::arg("X"),
                    "Add intercept column to matrix")
        .def_static("safe_log",
                    &utils::MathUtils::safe_log,
                    py::arg("v"),
                    "Safe log for vector (clips to eps)")
        .def_static("safe_log_matrix",
                    &utils::MathUtils::safe_log_matrix,
                    py::arg("m"),
                    "Safe log for matrix (clips to eps)")
        
        // Weight initialization
        .def_static("he_initialization",
                    &utils::MathUtils::he_initialization,
                    py::arg("input_size"), py::arg("output_size"),
                    "He initialization for ReLU (std=sqrt(2/fan_in))")
        .def_static("xavier_initialization",
                    &utils::MathUtils::xavier_initialization,
                    py::arg("input_size"), py::arg("output_size"),
                    "Xavier initialization for sigmoid/tanh (limit=sqrt(6/(fan_in+fan_out)))")
        
        // One-hot encoding
        .def_static("one_hot_encode",
                    &utils::MathUtils::one_hot_encode,
                    py::arg("labels"), py::arg("num_classes"),
                    "One-hot encode labels")
        
        // Normalization
        .def_static("normalize_minmax",
                    [](Eigen::MatrixXd& X, Eigen::VectorXd& min_vals,
                       Eigen::VectorXd& max_vals, const std::string& model_type) {
                        utils::MathUtils::normalize_minmax(X, min_vals, max_vals, model_type);
                    },
                    py::arg("X"), py::arg("min_vals"), py::arg("max_vals"),
                    py::arg("model_type") = "",
                    "Normalize data to [0, 1] range")
        .def_static("standardize_features",
                    [](Eigen::MatrixXd& X, Eigen::VectorXd& mean,
                       Eigen::VectorXd& std, const std::string& model_type) {
                        utils::MathUtils::standardize_features(X, mean, std, model_type);
                    },
                    py::arg("X"), py::arg("mean"), py::arg("std"),
                    py::arg("model_type") = "",
                    "Standardize features (mean=0, std=1)")
        
        // Machine learning utilities
        .def_static("train_test_split",
                    &utils::MathUtils::train_test_split,
                    py::arg("X"), py::arg("y"),
                    py::arg("test_size") = 0.2,
                    py::arg("random_state") = 42,
                    py::arg("model_type") = "",
                    "Split data into train and test sets")
        .def_static("accuracy_score",
                    &utils::MathUtils::accuracy_score,
                    py::arg("y_true"), py::arg("y_pred"),
                    py::arg("model_type") = "",
                    "Compute accuracy score")
        
        // Gradient computation
        .def_static("compute_gradient_linear",
                    &utils::MathUtils::compute_gradient_linear,
                    py::arg("X"), py::arg("y"), py::arg("theta"),
                    py::arg("lambda") = 0.0,
                    "Compute gradient for linear regression")
        .def_static("compute_gradient_logistic",
                    &utils::MathUtils::compute_gradient_logistic,
                    py::arg("X"), py::arg("y"), py::arg("theta"),
                    py::arg("lambda") = 0.0,
                    "Compute gradient for logistic regression")
        
        // Utility methods
        .def("__repr__", []() {
            return "MathUtils - Mathematical utilities for ML";
        });
}