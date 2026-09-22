#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "models/linear_regression.h"
#include "utils/binding_utils.h"

namespace py = pybind11;

void bind_linear_regression(py::module_& m) {
    py::class_<models::LinearRegression, models::Estimator, 
               std::shared_ptr<models::LinearRegression>> cls(m, "LinearRegression");

    cls
        // Costruttori
        .def(py::init<>())
        .def(py::init<double, int, double, models::LinearRegression::Solver>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("solver") = models::LinearRegression::Solver::GRADIENT_DESCENT)

        // Training (fit)
        .def("fit", [](models::LinearRegression& self, PyArrayDouble X, PyArrayDouble y) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            self.fit(X_mat, y_vec);
            return &self; // Per permettere il method chaining in Python
        }, py::arg("X"), py::arg("y"), py::return_value_policy::reference)

        // Predict (gestisce sia input 2D MatrixXd che 1D VectorXd)
        .def("predict", [](const models::LinearRegression& self, const PyArrayDouble& X) -> py::object {
            if (X.ndim() == 1) {
                Eigen::VectorXd x_vec = array_to_vector(X, "X");
                double res = self.predict(x_vec);
                return py::cast(res);
            }
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd res = self.predict(X_mat);
            return py::cast(res);
        }, py::arg("X"))

        // Metrics & Scoring
        .def("score", [](const models::LinearRegression& self, PyArrayDouble X, PyArrayDouble y) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            return self.score(X_mat, y_vec);
        }, py::arg("X"), py::arg("y"))

        .def("mse", [](const models::LinearRegression& self, PyArrayDouble X, PyArrayDouble y) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            return self.mse(X_mat, y_vec);
        }, py::arg("X"), py::arg("y"))

        .def("mae", [](const models::LinearRegression& self, PyArrayDouble X, PyArrayDouble y) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            return self.mae(X_mat, y_vec);
        }, py::arg("X"), py::arg("y"))

        .def("r2_score", [](const models::LinearRegression& self, PyArrayDouble X, PyArrayDouble y) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            return self.r2_score(X_mat, y_vec);
        }, py::arg("X"), py::arg("y"))

        // Metodo Statico per la Validation
        .def_static("cross_val_score", [](PyArrayDouble X, PyArrayDouble y, int cv, 
                                          models::LinearRegression::Solver solver) {
            Eigen::MatrixXd X_mat = array_to_matrix(X, "X");
            Eigen::VectorXd y_vec = array_to_vector(y, "y");
            return models::LinearRegression::cross_val_score(X_mat, y_vec, cv, solver);
        }, py::arg("X"), py::arg("y"), py::arg("cv") = 5, 
           py::arg("solver") = models::LinearRegression::Solver::GRADIENT_DESCENT)

        // Properties / Getters
        .def_property_readonly("coefficients", &models::LinearRegression::coefficients)
        .def_property_readonly("intercept", &models::LinearRegression::intercept)
        .def_property_readonly("cost_history", &models::LinearRegression::cost_history)

        // Setters
        .def("set_max_iterations", &models::LinearRegression::set_max_iterations, py::arg("max_iter"))
        .def("set_lambda", &models::LinearRegression::set_lambda, py::arg("lambda"))

        // Serialization & String Representation
        .def("to_string", &models::LinearRegression::to_string)
        .def("get_model_type", &models::LinearRegression::get_model_type)
        .def("__repr__", &models::LinearRegression::to_string);
}