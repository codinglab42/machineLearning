#include "linear_regression.h"
#include "models/linear_regression.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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
        
        // Fit e Predict
        .def("fit", 
             py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
                 &models::LinearRegression::fit),
             py::arg("X"), py::arg("y"))
        .def("predict", 
             py::overload_cast<const Eigen::MatrixXd&>(
                 &models::LinearRegression::predict, py::const_),
             py::arg("X"))
        .def("predict_single", 
             py::overload_cast<const Eigen::VectorXd&>(
                 &models::LinearRegression::predict, py::const_),
             py::arg("x"))
        .def("score", &models::LinearRegression::score, py::arg("X"), py::arg("y"))
        
        // Metriche
        .def("mse", &models::LinearRegression::mse, py::arg("X"), py::arg("y"))
        .def("mae", &models::LinearRegression::mae, py::arg("X"), py::arg("y"))
        .def("r2_score", &models::LinearRegression::r2_score, py::arg("X"), py::arg("y"))
        
        // Serializzazione
        .def("save", &models::LinearRegression::save, py::arg("filename"))
        .def("load", &models::LinearRegression::load, py::arg("filename"))
        .def("to_string", &models::LinearRegression::to_string)
        
        // Cross-validation
        .def_static("cross_val_score", &models::LinearRegression::cross_val_score,
                    py::arg("X"), py::arg("y"),
                    py::arg("cv") = 5,
                    py::arg("solver") = models::LinearRegression::Solver::GRADIENT_DESCENT)
        
        // Proprietà
        .def_property_readonly("coefficients", &models::LinearRegression::coefficients)
        .def_property_readonly("intercept", &models::LinearRegression::intercept)
        .def_property_readonly("cost_history", &models::LinearRegression::cost_history)
        
        // Setters
        .def("set_max_iterations", &models::LinearRegression::set_max_iterations,
             py::arg("max_iter"))
        .def("set_lambda", &models::LinearRegression::set_lambda, py::arg("lambda"))
        
        .def("__repr__", &models::LinearRegression::to_string);
}