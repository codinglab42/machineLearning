#include "logistic_regression.h"
#include "models/logistic_regression.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_logistic_regression(py::module_& m) {
    py::class_<models::LogisticRegression, models::Estimator,
               std::shared_ptr<models::LogisticRegression>> cls(m, "LogisticRegression");
    
    cls
        // Costruttori
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.1,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)
        
        // Fit e Predict
        .def("fit", 
             py::overload_cast<const Eigen::MatrixXd&, const Eigen::VectorXd&>(
                 &models::LogisticRegression::fit),
             py::arg("X"), py::arg("y"))
        .def("predict", &models::LogisticRegression::predict, py::arg("X"))
        .def("predict_class", &models::LogisticRegression::predict_class,
             py::arg("X"), py::arg("threshold") = 0.5)
        .def("score", &models::LogisticRegression::score, py::arg("X"), py::arg("y"))
        
        // Metriche specifiche
        .def("precision_recall_f1", &models::LogisticRegression::precision_recall_f1,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)
        .def("confusion_matrix", &models::LogisticRegression::confusion_matrix,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)
        
        // Serializzazione
        .def("save", &models::LogisticRegression::save, py::arg("filename"))
        .def("load", &models::LogisticRegression::load, py::arg("filename"))
        .def("to_string", &models::LogisticRegression::to_string)
        
        // Proprietà
        .def_property_readonly("coefficients", &models::LogisticRegression::coefficients)
        .def_property_readonly("intercept", &models::LogisticRegression::intercept)
        .def_property_readonly("cost_history", &models::LogisticRegression::cost_history)
        .def_property_readonly("accuracy_history", &models::LogisticRegression::accuracy_history)
        
        // Setters
        .def("set_max_iterations", &models::LogisticRegression::set_max_iterations,
             py::arg("max_iter"))
        .def("set_lambda", &models::LogisticRegression::set_lambda, py::arg("lambda"))
        .def("set_tolerance", &models::LogisticRegression::set_tolerance,
             py::arg("tolerance"))
        .def("set_verbose", &models::LogisticRegression::set_verbose,
             py::arg("verbose"))
        
        .def("__repr__", &models::LogisticRegression::to_string);
}