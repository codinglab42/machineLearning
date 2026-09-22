#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "models/logistic_regression.h"
#include "utils/binding_utils.h"

namespace py = pybind11;

void bind_logistic_regression(py::module_& m) {
    py::class_<models::LogisticRegression, models::Estimator,
               std::shared_ptr<models::LogisticRegression>> cls(m, "LogisticRegression");

    cls
        // Costruttori
        .def(py::init<>())
        .def(py::init<double, int, double, double, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("max_iter") = 1000,
             py::arg("lambda") = 0.0,
             py::arg("tolerance") = 1e-4,
             py::arg("verbose") = false)

        // Fit
        .def("fit", [](models::LogisticRegression& self, PyArrayDouble X, PyArrayDouble y) {
            auto X_map = array_to_matrix(X, "X");
            auto y_map = array_to_vector(y, "y");

            if (X_map.rows() != y_map.rows()) {
                throw std::invalid_argument(
                    "Mismatch tra il numero di campioni in X (" + std::to_string(X_map.rows()) +
                    ") e y (" + std::to_string(y_map.rows()) + ")."
                );
            }
            self.fit(X_map, y_map);
        }, py::arg("X"), py::arg("y"))

        // Predict (restituisce probabilità) e Predict Class (0 o 1)
        .def("predict", [](const models::LogisticRegression& self, PyArrayDouble X) {
            auto X_map = array_to_matrix(X, "X");
            return self.predict(X_map);
        }, py::arg("X"))

        .def("predict_proba", [](const models::LogisticRegression& self, PyArrayDouble X) {
            auto X_map = array_to_matrix(X, "X");
            return self.predict(X_map); // In LogisticRegression predict() restituisce p = sigmoid(X * theta)
        }, py::arg("X"))

        .def("predict_class", [](const models::LogisticRegression& self, PyArrayDouble X, double threshold) {
            auto X_map = array_to_matrix(X, "X");
            return self.predict_class(X_map, threshold);
        }, py::arg("X"), py::arg("threshold") = 0.5)

        // Score e Metriche
        .def("score", [](const models::LogisticRegression& self, PyArrayDouble X, PyArrayDouble y) {
            auto X_map = array_to_matrix(X, "X");
            auto y_map = array_to_vector(y, "y");
            return self.score(X_map, y_map);
        }, py::arg("X"), py::arg("y"))

        .def("precision_recall_f1", [](const models::LogisticRegression& self, PyArrayDouble X, PyArrayDouble y, double threshold) {
            auto X_map = array_to_matrix(X, "X");
            auto y_map = array_to_vector(y, "y");
            return self.precision_recall_f1(X_map, y_map, threshold);
        }, py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)

        .def("confusion_matrix", [](const models::LogisticRegression& self, PyArrayDouble X, PyArrayDouble y, double threshold) {
            auto X_map = array_to_matrix(X, "X");
            auto y_map = array_to_vector(y, "y");
            return self.confusion_matrix(X_map, y_map, threshold);
        }, py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5)

        // Serializzazione / Info
        .def("to_string", &models::LogisticRegression::to_string)
        .def("get_model_type", &models::LogisticRegression::get_model_type)

        // Proprietà (Getters)
        .def_property_readonly("coefficients", &models::LogisticRegression::coefficients)
        .def_property_readonly("intercept", &models::LogisticRegression::intercept)
        .def_property_readonly("cost_history", &models::LogisticRegression::cost_history)
        .def_property_readonly("accuracy_history", &models::LogisticRegression::accuracy_history)
        .def_property_readonly("n_iter", &models::LogisticRegression::get_n_iter)

        // Setters
        .def("set_max_iterations", &models::LogisticRegression::set_max_iterations, py::arg("max_iter"))
        .def("set_lambda", &models::LogisticRegression::set_lambda, py::arg("lambda"))
        .def("set_tolerance", &models::LogisticRegression::set_tolerance, py::arg("tolerance"))
        .def("set_verbose", &models::LogisticRegression::set_verbose, py::arg("verbose"))

        .def("__repr__", &models::LogisticRegression::to_string);
}