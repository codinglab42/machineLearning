#include "estimator.h"
#include "models/estimator.h"

namespace py = pybind11;

void bind_estimator(py::module_& m) {
    py::class_<models::Estimator, std::shared_ptr<models::Estimator>>(m, "Estimator")
        .def("fit", 
             [](models::Estimator& estimator, const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
                 estimator.fit(X, y);
             },
             py::arg("X"), py::arg("y"))
        .def("predict", 
             [](const models::Estimator& estimator, const Eigen::MatrixXd& X) {
                 return estimator.predict(X);
             },
             py::arg("X"))
        .def("score", 
             [](const models::Estimator& estimator, const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
                 return estimator.score(X, y);
             },
             py::arg("X"), py::arg("y"))
        .def("save", &models::Estimator::save, py::arg("filename"))
        .def("load", &models::Estimator::load, py::arg("filename"))
        .def("to_string", &models::Estimator::to_string)
        .def("get_model_type", &models::Estimator::get_model_type)
        .def("set_learning_rate", 
             [](models::Estimator& estimator, double lr) {
                 estimator.set_learning_rate(lr);
             },
             py::arg("learning_rate"))
        .def("get_learning_rate", 
             [](const models::Estimator& estimator) {
                 return estimator.get_learning_rate();
             })
        .def("set_optimizer", 
             [](models::Estimator& estimator, models::OptimizerType type, double learning_rate) {
                 estimator.set_optimizer(type, learning_rate);
             },
             py::arg("type"), py::arg("learning_rate") = 0.01)
        .def("__repr__", &models::Estimator::to_string);
}