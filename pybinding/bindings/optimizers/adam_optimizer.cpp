#include "adam_optimizer.h"
#include "components/optimizers/adam_optimizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_adam_optimizer(py::module_& m) {
    py::class_<models::AdamOptimizer, models::Optimizer,
               std::shared_ptr<models::AdamOptimizer>>(m, "AdamOptimizer")
        .def(py::init<double, double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("decay") = 0.0)
        .def("update_weights",
             static_cast<void (models::AdamOptimizer::*)(Eigen::MatrixXd&, const Eigen::MatrixXd&)>(
                 &models::AdamOptimizer::update),
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             static_cast<void (models::AdamOptimizer::*)(Eigen::VectorXd&, const Eigen::VectorXd&)>(
                 &models::AdamOptimizer::update),
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::AdamOptimizer::reset)
        .def("clone", &models::AdamOptimizer::clone)
        .def("__repr__", [](const models::AdamOptimizer& optimizer) {
            return "AdamOptimizer(lr=" + std::to_string(optimizer.get_learning_rate()) +
                   ", beta1=" + std::to_string(optimizer.get_beta1()) +
                   ", beta2=" + std::to_string(optimizer.get_beta2()) + ")";
        });
}