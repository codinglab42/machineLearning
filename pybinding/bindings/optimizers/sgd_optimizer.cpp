#include "sgd_optimizer.h"
#include "components/optimizers/sgd_optimizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_sgd_optimizer(py::module_& m) {
    py::class_<models::SGDOptimizer, models::Optimizer,
               std::shared_ptr<models::SGDOptimizer>>(m, "SGDOptimizer")
        .def(py::init<double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("decay") = 0.0)
        .def("update_weights",
             static_cast<void (models::SGDOptimizer::*)(Eigen::MatrixXd&, const Eigen::MatrixXd&)>(
                 &models::SGDOptimizer::update),
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             static_cast<void (models::SGDOptimizer::*)(Eigen::VectorXd&, const Eigen::VectorXd&)>(
                 &models::SGDOptimizer::update),
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::SGDOptimizer::reset)
        .def("clone", &models::SGDOptimizer::clone)
        .def("__repr__", [](const models::SGDOptimizer& optimizer) {
            return "SGDOptimizer(lr=" + std::to_string(optimizer.get_learning_rate()) +
                   ", decay=" + std::to_string(optimizer.get_decay()) + ")";
        });
}