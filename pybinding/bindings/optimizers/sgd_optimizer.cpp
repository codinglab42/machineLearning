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
             [](models::SGDOptimizer& optimizer, Eigen::MatrixXd& weights, 
                const Eigen::MatrixXd& gradient) {
                 // ⭐ Chiamata esplicita al metodo update
                 optimizer.update(weights, gradient);
             },
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             [](models::SGDOptimizer& optimizer, Eigen::VectorXd& bias, 
                const Eigen::VectorXd& gradient) {
                 optimizer.update(bias, gradient);
             },
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::SGDOptimizer::reset)
        .def("clone", &models::SGDOptimizer::clone)
        .def("__repr__", [](const models::SGDOptimizer& optimizer) {
            return "SGDOptimizer(lr=" + std::to_string(optimizer.get_learning_rate()) +
                   ", decay=" + std::to_string(optimizer.get_decay()) + ")";
        });
}