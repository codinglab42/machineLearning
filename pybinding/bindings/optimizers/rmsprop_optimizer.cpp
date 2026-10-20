#include "rmsprop_optimizer.h"
#include "components/optimizers/rmsprop_optimizer.h"
#include <pybind11/eigen.h>
#include <memory>

namespace py = pybind11;

void bind_rmsprop_optimizer(py::module_& m) {
    py::class_<models::RMSpropOptimizer, models::Optimizer,
               std::shared_ptr<models::RMSpropOptimizer>>(m, "RMSpropOptimizer")
        .def(py::init<double, double, double, double, double>(),
             py::arg("learning_rate") = 0.001,
             py::arg("rho") = 0.9,
             py::arg("epsilon") = 1e-8,
             py::arg("decay") = 0.0,
             py::arg("momentum") = 0.0)
        .def("update_weights",
             [](models::RMSpropOptimizer& optimizer,
                Eigen::Ref<Eigen::MatrixXd> weights,
                const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
                 optimizer.update_weights(weights, gradient);
             },
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             [](models::RMSpropOptimizer& optimizer,
                Eigen::Ref<Eigen::VectorXd> bias,
                const Eigen::Ref<const Eigen::VectorXd>& gradient) {
                 optimizer.update_bias(bias, gradient);
             },
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::RMSpropOptimizer::reset)
        .def("clone", [](const models::RMSpropOptimizer& optimizer) {
            return std::shared_ptr<models::Optimizer>(optimizer.clone());
        })
        .def("__repr__", [](const models::RMSpropOptimizer& optimizer) {
            return "RMSpropOptimizer(lr=" +
                   std::to_string(optimizer.get_learning_rate()) + ")";
        });
}
