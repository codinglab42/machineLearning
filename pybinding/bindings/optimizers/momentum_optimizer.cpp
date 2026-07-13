#include "momentum_optimizer.h"
#include "components/optimizers/momentum_optimizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_momentum_optimizer(py::module_& m) {
    py::class_<models::MomentumOptimizer, models::Optimizer,
               std::shared_ptr<models::MomentumOptimizer>>(m, "MomentumOptimizer")
        .def(py::init<double, double, double, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("momentum") = 0.9,
             py::arg("decay") = 0.0,
             py::arg("nesterov") = false)
        .def("update_weights",
             [](models::MomentumOptimizer& optimizer, Eigen::MatrixXd& weights, 
                const Eigen::MatrixXd& gradient) {
                 optimizer.update(weights, gradient);
             },
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             [](models::MomentumOptimizer& optimizer, Eigen::VectorXd& bias, 
                const Eigen::VectorXd& gradient) {
                 optimizer.update(bias, gradient);
             },
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::MomentumOptimizer::reset)
        .def("clone", &models::MomentumOptimizer::clone)
        .def("__repr__", [](const models::MomentumOptimizer& optimizer) {
            return "MomentumOptimizer(lr=" + std::to_string(optimizer.get_learning_rate()) + ")";
        });
}