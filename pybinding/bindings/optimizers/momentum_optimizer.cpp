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
             static_cast<void (models::MomentumOptimizer::*)(Eigen::MatrixXd&, const Eigen::MatrixXd&)>(
                 &models::MomentumOptimizer::update),
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             static_cast<void (models::MomentumOptimizer::*)(Eigen::VectorXd&, const Eigen::VectorXd&)>(
                 &models::MomentumOptimizer::update),
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::MomentumOptimizer::reset)
        .def("clone", &models::MomentumOptimizer::clone)
        .def("__repr__", [](const models::MomentumOptimizer& optimizer) {
            return "MomentumOptimizer(lr=" + std::to_string(optimizer.get_learning_rate()) +
                   ", momentum=" + std::to_string(optimizer.get_momentum()) +
                   ", nesterov=" + std::to_string(optimizer.get_nesterov()) + ")";
        });
}