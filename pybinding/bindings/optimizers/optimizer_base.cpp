#include "optimizer_base.h"
#include "components/optimizers/optimizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_optimizer_base(py::module_& m) {
    py::class_<models::Optimizer, std::shared_ptr<models::Optimizer>>(m, "Optimizer")
        .def("update_weights",
             [](models::Optimizer& optimizer, Eigen::MatrixXd& weights,
                const Eigen::MatrixXd& gradient) {
                 optimizer.update_weights(weights, gradient);
             },
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             [](models::Optimizer& optimizer, Eigen::VectorXd& bias,
                const Eigen::VectorXd& gradient) {
                 optimizer.update_bias(bias, gradient);
             },
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::Optimizer::reset)
        .def("get_learning_rate", &models::Optimizer::get_learning_rate)
        .def("get_current_learning_rate", &models::Optimizer::get_current_learning_rate)
        .def("set_learning_rate", &models::Optimizer::set_learning_rate, py::arg("lr"))
        .def("get_decay", &models::Optimizer::get_decay)
        .def("set_decay", &models::Optimizer::set_decay, py::arg("decay"))
        .def("get_iterations", &models::Optimizer::get_iterations)
        .def("get_type_str", &models::Optimizer::get_type_str)
        .def("get_type", &models::Optimizer::get_type)
        .def("clone", &models::Optimizer::clone)
        .def("__repr__", [](const models::Optimizer& optimizer) {
            return std::string("Optimizer(") + optimizer.get_type_str() +
                   ", lr=" + std::to_string(optimizer.get_learning_rate()) + ")";
        });
}
