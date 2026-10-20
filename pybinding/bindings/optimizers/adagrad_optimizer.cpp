#include "adagrad_optimizer.h"
#include "components/optimizers/adagrad_optimizer.h"
#include <pybind11/eigen.h>
#include <memory>

namespace py = pybind11;

void bind_adagrad_optimizer(py::module_& m) {
    py::class_<models::AdagradOptimizer, models::Optimizer,
               std::shared_ptr<models::AdagradOptimizer>>(m, "AdagradOptimizer")
        .def(py::init<double, double, double>(),
             py::arg("learning_rate") = 0.01,
             py::arg("epsilon") = 1e-8,
             py::arg("decay") = 0.0)
        .def("update_weights",
             [](models::AdagradOptimizer& optimizer,
                Eigen::Ref<Eigen::MatrixXd> weights,
                const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
                 optimizer.update_weights(weights, gradient);
             },
             py::arg("weights"), py::arg("gradient"))
        .def("update_bias",
             [](models::AdagradOptimizer& optimizer,
                Eigen::Ref<Eigen::VectorXd> bias,
                const Eigen::Ref<const Eigen::VectorXd>& gradient) {
                 optimizer.update_bias(bias, gradient);
             },
             py::arg("bias"), py::arg("gradient"))
        .def("reset", &models::AdagradOptimizer::reset)
        .def("clone", [](const models::AdagradOptimizer& optimizer) {
            return std::shared_ptr<models::Optimizer>(optimizer.clone());
        })
        .def("__repr__", [](const models::AdagradOptimizer& optimizer) {
            return "AdagradOptimizer(lr=" +
                   std::to_string(optimizer.get_learning_rate()) + ")";
        });
}
