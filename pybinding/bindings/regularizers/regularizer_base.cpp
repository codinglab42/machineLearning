#include "regularizer_base.h"
#include "components/regularizers/regularizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_regularizer_base(py::module_& m) {
    py::class_<models::Regularizer, std::shared_ptr<models::Regularizer>>(m, "Regularizer")
        .def("compute_loss_matrix",
             [](const models::Regularizer& reg, const Eigen::MatrixXd& weights) -> double {
                 return reg.compute_loss(weights);
             },
             py::arg("weights"),
             "Compute regularization loss for weight matrix")
        .def("compute_loss_vector",
             [](const models::Regularizer& reg, const Eigen::VectorXd& bias) -> double {
                 return reg.compute_loss(bias);
             },
             py::arg("bias"),
             "Compute regularization loss for bias vector")
        .def("compute_gradient_matrix",
             [](const models::Regularizer& reg, const Eigen::MatrixXd& weights) -> Eigen::MatrixXd {
                 return reg.compute_gradient(weights);
             },
             py::arg("weights"),
             "Compute gradient for weight matrix")
        .def("compute_gradient_vector",
             [](const models::Regularizer& reg, const Eigen::VectorXd& bias) -> Eigen::VectorXd {
                 return reg.compute_gradient(bias);
             },
             py::arg("bias"),
             "Compute gradient for bias vector")
        .def("get_strength", &models::Regularizer::get_strength)
        .def("set_strength", &models::Regularizer::set_strength, py::arg("strength"))
        .def("get_type_str", &models::Regularizer::get_type_str)
        .def("get_type", &models::Regularizer::get_type)
        .def("clone", &models::Regularizer::clone)
        .def("__repr__", [](const models::Regularizer& reg) {
            return std::string("Regularizer(") + reg.get_type_str() +
                   ", strength=" + std::to_string(reg.get_strength()) + ")";
        });
}