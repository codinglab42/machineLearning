#include "l2_regularizer.h"
#include "components/regularizers/l2_regularizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_l2_regularizer(py::module_& m) {
    py::class_<models::L2Regularizer, models::Regularizer,
               std::shared_ptr<models::L2Regularizer>>(m, "L2Regularizer")
        .def(py::init<double>(), py::arg("strength") = 0.01)
        .def("compute_loss_matrix",
             static_cast<double (models::L2Regularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::L2Regularizer::compute_loss),
             py::arg("weights"),
             "L2 loss: 0.5 * λ * Σw²")
        .def("compute_loss_vector",
             static_cast<double (models::L2Regularizer::*)(const Eigen::VectorXd&) const>(
                 &models::L2Regularizer::compute_loss),
             py::arg("bias"),
             "L2 loss: 0.5 * λ * Σb²")
        .def("compute_gradient_matrix",
             static_cast<Eigen::MatrixXd (models::L2Regularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::L2Regularizer::compute_gradient),
             py::arg("weights"),
             "L2 gradient: λ * w")
        .def("compute_gradient_vector",
             static_cast<Eigen::VectorXd (models::L2Regularizer::*)(const Eigen::VectorXd&) const>(
                 &models::L2Regularizer::compute_gradient),
             py::arg("bias"),
             "L2 gradient: λ * b")
        .def("clone", &models::L2Regularizer::clone)
        .def("__repr__", [](const models::L2Regularizer& reg) {
            return "L2Regularizer(strength=" + std::to_string(reg.get_strength()) + ")";
        });
}