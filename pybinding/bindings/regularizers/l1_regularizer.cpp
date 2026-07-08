#include "l1_regularizer.h"
#include "components/regularizers/l1_regularizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_l1_regularizer(py::module_& m) {
    py::class_<models::L1Regularizer, models::Regularizer,
               std::shared_ptr<models::L1Regularizer>>(m, "L1Regularizer")
        .def(py::init<double>(), py::arg("strength") = 0.01)
        .def("compute_loss_matrix",
             static_cast<double (models::L1Regularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::L1Regularizer::compute_loss),
             py::arg("weights"),
             "L1 loss: λ * Σ|w|")
        .def("compute_loss_vector",
             static_cast<double (models::L1Regularizer::*)(const Eigen::VectorXd&) const>(
                 &models::L1Regularizer::compute_loss),
             py::arg("bias"),
             "L1 loss: λ * Σ|b|")
        .def("compute_gradient_matrix",
             static_cast<Eigen::MatrixXd (models::L1Regularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::L1Regularizer::compute_gradient),
             py::arg("weights"),
             "L1 gradient: λ * sign(w)")
        .def("compute_gradient_vector",
             static_cast<Eigen::VectorXd (models::L1Regularizer::*)(const Eigen::VectorXd&) const>(
                 &models::L1Regularizer::compute_gradient),
             py::arg("bias"),
             "L1 gradient: λ * sign(b)")
        .def("clone", &models::L1Regularizer::clone)
        .def("__repr__", [](const models::L1Regularizer& reg) {
            return "L1Regularizer(strength=" + std::to_string(reg.get_strength()) + ")";
        });
}