#include "elastic_net_regularizer.h"
#include "components/regularizers/elastic_net_regularizer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_elastic_net_regularizer(py::module_& m) {
    py::class_<models::ElasticNetRegularizer, models::Regularizer,
               std::shared_ptr<models::ElasticNetRegularizer>>(m, "ElasticNetRegularizer")
        .def(py::init<double, double>(),
             py::arg("strength") = 0.01,
             py::arg("l1_ratio") = 0.5,
             "Elastic Net: λ * (l1_ratio * L1 + (1-l1_ratio) * L2)")
        .def("compute_loss_matrix",
             static_cast<double (models::ElasticNetRegularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::ElasticNetRegularizer::compute_loss),
             py::arg("weights"),
             "Elastic Net loss")
        .def("compute_loss_vector",
             static_cast<double (models::ElasticNetRegularizer::*)(const Eigen::VectorXd&) const>(
                 &models::ElasticNetRegularizer::compute_loss),
             py::arg("bias"),
             "Elastic Net loss for bias")
        .def("compute_gradient_matrix",
             static_cast<Eigen::MatrixXd (models::ElasticNetRegularizer::*)(const Eigen::MatrixXd&) const>(
                 &models::ElasticNetRegularizer::compute_gradient),
             py::arg("weights"),
             "Elastic Net gradient")
        .def("compute_gradient_vector",
             static_cast<Eigen::VectorXd (models::ElasticNetRegularizer::*)(const Eigen::VectorXd&) const>(
                 &models::ElasticNetRegularizer::compute_gradient),
             py::arg("bias"),
             "Elastic Net gradient for bias")
        .def("get_l1_ratio", &models::ElasticNetRegularizer::get_l1_ratio,
             "Get L1 ratio (0 = L2 only, 1 = L1 only)")
        .def("set_l1_ratio", &models::ElasticNetRegularizer::set_l1_ratio, py::arg("ratio"),
             "Set L1 ratio (0 = L2 only, 1 = L1 only)")
        .def("clone", &models::ElasticNetRegularizer::clone)
        .def("__repr__", [](const models::ElasticNetRegularizer& reg) {
            return "ElasticNetRegularizer(strength=" + std::to_string(reg.get_strength()) +
                   ", l1_ratio=" + std::to_string(reg.get_l1_ratio()) + ")";
        });
}