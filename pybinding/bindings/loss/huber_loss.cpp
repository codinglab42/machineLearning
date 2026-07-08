#include "huber_loss.h"
#include "components/loss/huber_loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_huber_loss(py::module_& m) {
    py::class_<loss::HuberLoss, loss::Loss,
               std::shared_ptr<loss::HuberLoss>>(m, "HuberLoss")
        .def(py::init<double>(), py::arg("delta") = 1.0)
        .def("compute_vector",
             static_cast<double (loss::HuberLoss::*)(const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
                 &loss::HuberLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             static_cast<double (loss::HuberLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::HuberLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             static_cast<Eigen::MatrixXd (loss::HuberLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::HuberLoss::gradient),
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::HuberLoss::name)
        .def("set_delta", &loss::HuberLoss::set_delta, py::arg("delta"))
        .def("get_delta", &loss::HuberLoss::get_delta)
        .def("__repr__", [](const loss::HuberLoss& loss) {
            return "HuberLoss(delta=" + std::to_string(loss.get_delta()) + ")";
        });
}