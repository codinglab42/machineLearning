#include "mean_absolute_error_loss.h"
#include "components/loss/mean_absolute_error_loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_mean_absolute_error_loss(py::module_& m) {
    py::class_<loss::MeanAbsoluteErrorLoss, loss::Loss,
               std::shared_ptr<loss::MeanAbsoluteErrorLoss>>(m, "MeanAbsoluteErrorLoss")
        .def(py::init<>())
        .def("compute_vector",
             static_cast<double (loss::MeanAbsoluteErrorLoss::*)(const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
                 &loss::MeanAbsoluteErrorLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             static_cast<double (loss::MeanAbsoluteErrorLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::MeanAbsoluteErrorLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             static_cast<Eigen::MatrixXd (loss::MeanAbsoluteErrorLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::MeanAbsoluteErrorLoss::gradient),
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::MeanAbsoluteErrorLoss::name)
        .def("__repr__", [](const loss::MeanAbsoluteErrorLoss& loss) {
            return "MeanAbsoluteErrorLoss()";
        });
}