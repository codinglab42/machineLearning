#include "mean_squared_error_loss.h"
#include "components/loss/mean_squared_error_loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_mean_squared_error_loss(py::module_& m) {
    py::class_<loss::MeanSquaredErrorLoss, loss::Loss,
               std::shared_ptr<loss::MeanSquaredErrorLoss>>(m, "MeanSquaredErrorLoss")
        .def(py::init<>())
        .def("compute_vector",
             static_cast<double (loss::MeanSquaredErrorLoss::*)(const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
                 &loss::MeanSquaredErrorLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             static_cast<double (loss::MeanSquaredErrorLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::MeanSquaredErrorLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             static_cast<Eigen::MatrixXd (loss::MeanSquaredErrorLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::MeanSquaredErrorLoss::gradient),
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::MeanSquaredErrorLoss::name)
        .def("__repr__", [](const loss::MeanSquaredErrorLoss& loss) {
            return "MeanSquaredErrorLoss()";
        });
}