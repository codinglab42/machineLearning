#include "binary_cross_entropy_loss.h"
#include "components/loss/binary_cross_entropy_loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_binary_cross_entropy_loss(py::module_& m) {
    py::class_<loss::BinaryCrossEntropyLoss, loss::Loss,
               std::shared_ptr<loss::BinaryCrossEntropyLoss>>(m, "BinaryCrossEntropyLoss")
        .def(py::init<>())
        .def("compute_vector",
             static_cast<double (loss::BinaryCrossEntropyLoss::*)(const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
                 &loss::BinaryCrossEntropyLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             static_cast<double (loss::BinaryCrossEntropyLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::BinaryCrossEntropyLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             static_cast<Eigen::MatrixXd (loss::BinaryCrossEntropyLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::BinaryCrossEntropyLoss::gradient),
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::BinaryCrossEntropyLoss::name)
        .def("__repr__", [](const loss::BinaryCrossEntropyLoss& loss) {
            return "BinaryCrossEntropyLoss()";
        });
}