#include "categorical_cross_entropy_loss.h"
#include "components/loss/categorical_cross_entropy_loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_categorical_cross_entropy_loss(py::module_& m) {
    py::class_<loss::CategoricalCrossEntropyLoss, loss::Loss,
               std::shared_ptr<loss::CategoricalCrossEntropyLoss>>(m, "CategoricalCrossEntropyLoss")
        .def(py::init<>())
        .def("compute_vector",
             static_cast<double (loss::CategoricalCrossEntropyLoss::*)(const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
                 &loss::CategoricalCrossEntropyLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             static_cast<double (loss::CategoricalCrossEntropyLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::CategoricalCrossEntropyLoss::compute),
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             static_cast<Eigen::MatrixXd (loss::CategoricalCrossEntropyLoss::*)(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const>(
                 &loss::CategoricalCrossEntropyLoss::gradient),
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::CategoricalCrossEntropyLoss::name)
        .def("__repr__", [](const loss::CategoricalCrossEntropyLoss& loss) {
            return "CategoricalCrossEntropyLoss()";
        });
}