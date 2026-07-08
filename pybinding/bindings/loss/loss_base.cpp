#include "loss_base.h"
#include "components/loss/loss.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_loss_base(py::module_& m) {
    py::class_<loss::Loss, std::shared_ptr<loss::Loss>>(m, "Loss")
        .def("compute_vector",
             [](const loss::Loss& loss, const Eigen::VectorXd& y_true,
                const Eigen::VectorXd& y_pred) -> double {
                 return loss.compute(y_true, y_pred);
             },
             py::arg("y_true"), py::arg("y_pred"))
        .def("compute_matrix",
             [](const loss::Loss& loss, const Eigen::MatrixXd& y_true,
                const Eigen::MatrixXd& y_pred) -> double {
                 return loss.compute(y_true, y_pred);
             },
             py::arg("y_true"), py::arg("y_pred"))
        .def("gradient",
             [](const loss::Loss& loss, const Eigen::MatrixXd& y_true,
                const Eigen::MatrixXd& y_pred) -> Eigen::MatrixXd {
                 return loss.gradient(y_true, y_pred);
             },
             py::arg("y_true"), py::arg("y_pred"))
        .def("name", &loss::Loss::name)
        .def("__repr__", [](const loss::Loss& loss) {
            return std::string("Loss(") + loss.name() + ")";
        });
}