#include "dropout_layer.h"
#include "components/layers/dropout_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_dropout_layer(py::module_& m) {
    py::class_<layers::DropoutLayer, layers::Layer,
               std::shared_ptr<layers::DropoutLayer>>(m, "DropoutLayer")
        .def(py::init<double>(), py::arg("rate") = 0.5)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::DropoutLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::DropoutLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::DropoutLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::DropoutLayer::backward),
             py::arg("gradient"))
        .def("get_rate", &layers::DropoutLayer::get_rate)
        .def("get_type", &layers::DropoutLayer::get_type)
        .def("get_config", &layers::DropoutLayer::get_config)
        .def("get_input_size", &layers::DropoutLayer::get_input_size)
        .def("get_output_size", &layers::DropoutLayer::get_output_size)
        .def("set_input_shape", &layers::DropoutLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::DropoutLayer::get_config);
}