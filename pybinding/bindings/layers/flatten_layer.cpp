#include "flatten_layer.h"
#include "components/layers/flatten_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_flatten_layer(py::module_& m) {
    py::class_<layers::FlattenLayer, layers::Layer,
               std::shared_ptr<layers::FlattenLayer>>(m, "FlattenLayer")
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::FlattenLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::FlattenLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::FlattenLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::FlattenLayer::backward),
             py::arg("gradient"))
        .def("get_type", &layers::FlattenLayer::get_type)
        .def("get_config", &layers::FlattenLayer::get_config)
        .def("get_input_size", &layers::FlattenLayer::get_input_size)
        .def("get_output_size", &layers::FlattenLayer::get_output_size)
        .def("set_input_shape", &layers::FlattenLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::FlattenLayer::get_config);
}