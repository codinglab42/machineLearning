#include "conv2d_layer.h"
#include "components/layers/conv2d_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_conv2d_layer(py::module_& m) {
    py::class_<layers::Conv2DLayer, layers::Layer,
               std::shared_ptr<layers::Conv2DLayer>>(m, "Conv2DLayer")
        .def(py::init<int, int, int, const std::string&, const std::string&>(),
             py::arg("filters"), py::arg("kernel_size"), py::arg("strides") = 1,
             py::arg("padding") = "valid", py::arg("activation") = "relu")
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::Conv2DLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::Conv2DLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::Conv2DLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::Conv2DLayer::backward),
             py::arg("gradient"))
        .def("get_weights", &layers::Conv2DLayer::get_weights)
        .def("set_weights", &layers::Conv2DLayer::set_weights, py::arg("weights"))
        .def("get_biases", &layers::Conv2DLayer::get_biases)
        .def("set_biases", &layers::Conv2DLayer::set_biases, py::arg("biases"))
        .def("get_type", &layers::Conv2DLayer::get_type)
        .def("get_config", &layers::Conv2DLayer::get_config)
        .def("get_input_size", &layers::Conv2DLayer::get_input_size)
        .def("get_output_size", &layers::Conv2DLayer::get_output_size)
        .def("set_input_shape", &layers::Conv2DLayer::set_input_shape, py::arg("input_size"))
        .def("get_parameter_count", &layers::Conv2DLayer::get_parameter_count)
        .def("has_weights", &layers::Conv2DLayer::has_weights)
        .def("get_use_bias", &layers::Conv2DLayer::get_use_bias)
        .def("initialize_weights", &layers::Conv2DLayer::initialize_weights)
        .def("__repr__", &layers::Conv2DLayer::get_config);
}