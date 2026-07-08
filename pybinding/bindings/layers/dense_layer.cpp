#include "dense_layer.h"
#include "components/layers/dense_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_dense_layer(py::module_& m) {
    py::class_<layers::DenseLayer, layers::Layer,
               std::shared_ptr<layers::DenseLayer>>(m, "DenseLayer")
        .def(py::init<int, const std::string&, bool>(),
             py::arg("units"), py::arg("activation") = "relu", py::arg("use_bias") = true)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::DenseLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::DenseLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::DenseLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::DenseLayer::backward),
             py::arg("gradient"))
        .def("get_weights", &layers::DenseLayer::get_weights)
        .def("set_weights", &layers::DenseLayer::set_weights, py::arg("weights"))
        .def("get_biases", &layers::DenseLayer::get_biases)
        .def("set_biases", &layers::DenseLayer::set_biases, py::arg("biases"))
        .def("get_type", &layers::DenseLayer::get_type)
        .def("get_config", &layers::DenseLayer::get_config)
        .def("get_input_size", &layers::DenseLayer::get_input_size)
        .def("get_output_size", &layers::DenseLayer::get_output_size)
        .def("set_input_shape", &layers::DenseLayer::set_input_shape, py::arg("input_size"))
        .def("get_parameter_count", &layers::DenseLayer::get_parameter_count)
        .def("has_weights", &layers::DenseLayer::has_weights)
        .def("get_use_bias", &layers::DenseLayer::get_use_bias)
        .def("initialize_weights", &layers::DenseLayer::initialize_weights)
        .def("__repr__", &layers::DenseLayer::get_config);
}