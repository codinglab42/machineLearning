#include "batch_norm_layer.h"
#include "components/layers/batch_norm_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_batch_norm_layer(py::module_& m) {
    py::class_<layers::BatchNormLayer, layers::Layer,
               std::shared_ptr<layers::BatchNormLayer>>(m, "BatchNormLayer")
        .def(py::init<double, double>(), py::arg("epsilon") = 1e-5, py::arg("momentum") = 0.9)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::BatchNormLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::BatchNormLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::BatchNormLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::BatchNormLayer::backward),
             py::arg("gradient"))
        .def("get_weights", &layers::BatchNormLayer::get_weights)
        .def("set_weights", &layers::BatchNormLayer::set_weights, py::arg("weights"))
        .def("get_biases", &layers::BatchNormLayer::get_biases)
        .def("set_biases", &layers::BatchNormLayer::set_biases, py::arg("biases"))
        .def("get_type", &layers::BatchNormLayer::get_type)
        .def("get_config", &layers::BatchNormLayer::get_config)
        .def("get_input_size", &layers::BatchNormLayer::get_input_size)
        .def("get_output_size", &layers::BatchNormLayer::get_output_size)
        .def("set_input_shape", &layers::BatchNormLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::BatchNormLayer::get_config);
}