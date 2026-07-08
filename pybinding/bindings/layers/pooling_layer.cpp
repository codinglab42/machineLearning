#include "pooling_layer.h"
#include "components/layers/pooling_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_pooling_layer(py::module_& m) {
    py::class_<layers::PoolingLayer, layers::Layer,
               std::shared_ptr<layers::PoolingLayer>>(m, "PoolingLayer")
        .def(py::init<int, int, layers::PoolingLayer::PoolType, int>(),
             py::arg("pool_size") = 2, py::arg("stride") = 2,
             py::arg("type") = layers::PoolingLayer::MAX, py::arg("channels") = 1)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::PoolingLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::PoolingLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::PoolingLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::PoolingLayer::backward),
             py::arg("gradient"))
        .def("get_type", &layers::PoolingLayer::get_type)
        .def("get_config", &layers::PoolingLayer::get_config)
        .def("get_input_size", &layers::PoolingLayer::get_input_size)
        .def("get_output_size", &layers::PoolingLayer::get_output_size)
        .def("set_input_shape", &layers::PoolingLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::PoolingLayer::get_config);
}