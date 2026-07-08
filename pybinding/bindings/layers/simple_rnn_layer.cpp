#include "simple_rnn_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_simple_rnn_layer(py::module_& m) {
    py::class_<layers::SimpleRNNLayer, layers::Layer,
               std::shared_ptr<layers::SimpleRNNLayer>>(m, "SimpleRNNLayer")
        .def(py::init<int, int, const std::string&, bool>(),
             py::arg("units"), py::arg("input_size"),
             py::arg("activation") = "tanh", py::arg("use_bias") = true)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::SimpleRNNLayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::SimpleRNNLayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::SimpleRNNLayer::*)(const Eigen::MatrixXd&)>(
                 &layers::SimpleRNNLayer::backward),
             py::arg("gradient"))
        .def("get_weights", &layers::SimpleRNNLayer::get_weights)
        .def("set_weights", &layers::SimpleRNNLayer::set_weights, py::arg("weights"))
        .def("get_biases", &layers::SimpleRNNLayer::get_biases)
        .def("set_biases", &layers::SimpleRNNLayer::set_biases, py::arg("biases"))
        .def("reset_state", &layers::SimpleRNNLayer::reset_state)
        .def("get_hidden_state", &layers::SimpleRNNLayer::get_hidden_state)
        .def("get_type", &layers::SimpleRNNLayer::get_type)
        .def("get_config", &layers::SimpleRNNLayer::get_config)
        .def("get_input_size", &layers::SimpleRNNLayer::get_input_size)
        .def("get_output_size", &layers::SimpleRNNLayer::get_output_size)
        .def("set_input_shape", &layers::SimpleRNNLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::SimpleRNNLayer::get_config);
}