#include "gru_layer.h"
#include "components/layers/gru_layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

void bind_gru_layer(py::module_& m) {
    py::class_<layers::GRULayer, layers::Layer,
               std::shared_ptr<layers::GRULayer>>(m, "GRULayer")
        .def(py::init<int, int, const std::string&, const std::string&, bool>(),
             py::arg("units"), py::arg("input_size"),
             py::arg("activation") = "tanh",
             py::arg("recurrent_activation") = "sigmoid",
             py::arg("use_bias") = true)
        .def(py::init<>())
        .def("forward", 
             static_cast<Eigen::MatrixXd (layers::GRULayer::*)(const Eigen::MatrixXd&, bool)>(
                 &layers::GRULayer::forward),
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             static_cast<Eigen::MatrixXd (layers::GRULayer::*)(const Eigen::MatrixXd&)>(
                 &layers::GRULayer::backward),
             py::arg("gradient"))
        .def("get_weights", &layers::GRULayer::get_weights)
        .def("set_weights", &layers::GRULayer::set_weights, py::arg("weights"))
        .def("get_biases", &layers::GRULayer::get_biases)
        .def("set_biases", &layers::GRULayer::set_biases, py::arg("biases"))
        .def("reset_state", &layers::GRULayer::reset_state)
        .def("get_hidden_state", &layers::GRULayer::get_hidden_state)
        .def("get_type", &layers::GRULayer::get_type)
        .def("get_config", &layers::GRULayer::get_config)
        .def("get_input_size", &layers::GRULayer::get_input_size)
        .def("get_output_size", &layers::GRULayer::get_output_size)
        .def("set_input_shape", &layers::GRULayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::GRULayer::get_config);
}