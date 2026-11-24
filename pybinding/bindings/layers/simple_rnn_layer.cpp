#include "simple_rnn_layer.h"
#include "components/layers/simple_rnn_layer.h"
#include <pybind11/eigen.h>
#include "utils/binding_utils.h"

namespace py = pybind11;

void bind_simple_rnn_layer(py::module_& m) {
    py::class_<layers::SimpleRNNLayer, layers::Layer,
               std::shared_ptr<layers::SimpleRNNLayer>>(m, "SimpleRNNLayer")
        .def(py::init<int, int, const std::string&, bool>(),
             py::arg("units"), py::arg("input_size"),
             py::arg("activation") = "tanh", py::arg("use_bias") = true)
        .def(py::init<>())
        .def("forward", &recurrent_forward<layers::SimpleRNNLayer>,
             py::arg("input"), py::arg("training") = false)
        .def("backward", &recurrent_backward<layers::SimpleRNNLayer>,
             py::arg("gradient"))
        .def("set_return_sequences", [](layers::SimpleRNNLayer& layer, bool value) { layer.set_return_sequences(value); })
        .def("get_return_sequences", [](const layers::SimpleRNNLayer& layer) { return layer.get_return_sequences(); })
        .def("get_weights", &layers::SimpleRNNLayer::get_weights)
        .def("set_weights", [](layers::SimpleRNNLayer& layer, const PyArrayDouble& weights) {
            layer.set_weights(Eigen::MatrixXd(array_to_matrix(weights, "weights")));
        }, py::arg("weights"))
        .def("get_biases", &layers::SimpleRNNLayer::get_biases)
        .def("set_biases", [](layers::SimpleRNNLayer& layer, const PyArrayDouble& biases) {
            layer.set_biases(Eigen::VectorXd(array_to_vector(biases, "biases")));
        }, py::arg("biases"))
        .def("reset_state", &layers::SimpleRNNLayer::reset_state)
        .def("get_hidden_state", &layers::SimpleRNNLayer::get_hidden_state)
        .def("get_type", &layers::SimpleRNNLayer::get_type)
        .def("get_config", &layers::SimpleRNNLayer::get_config)
        .def("get_input_size", &layers::SimpleRNNLayer::get_input_size)
        .def("get_output_size", &layers::SimpleRNNLayer::get_output_size)
        .def("set_input_shape", &layers::SimpleRNNLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::SimpleRNNLayer::get_config);
}