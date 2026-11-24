#include "lstm_layer.h"
#include "components/layers/lstm_layer.h"
#include <pybind11/eigen.h>
#include "utils/binding_utils.h"

namespace py = pybind11;

void bind_lstm_layer(py::module_& m) {
    py::class_<layers::LSTMLayer, layers::Layer,
               std::shared_ptr<layers::LSTMLayer>>(m, "LSTMLayer")
        .def(py::init<int, int, const std::string&, const std::string&, bool>(),
             py::arg("units"), py::arg("input_size"),
             py::arg("activation") = "tanh",
             py::arg("recurrent_activation") = "sigmoid",
             py::arg("use_bias") = true)
        .def(py::init<>())
        .def("forward", &recurrent_forward<layers::LSTMLayer>,
             py::arg("input"), py::arg("training") = false)
        .def("backward", &recurrent_backward<layers::LSTMLayer>,
             py::arg("gradient"))
        .def("set_return_sequences", [](layers::LSTMLayer& layer, bool value) { layer.set_return_sequences(value); })
        .def("get_return_sequences", [](const layers::LSTMLayer& layer) { return layer.get_return_sequences(); })
        .def("get_weights", &layers::LSTMLayer::get_weights)
        .def("set_weights", [](layers::LSTMLayer& layer, const PyArrayDouble& weights) {
            layer.set_weights(Eigen::MatrixXd(array_to_matrix(weights, "weights")));
        }, py::arg("weights"))
        .def("get_biases", &layers::LSTMLayer::get_biases)
        .def("set_biases", [](layers::LSTMLayer& layer, const PyArrayDouble& biases) {
            layer.set_biases(Eigen::VectorXd(array_to_vector(biases, "biases")));
        }, py::arg("biases"))
        .def("reset_state", &layers::LSTMLayer::reset_state)
        .def("get_hidden_state", &layers::LSTMLayer::get_hidden_state)
        .def("get_cell_state", &layers::LSTMLayer::get_cell_state)
        .def("get_type", &layers::LSTMLayer::get_type)
        .def("get_config", &layers::LSTMLayer::get_config)
        .def("get_input_size", &layers::LSTMLayer::get_input_size)
        .def("get_output_size", &layers::LSTMLayer::get_output_size)
        .def("set_input_shape", &layers::LSTMLayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::LSTMLayer::get_config);
}