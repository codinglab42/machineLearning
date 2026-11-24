#include "gru_layer.h"
#include "components/layers/gru_layer.h"
#include <pybind11/eigen.h>
#include "utils/binding_utils.h"

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
        .def("forward", &recurrent_forward<layers::GRULayer>,
             py::arg("input"), py::arg("training") = false)
        .def("backward", &recurrent_backward<layers::GRULayer>,
             py::arg("gradient"))
        .def("set_return_sequences", [](layers::GRULayer& layer, bool value) { layer.set_return_sequences(value); })
        .def("get_return_sequences", [](const layers::GRULayer& layer) { return layer.get_return_sequences(); })
        .def("get_weights", &layers::GRULayer::get_weights)
        .def("set_weights", [](layers::GRULayer& layer, const PyArrayDouble& weights) {
            layer.set_weights(Eigen::MatrixXd(array_to_matrix(weights, "weights")));
        }, py::arg("weights"))
        .def("get_biases", &layers::GRULayer::get_biases)
        .def("set_biases", [](layers::GRULayer& layer, const PyArrayDouble& biases) {
            layer.set_biases(Eigen::VectorXd(array_to_vector(biases, "biases")));
        }, py::arg("biases"))
        .def("reset_state", &layers::GRULayer::reset_state)
        .def("get_hidden_state", &layers::GRULayer::get_hidden_state)
        .def("get_type", &layers::GRULayer::get_type)
        .def("get_config", &layers::GRULayer::get_config)
        .def("get_input_size", &layers::GRULayer::get_input_size)
        .def("get_output_size", &layers::GRULayer::get_output_size)
        .def("set_input_shape", &layers::GRULayer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::GRULayer::get_config);
}