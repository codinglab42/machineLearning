#include "layer_base.h"
#include "components/layers/layer.h"
#include <pybind11/eigen.h>

namespace py = pybind11;

// Helper per forward overloaded
template<typename T>
auto get_forward_with_training() {
    return static_cast<Eigen::MatrixXd (T::*)(const Eigen::MatrixXd&, bool)>(&T::forward);
}

template<typename T>
auto get_forward_simple() {
    return static_cast<Eigen::MatrixXd (T::*)(const Eigen::MatrixXd&)>(&T::forward);
}

template<typename T>
auto get_backward() {
    return static_cast<Eigen::MatrixXd (T::*)(const Eigen::MatrixXd&)>(&T::backward);
}

void bind_layer_base(py::module_& m) {
    py::class_<layers::Layer, std::shared_ptr<layers::Layer>>(m, "Layer")
        .def("forward", 
             [](layers::Layer& layer, const Eigen::MatrixXd& input, bool training) {
                 return layer.forward(input, training);
             },
             py::arg("input"), py::arg("training") = false)
        .def("backward", 
             [](layers::Layer& layer, const Eigen::MatrixXd& gradient) {
                 return layer.backward(gradient);
             },
             py::arg("gradient"))
        .def("has_weights", &layers::Layer::has_weights)
        .def("get_weights", &layers::Layer::get_weights)
        .def("set_weights", &layers::Layer::set_weights, py::arg("weights"))
        .def("get_weights_gradient", &layers::Layer::get_weights_gradient)
        .def("get_use_bias", &layers::Layer::get_use_bias)
        .def("get_biases", &layers::Layer::get_biases)
        .def("set_biases", &layers::Layer::set_biases, py::arg("biases"))
        .def("get_bias_gradient", &layers::Layer::get_bias_gradient)
        .def("get_parameter_count", &layers::Layer::get_parameter_count)
        .def("clear_cache", &layers::Layer::clear_cache)
        .def("get_layer_type", &layers::Layer::get_layer_type)
        .def("get_type", &layers::Layer::get_type)
        .def("get_config", &layers::Layer::get_config)
        .def("get_input_size", &layers::Layer::get_input_size)
        .def("get_output_size", &layers::Layer::get_output_size)
        .def("set_input_shape", &layers::Layer::set_input_shape, py::arg("input_size"))
        .def("__repr__", &layers::Layer::get_config);
}