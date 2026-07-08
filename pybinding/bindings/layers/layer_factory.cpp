#include "layer_factory.h"
#include "components/layers/layer_factory.h"
#include "components/layers/layer.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_layer_factory(py::module_& m) {
    py::class_<layers::LayerFactory>(m, "LayerFactory")
        .def_static("create", 
            [](layers::LayerType type) -> std::shared_ptr<layers::Layer> {
                return layers::LayerFactory::create(type);
            },
            py::arg("type"))
        .def_static("create_by_name", 
            [](const std::string& name) -> std::shared_ptr<layers::Layer> {
                return layers::LayerFactory::create(name);
            },
            py::arg("name"))
        .def_static("get_name", &layers::LayerFactory::get_name, py::arg("type"))
        .def_static("serialize_type", &layers::LayerFactory::serialize_type,
                    py::arg("out"), py::arg("type"))
        .def_static("deserialize_type", &layers::LayerFactory::deserialize_type,
                    py::arg("in"))
        .def_static("register_all_layers", &layers::LayerFactory::register_all_layers);
}