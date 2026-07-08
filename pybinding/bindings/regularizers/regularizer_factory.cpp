#include "regularizer_factory.h"
#include "components/regularizers/regularizer_factory.h"
#include "components/regularizers/regularizer.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_regularizer_factory(py::module_& m) {
    py::class_<models::RegularizerFactory>(m, "RegularizerFactory")
        .def_static("create_by_type",
            [](models::RegularizerType type, double strength,
               const std::unordered_map<std::string, double>& params) {
                return models::RegularizerFactory::create(type, strength, params);
            },
            py::arg("type"),
            py::arg("strength") = 0.01,
            py::arg("params") = std::unordered_map<std::string, double>(),
            "Create regularizer by type enum")
        .def_static("create_by_name",
            [](const std::string& type_str, double strength,
               const std::unordered_map<std::string, double>& params) {
                return models::RegularizerFactory::create(type_str, strength, params);
            },
            py::arg("type_str"),
            py::arg("strength") = 0.01,
            py::arg("params") = std::unordered_map<std::string, double>(),
            "Create regularizer by name string")
        .def_static("string_to_type",
            &models::RegularizerFactory::string_to_type,
            py::arg("type_str"),
            "Convert string to RegularizerType enum")
        .def_static("type_to_string",
            &models::RegularizerFactory::type_to_string,
            py::arg("type"),
            "Convert RegularizerType enum to string")
        .def_static("list_regularizers",
            []() -> std::vector<std::string> {
                return {"none", "l1", "l2", "elastic_net"};
            },
            "List all available regularizer names");
}