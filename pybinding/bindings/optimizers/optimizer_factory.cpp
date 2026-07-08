#include "optimizer_factory.h"
#include "components/optimizers/optimizer_factory.h"
#include "components/optimizers/optimizer.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_optimizer_factory(py::module_& m) {
    py::class_<models::OptimizerFactory>(m, "OptimizerFactory")
        .def_static("create_by_type",
            [](models::OptimizerType type, double learning_rate,
               const std::unordered_map<std::string, double>& params) {
                return models::OptimizerFactory::create(type, learning_rate, params);
            },
            py::arg("type"),
            py::arg("learning_rate") = 0.01,
            py::arg("params") = std::unordered_map<std::string, double>(),
            "Create optimizer by type enum")
        .def_static("create_by_name",
            [](const std::string& type_str, double learning_rate,
               const std::unordered_map<std::string, double>& params) {
                return models::OptimizerFactory::create(type_str, learning_rate, params);
            },
            py::arg("type_str"),
            py::arg("learning_rate") = 0.01,
            py::arg("params") = std::unordered_map<std::string, double>(),
            "Create optimizer by name string")
        .def_static("string_to_type",
            &models::OptimizerFactory::string_to_type,
            py::arg("type_str"),
            "Convert string to OptimizerType enum")
        .def_static("type_to_string",
            &models::OptimizerFactory::type_to_string,
            py::arg("type"),
            "Convert OptimizerType enum to string")
        .def_static("list_optimizers",
            []() -> std::vector<std::string> {
                return {"sgd", "momentum", "adam"};
            },
            "List all available optimizer names");
}