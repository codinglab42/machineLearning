#include "loss_factory.h"
#include "components/loss/loss_factory.h"
#include "components/loss/loss.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_loss_factory(py::module_& m) {
    py::class_<loss::LossFactory>(m, "LossFactory")
        .def_static("create",
                    [](const std::string& name) -> std::shared_ptr<loss::Loss> {
                        return loss::LossFactory::create(name);
                    },
                    py::arg("name"),
                    "Create a loss function by name")
        .def_static("register_loss",
                    [](const std::string& name, py::function creator) {
                        // Nota: questo è un wrapper per registrare loss da Python
                        // L'implementazione dipende da come vuoi gestire la registrazione dinamica
                        // Per ora, registra solo le loss C++ esistenti
                        loss::LossFactory::register_all_losses();
                    },
                    py::arg("name"), py::arg("creator"))
        .def_static("register_all_losses",
                    &loss::LossFactory::register_all_losses,
                    "Register all built-in loss functions")
        .def_static("list_losses",
                    []() -> std::vector<std::string> {
                        // Nota: questo richiede un metodo per elencare le loss registrate
                        // Per ora restituisce una lista statica
                        return {"mse", "mae", "binary_crossentropy",
                                "categorical_crossentropy", "huber"};
                    },
                    "List all available loss function names");
}