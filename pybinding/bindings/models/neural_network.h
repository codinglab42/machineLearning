#ifndef BINDINGS_MODELS_NEURAL_NETWORK_H
#define BINDINGS_MODELS_NEURAL_NETWORK_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_neural_network(py::module_& m);

#endif // BINDINGS_MODELS_NEURAL_NETWORK_H