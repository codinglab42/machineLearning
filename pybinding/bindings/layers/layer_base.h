#ifndef BINDINGS_LAYERS_LAYER_BASE_H
#define BINDINGS_LAYERS_LAYER_BASE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_layer_base(py::module_& m);

#endif // BINDINGS_LAYERS_LAYER_BASE_H