#ifndef BINDINGS_LAYERS_DROPOUT_LAYER_H
#define BINDINGS_LAYERS_DROPOUT_LAYER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_dropout_layer(py::module_& m);

#endif // BINDINGS_LAYERS_DROPOUT_LAYER_H