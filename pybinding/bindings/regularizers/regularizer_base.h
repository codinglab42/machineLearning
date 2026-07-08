#ifndef BINDINGS_REGULARIZERS_REGULARIZER_BASE_H
#define BINDINGS_REGULARIZERS_REGULARIZER_BASE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_regularizer_base(py::module_& m);

#endif // BINDINGS_REGULARIZERS_REGULARIZER_BASE_H