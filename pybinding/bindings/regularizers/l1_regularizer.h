#ifndef BINDINGS_REGULARIZERS_L1_REGULARIZER_H
#define BINDINGS_REGULARIZERS_L1_REGULARIZER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_l1_regularizer(py::module_& m);

#endif // BINDINGS_REGULARIZERS_L1_REGULARIZER_H