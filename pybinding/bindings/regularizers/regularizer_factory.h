#ifndef BINDINGS_REGULARIZERS_REGULARIZER_FACTORY_H
#define BINDINGS_REGULARIZERS_REGULARIZER_FACTORY_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_regularizer_factory(py::module_& m);

#endif // BINDINGS_REGULARIZERS_REGULARIZER_FACTORY_H