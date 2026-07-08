#ifndef BINDINGS_CORE_MODULE_H
#define BINDINGS_CORE_MODULE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_module(py::module_& m);

#endif // BINDINGS_CORE_MODULE_H