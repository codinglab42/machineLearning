#ifndef BINDINGS_CORE_ENUMS_H
#define BINDINGS_CORE_ENUMS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_enums(py::module_& m);

#endif // BINDINGS_CORE_ENUMS_H