#ifndef BINDINGS_CORE_EXCEPTIONS_H
#define BINDINGS_CORE_EXCEPTIONS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_exceptions(py::module_& m);

#endif // BINDINGS_CORE_EXCEPTIONS_H