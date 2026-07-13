#ifndef BINDINGS_UTILS_MATH_UTILS_H
#define BINDINGS_UTILS_MATH_UTILS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_math_utils(py::module_& m);

#endif // BINDINGS_UTILS_MATH_UTILS_H