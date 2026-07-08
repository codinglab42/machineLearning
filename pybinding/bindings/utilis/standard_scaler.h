#ifndef BINDINGS_UTILS_STANDARD_SCALER_H
#define BINDINGS_UTILS_STANDARD_SCALER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_standard_scaler(py::module_& m);

#endif // BINDINGS_UTILS_STANDARD_SCALER_H