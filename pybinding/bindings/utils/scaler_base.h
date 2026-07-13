#ifndef BINDINGS_UTILS_SCALER_BASE_H
#define BINDINGS_UTILS_SCALER_BASE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_scaler_base(py::module_& m);
void bind_standard_scaler(py::module_& m);
void bind_minmax_scaler(py::module_& m); 

#endif // BINDINGS_UTILS_SCALER_BASE_H