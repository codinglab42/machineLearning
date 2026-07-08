#ifndef BINDINGS_MODELS_LINEAR_REGRESSION_H
#define BINDINGS_MODELS_LINEAR_REGRESSION_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_linear_regression(py::module_& m);

#endif // BINDINGS_MODELS_LINEAR_REGRESSION_H