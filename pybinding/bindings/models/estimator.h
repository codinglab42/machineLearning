#ifndef BINDINGS_MODELS_ESTIMATOR_H
#define BINDINGS_MODELS_ESTIMATOR_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_estimator(py::module_& m);

#endif // BINDINGS_MODELS_ESTIMATOR_H