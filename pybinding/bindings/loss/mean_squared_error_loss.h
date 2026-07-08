#ifndef BINDINGS_LOSS_MEAN_SQUARED_ERROR_LOSS_H
#define BINDINGS_LOSS_MEAN_SQUARED_ERROR_LOSS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_mean_squared_error_loss(py::module_& m);

#endif // BINDINGS_LOSS_MEAN_SQUARED_ERROR_LOSS_H