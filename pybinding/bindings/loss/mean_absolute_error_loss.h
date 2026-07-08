#ifndef BINDINGS_LOSS_MEAN_ABSOLUTE_ERROR_LOSS_H
#define BINDINGS_LOSS_MEAN_ABSOLUTE_ERROR_LOSS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_mean_absolute_error_loss(py::module_& m);

#endif // BINDINGS_LOSS_MEAN_ABSOLUTE_ERROR_LOSS_H