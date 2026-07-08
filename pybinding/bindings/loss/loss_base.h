#ifndef BINDINGS_LOSS_LOSS_BASE_H
#define BINDINGS_LOSS_LOSS_BASE_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_loss_base(py::module_& m);

#endif // BINDINGS_LOSS_LOSS_BASE_H