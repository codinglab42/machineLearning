#ifndef BINDINGS_LOSS_BINARY_CROSS_ENTROPY_LOSS_H
#define BINDINGS_LOSS_BINARY_CROSS_ENTROPY_LOSS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_binary_cross_entropy_loss(py::module_& m);

#endif // BINDINGS_LOSS_BINARY_CROSS_ENTROPY_LOSS_H