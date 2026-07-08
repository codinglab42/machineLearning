#ifndef BINDINGS_OPTIMIZERS_ADAM_OPTIMIZER_H
#define BINDINGS_OPTIMIZERS_ADAM_OPTIMIZER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_adam_optimizer(py::module_& m);

#endif // BINDINGS_OPTIMIZERS_ADAM_OPTIMIZER_H