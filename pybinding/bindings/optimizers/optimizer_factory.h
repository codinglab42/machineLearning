#ifndef BINDINGS_OPTIMIZERS_OPTIMIZER_FACTORY_H
#define BINDINGS_OPTIMIZERS_OPTIMIZER_FACTORY_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_optimizer_factory(py::module_& m);

#endif // BINDINGS_OPTIMIZERS_OPTIMIZER_FACTORY_H