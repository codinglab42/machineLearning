#ifndef BINDINGS_REGULARIZERS_ELASTIC_NET_REGULARIZER_H
#define BINDINGS_REGULARIZERS_ELASTIC_NET_REGULARIZER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_elastic_net_regularizer(py::module_& m);

#endif // BINDINGS_REGULARIZERS_ELASTIC_NET_REGULARIZER_H