#include "minmax_scaler.h"
#include "utils/minmax_scaler.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_minmax_scaler(py::module_& m) {
    py::class_<ml::MinMaxScaler, ml::Scaler,
               std::shared_ptr<ml::MinMaxScaler>>(m, "MinMaxScaler")
        .def(py::init<>(), "Create MinMaxScaler with default range [0, 1]")
        .def(py::init<double, double>(),
             py::arg("feature_range_min") = 0.0,
             py::arg("feature_range_max") = 1.0,
             "Create MinMaxScaler with custom range")
        .def("fit",
             &ml::MinMaxScaler::fit,
             py::arg("X"),
             "Fit MinMaxScaler to data (min=0, max=1)")
        .def("transform",
             &ml::MinMaxScaler::transform,
             py::arg("X"),
             "Transform data using fitted parameters")
        .def("fit_transform",
             &ml::MinMaxScaler::fit_transform,
             py::arg("X"),
             "Fit and transform data in one step")
        .def("inverse_transform",
             &ml::MinMaxScaler::inverse_transform,
             py::arg("X_scaled"),
             "Transform back to original scale")
        .def("get_min",
             &ml::MinMaxScaler::get_min,
             "Get min values")
        .def("get_max",
             &ml::MinMaxScaler::get_max,
             "Get max values")
        .def("set_params",
             &ml::MinMaxScaler::set_params,
             py::arg("min"), py::arg("max"),
             "Set scaler parameters manually")
        .def("get_type", &ml::MinMaxScaler::get_type)
        .def("__repr__", [](const ml::MinMaxScaler& scaler) {
            auto min_vals = scaler.get_min();
            auto max_vals = scaler.get_max();
            return "MinMaxScaler(min_size=" + std::to_string(min_vals.size()) +
                   ", max_size=" + std::to_string(max_vals.size()) + ")";
        });
}