#include "standard_scaler.h"
#include "utils/standard_scaler.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_standard_scaler(py::module_& m) {
    py::class_<ml::StandardScaler, ml::Scaler,
               std::shared_ptr<ml::StandardScaler>>(m, "StandardScaler")
        .def(py::init<>(), "Create StandardScaler with default epsilon=1e-8")
        .def(py::init<double>(), py::arg("epsilon"), "Create StandardScaler with custom epsilon")
        .def("fit",
             &ml::StandardScaler::fit,
             py::arg("X"),
             "Fit StandardScaler to data (mean=0, std=1)")
        .def("transform",
             &ml::StandardScaler::transform,
             py::arg("X"),
             "Transform data using fitted parameters")
        .def("fit_transform",
             &ml::StandardScaler::fit_transform,
             py::arg("X"),
             "Fit and transform data in one step")
        .def("inverse_transform",
             &ml::StandardScaler::inverse_transform,
             py::arg("X_scaled"),
             "Transform back to original scale")
        .def("get_mean",
             &ml::StandardScaler::get_mean,
             "Get mean values")
        .def("get_std",
             &ml::StandardScaler::get_std,
             "Get standard deviation values")
        .def("set_params",
             &ml::StandardScaler::set_params,
             py::arg("mean"), py::arg("std"),
             "Set scaler parameters manually")
        .def("get_type", &ml::StandardScaler::get_type)
        .def("__repr__", [](const ml::StandardScaler& scaler) {
            auto mean = scaler.get_mean();
            auto std = scaler.get_std();
            return "StandardScaler(mean_size=" + std::to_string(mean.size()) +
                   ", std_size=" + std::to_string(std.size()) + ")";
        });
}