#include "scaler_base.h"
#include "utils/scaler.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_scaler_base(py::module_& m) {
    py::class_<ml::Scaler, std::shared_ptr<ml::Scaler>>(m, "Scaler")
        .def("fit",
             [](ml::Scaler& scaler, const std::vector<std::vector<double>>& X) {
                 scaler.fit(X);
             },
             py::arg("X"),
             "Fit scaler to data")
        .def("transform",
             [](const ml::Scaler& scaler, const std::vector<std::vector<double>>& X) {
                 return scaler.transform(X);
             },
             py::arg("X"),
             "Transform data using fitted parameters")
        .def("fit_transform",
             [](ml::Scaler& scaler, std::vector<std::vector<double>>& X) {
                 return scaler.fit_transform(X);
             },
             py::arg("X"),
             "Fit and transform data in one step")
        .def("inverse_transform",
             [](const ml::Scaler& scaler, const std::vector<std::vector<double>>& X_scaled) {
                 return scaler.inverse_transform(X_scaled);
             },
             py::arg("X_scaled"),
             "Transform back to original scale")
        .def("get_type", &ml::Scaler::get_type)
        .def("__repr__", [](const ml::Scaler& scaler) {
            return std::string("Scaler(") + scaler.get_type() + ")";
        });
}