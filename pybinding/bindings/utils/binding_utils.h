#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Tipo helper per accettare array NumPy float32/float64 e gestirne lo stride
using PyArrayDouble = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Mappa un py::array_t in una MatrixXd Eigen 2D
inline Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
array_to_matrix(const PyArrayDouble& arr, const std::string& arg_name) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument(arg_name + " deve essere una matrice 2D.");
    }
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        static_cast<const double*>(buf.ptr), buf.shape[0], buf.shape[1]
    );
}

// Mappa un py::array_t accettando sia forma 1D (m,) che 2D (m, 1) per i vettori target
inline Eigen::Map<const Eigen::VectorXd>
array_to_vector(const PyArrayDouble& arr, const std::string& arg_name) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument(arg_name + " deve essere un vettore 1D o una matrice colonna 2D.");
    }
    if (buf.ndim == 2 && buf.shape[1] != 1) {
        throw std::invalid_argument(arg_name + " in formato 2D deve avere esattamente una colonna.");
    }
    return Eigen::Map<const Eigen::VectorXd>(
        static_cast<const double*>(buf.ptr), buf.shape[0]
    );
}

#endif // BINDING_UTILS_H