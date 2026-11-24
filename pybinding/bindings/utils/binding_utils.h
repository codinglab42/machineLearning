#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <limits>
#include <vector>
#include "components/layers/recurrent_layer.h"

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

inline layers::SequenceTensor array_to_sequence(const PyArrayDouble& arr, const std::string& name) {
    const py::buffer_info buf = arr.request();
    if (buf.ndim < 1 || buf.ndim > 3) {
        throw std::invalid_argument(name + " must have 1, 2 or 3 dimensions");
    }
    for (const auto dimension : buf.shape) {
        if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(name + " has invalid dimensions");
        }
    }
    layers::SequenceTensor tensor;
    tensor.batch_size = static_cast<int>(buf.shape[0]);
    tensor.time_steps = buf.ndim == 3 ? static_cast<int>(buf.shape[1]) : 1;
    tensor.features = buf.ndim == 1 ? 1 : static_cast<int>(buf.shape[buf.ndim - 1]);
    tensor.is_sequence = buf.ndim == 3;
    const auto* data = static_cast<const double*>(buf.ptr);
    tensor.data.assign(data, data + buf.size);
    tensor.validate();
    return tensor;
}

inline py::array_t<double> sequence_to_array(const layers::SequenceTensor& tensor) {
    tensor.validate();
    std::vector<py::ssize_t> shape{tensor.batch_size};
    if (tensor.is_sequence) shape.push_back(tensor.time_steps);
    shape.push_back(tensor.features);
    py::array_t<double> result(shape);
    std::copy(tensor.data.begin(), tensor.data.end(), result.mutable_data());
    return result;
}

template<typename T>
py::object recurrent_forward(T& layer, const PyArrayDouble& input, bool training) {
    if (input.ndim() == 3) {
        return sequence_to_array(layer.forward_sequence(array_to_sequence(input, "input"), training));
    }
    return py::cast(layer.forward(Eigen::MatrixXd(array_to_matrix(input, "input")), training));
}

template<typename T>
py::object recurrent_backward(T& layer, const PyArrayDouble& gradient) {
    if (gradient.ndim() == 3 || layer.last_forward_was_sequence()) {
        return sequence_to_array(layer.backward_sequence(array_to_sequence(gradient, "gradient")));
    }
    return py::cast(layer.backward(Eigen::MatrixXd(array_to_matrix(gradient, "gradient"))));
}

#endif // BINDING_UTILS_H