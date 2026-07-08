#include "module.h"

namespace py = pybind11;

void bind_module(py::module_& m) {
    m.doc() = R"pbdoc(
        Machine Learning Library Python Bindings
        =========================================
        
        A comprehensive C++ ML library with Python bindings.
        
        Features:
        - Linear Regression with multiple solvers
        - Logistic Regression with regularization  
        - Neural Networks (Dense, Conv2D, RNN, LSTM, GRU)
        - Loss Functions (MSE, MAE, BCE, CCE, Huber)
        - Optimizers (SGD, Momentum, Adam)
        - Regularizers (L1, L2, Elastic Net)
        - Cross-validation support
        - Model serialization
    )pbdoc";
    
    m.attr("__version__") = "3.0.0";
    m.attr("__author__") = "Maurizio Penna";
    
    m.def("test_library", []() {
        return "Machine Learning library v3.0.0 is working correctly!";
    });
    
    m.def("check_version", []() {
        return std::string("v3.0.0 - Neural Networks, RNN (LSTM, GRU), CNN, Exceptions, Serialization");
    });
}