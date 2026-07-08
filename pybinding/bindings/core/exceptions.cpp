#include "exceptions.h"
#include "exceptions/ml_exception.h"

namespace py = pybind11;

void bind_exceptions(py::module_& m) {
    // Registra il translator per le eccezioni C++
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const ml_exception::MLException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const ml_exception::NotFittedException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const ml_exception::DimensionMismatchException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const ml_exception::FeatureMismatchException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const ml_exception::InvalidParameterException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const ml_exception::EmptyDatasetException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const ml_exception::FileNotFoundException& e) {
            PyErr_SetString(PyExc_FileNotFoundError, e.what());
        } catch (const ml_exception::IOException& e) {
            PyErr_SetString(PyExc_IOError, e.what());
        } catch (const ml_exception::SerializationException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const ml_exception::DeserializationException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const ml_exception::ValidationException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const ml_exception::NotConvergedException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const ml_exception::MathException& e) {
            PyErr_SetString(PyExc_ArithmeticError, e.what());
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_Exception, e.what());
        } catch (...) {
            PyErr_SetString(PyExc_Exception, "Unknown C++ exception");
        }
    });
}