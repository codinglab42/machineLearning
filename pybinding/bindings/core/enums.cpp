#include "enums.h"
#include "components/optimizers/optimizer.h"
#include "components/regularizers/regularizer.h"
#include "components/layers/layer.h"
#include "components/layers/pooling_layer.h"
#include "models/linear_regression.h"

namespace py = pybind11;

void bind_enums(py::module_& m) {
    // ========================================================================
    // LinearRegression::Solver
    // ========================================================================
    py::enum_<models::LinearRegression::Solver>(m, "LinearSolver")
        .value("GRADIENT_DESCENT", models::LinearRegression::Solver::GRADIENT_DESCENT)
        .value("NORMAL_EQUATION", models::LinearRegression::Solver::NORMAL_EQUATION)
        .value("SVD", models::LinearRegression::Solver::SVD)
        .export_values();

    // ========================================================================
    // OptimizerType
    // ========================================================================
    py::enum_<models::OptimizerType>(m, "OptimizerType")
        .value("SGD", models::OptimizerType::SGD)
        .value("MOMENTUM", models::OptimizerType::MOMENTUM)
        .value("ADAM", models::OptimizerType::ADAM)
        .export_values();

    // ========================================================================
    // RegularizerType
    // ========================================================================
    py::enum_<models::RegularizerType>(m, "RegularizerType")
        .value("NONE", models::RegularizerType::NONE)
        .value("L1", models::RegularizerType::L1)
        .value("L2", models::RegularizerType::L2)
        .value("ELASTIC_NET", models::RegularizerType::ELASTIC_NET)
        .export_values();

    // ========================================================================
    // LayerType
    // ========================================================================
    py::enum_<layers::LayerType>(m, "LayerType")
        .value("DENSE", layers::LayerType::DENSE)
        .value("CONV2D", layers::LayerType::CONV2D)
        .value("MAX_POOLING", layers::LayerType::MAX_POOLING)
        .value("AVERAGE_POOLING", layers::LayerType::AVERAGE_POOLING)
        .value("FLATTEN", layers::LayerType::FLATTEN)
        .value("DROPOUT", layers::LayerType::DROPOUT)
        .value("BATCH_NORM", layers::LayerType::BATCH_NORM)
        .value("SIMPLE_RNN", layers::LayerType::SIMPLE_RNN)
        .value("LSTM", layers::LayerType::LSTM)
        .value("GRU", layers::LayerType::GRU)
        .export_values();

    // ========================================================================
    // PoolType (per PoolingLayer)
    // ========================================================================
    py::enum_<layers::PoolingLayer::PoolType>(m, "PoolType")
        .value("MAX", layers::PoolingLayer::MAX)
        .value("AVG", layers::PoolingLayer::AVG)
        .export_values();
}