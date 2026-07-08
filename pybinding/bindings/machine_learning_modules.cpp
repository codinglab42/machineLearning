#include <pybind11/pybind11.h>

// Core
#include "bindings/core/module.h"
#include "bindings/core/exceptions.h"
#include "bindings/core/enums.h"

// Models
#include "bindings/models/estimator.h"
#include "bindings/models/linear_regression.h"
#include "bindings/models/logistic_regression.h"
#include "bindings/models/neural_network.h"

// Layers
#include "bindings/layers/dense_layer.h"
#include "bindings/layers/conv2d_layer.h"
#include "bindings/layers/pooling_layer.h"
#include "bindings/layers/flatten_layer.h"
#include "bindings/layers/dropout_layer.h"
#include "bindings/layers/batch_norm_layer.h"
#include "bindings/layers/simple_rnn_layer.h"
#include "bindings/layers/lstm_layer.h"
#include "bindings/layers/gru_layer.h"
#include "bindings/layers/layer_factory.h"

// Loss
#include "bindings/loss/loss_functions.h"
#include "bindings/loss/loss_factory.h"

// Optimizers
#include "bindings/optimizers/optimizers.h"

// Regularizers
#include "bindings/regularizers/regularizers.h"

// Utils
#include "bindings/utils/scalers.h"
#include "bindings/utils/math_utils.h"

// Registra tutti i componenti
#include "components/layers/layer_factory.h"
#include "components/loss/loss_factory.h"

namespace py = pybind11;

PYBIND11_MODULE(machine_learning_module, m) {
    // ========================================================================
    // 1. CORE
    // ========================================================================
    bind_module(m);
    bind_exceptions(m);
    bind_enums(m);
    
    // ========================================================================
    // 2. MODELS
    // ========================================================================
    bind_estimator(m);
    bind_linear_regression(m);
    bind_logistic_regression(m);
    bind_neural_network(m);
    
    // ========================================================================
    // 3. LAYERS
    // ========================================================================
    bind_dense_layer(m);
    bind_conv2d_layer(m);
    bind_pooling_layer(m);
    bind_flatten_layer(m);
    bind_dropout_layer(m);
    bind_batch_norm_layer(m);
    bind_simple_rnn_layer(m);
    bind_lstm_layer(m);
    bind_gru_layer(m);
    bind_layer_factory(m);
    
    // ========================================================================
    // 4. LOSS
    // ========================================================================
    bind_loss_functions(m);
    bind_loss_factory(m);
    
    // ========================================================================
    // 5. OPTIMIZERS
    // ========================================================================
    bind_optimizers(m);
    
    // ========================================================================
    // 6. REGULARIZERS
    // ========================================================================
    bind_regularizers(m);
    
    // ========================================================================
    // 7. UTILS
    // ========================================================================
    bind_scalers(m);
    bind_math_utils(m);
    
    // ========================================================================
    // 8. INIZIALIZZAZIONE
    // ========================================================================
    layers::LayerFactory::register_all_layers();
    loss::LossFactory::register_all_losses();
}