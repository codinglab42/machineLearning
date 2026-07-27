#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// Core
#include "core/module.h"
#include "core/exceptions.h"
#include "core/enums.h"

// Models
#include "models/estimator.h"
#include "models/linear_regression.h"
#include "models/logistic_regression.h"
#include "models/neural_network.h"

// Layers
#include "layers/layer_base.h"
#include "layers/dense_layer.h"
#include "layers/conv2d_layer.h"
#include "layers/pooling_layer.h"
#include "layers/flatten_layer.h"
#include "layers/dropout_layer.h"
#include "layers/batch_norm_layer.h"
#include "layers/simple_rnn_layer.h"
#include "layers/lstm_layer.h"
#include "layers/gru_layer.h"
#include "layers/layer_factory.h"

// Loss - CORRETTO!
#include "loss/loss_base.h"
#include "loss/mean_squared_error_loss.h"
#include "loss/mean_absolute_error_loss.h"
#include "loss/binary_cross_entropy_loss.h"
#include "loss/categorical_cross_entropy_loss.h"
#include "loss/huber_loss.h"
#include "loss/loss_factory.h"

// Optimizers
#include "optimizers/optimizer_base.h"
#include "optimizers/sgd_optimizer.h"
#include "optimizers/momentum_optimizer.h"
#include "optimizers/adam_optimizer.h"
#include "optimizers/optimizer_factory.h"

// Regularizers
#include "regularizers/regularizer_base.h"
#include "regularizers/l1_regularizer.h"
#include "regularizers/l2_regularizer.h"
#include "regularizers/elastic_net_regularizer.h"
#include "regularizers/regularizer_factory.h"

// Utils
#include "utils/scaler_base.h"
#include "utils/standard_scaler.h"
#include "utils/minmax_scaler.h"
#include "utils/math_utils.h"

#include "components/layers/layer_factory.h"
#include "components/loss/loss_factory.h"

namespace py = pybind11;

PYBIND11_MODULE(machine_learning_modules, m) {
    // ========================================================================
    // CORE
    // ========================================================================
    bind_module(m);
    bind_exceptions(m);
    bind_enums(m);
    
    // ========================================================================
    // MODELS
    // ========================================================================
    bind_estimator(m);
    bind_linear_regression(m);
    bind_logistic_regression(m);
    bind_neural_network(m);
    
    // ========================================================================
    // LAYERS
    // ========================================================================
    bind_layer_base(m);
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
    // LOSS - CORRETTO!
    // ========================================================================
    bind_loss_base(m);
    bind_mean_squared_error_loss(m);
    bind_mean_absolute_error_loss(m);
    bind_binary_cross_entropy_loss(m);
    bind_categorical_cross_entropy_loss(m);
    bind_huber_loss(m);
    bind_loss_factory(m);
    
    // ========================================================================
    // OPTIMIZERS
    // ========================================================================
    bind_optimizer_base(m);
    bind_sgd_optimizer(m);
    bind_momentum_optimizer(m);
    bind_adam_optimizer(m);
    bind_optimizer_factory(m);
    
    // ========================================================================
    // REGULARIZERS
    // ========================================================================
    bind_regularizer_base(m);
    bind_l1_regularizer(m);
    bind_l2_regularizer(m);
    bind_elastic_net_regularizer(m);
    bind_regularizer_factory(m);
    
    // ========================================================================
    // UTILS
    // ========================================================================
    bind_scaler_base(m);
    bind_standard_scaler(m);
    bind_minmax_scaler(m);
    bind_math_utils(m);
    
    // ========================================================================
    // INIZIALIZZAZIONE
    // ========================================================================
    layers::LayerFactory::register_all_layers();
    loss::LossFactory::register_all_losses();
}