# ============================================================================
# TEST C++ (Con Google Test)
# ============================================================================
message(STATUS "🧪 Configuring C++ tests...")

# Nota: enable_testing() è già nel root, ma non fa male lasciarlo
enable_testing()

# Funzione helper per evitare ripetizioni (opzionale ma consigliata)
function(add_ml_test TEST_NAME SOURCES)
    add_executable(${TEST_NAME} ${SOURCES})
    target_link_libraries(${TEST_NAME} 
        PRIVATE 
            ML::Library        # La tua libreria (usando l'alias)
            GTest::gtest_main  # Include il main() di GTest automaticamente
            GTest::gtest
            GTest::gmock
            GTest::gmock_main
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Definizione dei Test ---

# Test per la cache dei layers
add_ml_test(test_dense_cache unit/cache/test_dense_cache.cpp)
add_ml_test(test_flatten_cache unit/cache/test_flatten_cache.cpp)
add_ml_test(test_dropout_cache unit/cache/test_dropout_cache.cpp)
add_ml_test(test_pooling_cache unit/cache/test_pooling_cache.cpp)
add_ml_test(test_conv_cache unit/cache/test_conv_cache.cpp)
add_ml_test(test_batchnorm_cache unit/cache/test_batchnorm_cache.cpp)
add_ml_test(test_rnn_cache unit/cache/test_rnn_cache.cpp)
add_ml_test(test_simple_rnn_cache unit/cache/test_simple_rnn_cache.cpp)
add_ml_test(test_lstm_cache unit/cache/test_lstm_cache.cpp)

# Test Atomici sui Layer (Gradient Checking, Forward/Backward)
add_ml_test(test_batch_norm_layer_deep unit/layers/test_batch_norm_layer_deep.cpp)
add_ml_test(test_dense_layer_deep unit/layers/test_dense_layer_deep.cpp)
add_ml_test(test_conv2d_layer_deep unit/layers/test_conv2d_layer_deep.cpp)
add_ml_test(test_dropout_layer_deep unit/layers/test_dropout_layer_deep.cpp)
add_ml_test(test_flatten_layer_deep unit/layers/test_flatten_layer_deep.cpp)
add_ml_test(test_pooling_layer_deep unit/layers/test_pooling_layer_deep.cpp)
add_ml_test(test_simple_rnn_layer_deep unit/layers/test_simple_rnn_layer_deep.cpp)
add_ml_test(test_gru_layer_deep unit/layers/test_gru_layer_deep.cpp)
add_ml_test(test_lstm_layer_deep unit/layers/test_lstm_layer_deep.cpp)
add_ml_test(test_gradient_dense_layer_deep unit/layers/test_gradient_dense_layer_deep.cpp)
add_ml_test(test_gradient_conv2d_layer_gemini unit/layers/test_gradient_conv2d_layer_gemini.cpp)

# Test sugli optimizers
add_ml_test(test_optimizers_deep unit/optimizers/test_optimizers_deep.cpp)

# Test sui regularizers
add_ml_test(test_regularizers_deep unit/regularizers/test_regularizers_deep.cpp)


# Test per Math Utils
add_ml_test(test_math_utils_deep unit/utils/test_math_utils_deep.cpp)

# Test per i Models
add_ml_test(test_linear_regression_model unit/models/test_linear_regression_model.cpp)
add_ml_test(test_logistic_regression_model unit/models/test_logistic_regression_model.cpp)
add_ml_test(test_neural_network_model unit/models/test_neural_network_model.cpp)
add_ml_test(test_neural_network_serialization unit/models/test_neural_network_serialization.cpp)

# Test per le Loss
add_ml_test(test_loss unit/loss/test_loss.cpp)
add_ml_test(test_binary_cross_entropy_loss unit/loss/test_binary_cross_entropy_loss.cpp)
add_ml_test(test_categorical_cross_entropy_loss unit/loss/test_categorical_cross_entropy_loss.cpp)
add_ml_test(test_mean_squared_error_loss unit/loss/test_mean_squared_error_loss.cpp)
add_ml_test(test_mean_absolute_error_loss unit/loss/test_mean_absolute_error_loss.cpp)
add_ml_test(test_huber_loss unit/loss/test_huber_loss.cpp)
add_ml_test(test_loss_factory_loss unit/loss/test_loss_factory_loss.cpp)
add_ml_test(test_loss_function unit/loss/test_loss_function.cpp)

# Test per la serializzzzazione
add_ml_test(test_dense_layer_serialization unit/serialization/test_dense_layer_serialization.cpp)
add_ml_test(test_conv2d_layer_serialization unit/serialization/test_conv2d_layer_serialization.cpp)
add_ml_test(test_pooling_layer_serialization unit/serialization/test_pooling_layer_serialization.cpp)
add_ml_test(test_batch_norm_layer_serialization unit/serialization/test_batch_norm_layer_serialization.cpp)
add_ml_test(test_simple_rnn_layer_serialization unit/serialization/test_simple_rnn_layer_serialization.cpp)
add_ml_test(test_lstm_layer_serialization unit/serialization/test_lstm_layer_serialization.cpp)
add_ml_test(test_gru_layer_serialization unit/serialization/test_gru_layer_serialization.cpp)
add_ml_test(test_dropout_layer_serialization unit/serialization/test_dropout_layer_serialization.cpp)
add_ml_test(test_flatten_layer_serialization unit/serialization/test_flatten_layer_serialization.cpp)

# Test integration
add_ml_test(test_linear_regression_integration integration/test_linear_regression_integration.cpp)
add_ml_test(test_logistic_regression_integration integration/test_logistic_regression_integration.cpp)
add_ml_test(test_neural_network_integration integration/test_neural_network_integration.cpp)
add_ml_test(test_real_datasets_integration integration/test_real_datasets_integration.cpp)
add_ml_test(test_loss_optimizer_integration integration/test_loss_optimizer_integration.cpp)
add_ml_test(test_loss_regularizer_integration integration/test_loss_regularizer_integration.cpp)
add_ml_test(test_neural_network_regularizer_integration integration/test_neural_network_regularizer_integration.cpp)
add_ml_test(test_layer_optimizer_integration integration/test_layer_optimizer_integration.cpp)
add_ml_test(test_end_to_end_integration integration/test_end_to_end_integration.cpp)
add_ml_test(test_convergence_integration integration/test_convergence_integration.cpp)
add_ml_test(test_convergence_xor integration/test_convergence_xor.cpp)
add_ml_test(test_neural_network_minimal_and integration/test_neural_network_minimal_and.cpp)

message(STATUS "✅ C++ tests configured")
