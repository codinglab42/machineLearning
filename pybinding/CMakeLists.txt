# ============================================================================
# PYTHON BINDINGS
# ============================================================================
message(STATUS "🐍 Configuring Python bindings...")

# ============================================================================
# SEARCH FOR PYTHON AND PYBIND11
# ============================================================================

# Trova Python (necessario prima di pybind11)
find_package(Python3 COMPONENTS Interpreter Development Development.Module REQUIRED)

# Trova pybind11 (dovrebbe essere già installato)
find_package(pybind11 CONFIG REQUIRED)

# ============================================================================
# COLLECT SOURCE FILES
# ============================================================================

set(BINDING_SOURCES
    bindings/machine_learning_module.cpp
    
    # Core
    bindings/core/enums.cpp
    bindings/core/exceptions.cpp
    bindings/core/module.cpp
    
    # Models
    bindings/models/estimator.cpp
    bindings/models/linear_regression.cpp
    bindings/models/logistic_regression.cpp
    bindings/models/neural_network.cpp
    
    # Layers
    bindings/layers/layer_base.cpp
    bindings/layers/dense_layer.cpp
    bindings/layers/conv2d_layer.cpp
    bindings/layers/pooling_layer.cpp
    bindings/layers/flatten_layer.cpp
    bindings/layers/dropout_layer.cpp
    bindings/layers/batch_norm_layer.cpp
    bindings/layers/simple_rnn_layer.cpp
    bindings/layers/lstm_layer.cpp
    bindings/layers/gru_layer.cpp
    bindings/layers/layer_factory.cpp
    
    # Loss
    bindings/loss/loss_base.cpp
    bindings/loss/mean_squared_error_loss.cpp
    bindings/loss/mean_absolute_error_loss.cpp
    bindings/loss/binary_cross_entropy_loss.cpp
    bindings/loss/categorical_cross_entropy_loss.cpp
    bindings/loss/huber_loss.cpp
    bindings/loss/loss_factory.cpp
    
    # Optimizers
    bindings/optimizers/optimizer_base.cpp
    bindings/optimizers/sgd_optimizer.cpp
    bindings/optimizers/momentum_optimizer.cpp
    bindings/optimizers/adam_optimizer.cpp
    bindings/optimizers/optimizer_factory.cpp
    
    # Regularizers
    bindings/regularizers/regularizer_base.cpp
    bindings/regularizers/l1_regularizer.cpp
    bindings/regularizers/l2_regularizer.cpp
    bindings/regularizers/elastic_net_regularizer.cpp
    bindings/regularizers/regularizer_factory.cpp
    
    # Utils
    bindings/utils/scaler_base.cpp
    bindings/utils/standard_scaler.cpp
    bindings/utils/minmax_scaler.cpp
    bindings/utils/math_utils.cpp
)

# ============================================================================
# CREATE PYTHON MODULE
# ============================================================================

pybind11_add_module(machine_learning_module ${BINDING_SOURCES})

# ============================================================================
# LINK LIBRARIES
# ============================================================================

# Link con la libreria principale ML
target_link_libraries(machine_learning_module PRIVATE 
    ml_library
    ${Python3_LIBRARIES}
)

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

target_include_directories(machine_learning_module PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Eigen3_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# ============================================================================
# COMPILE OPTIONS
# ============================================================================

# C++ standard (deve essere almeno C++17 per pybind11)
set_target_properties(machine_learning_module PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
    OUTPUT_NAME "machine_learning_module"
)

# ============================================================================
# MACOS SPECIFIC
# ============================================================================

if(APPLE)
    # Su macOS, il modulo deve essere un bundle
    set_target_properties(machine_learning_module PROPERTIES
        SUFFIX ".so"
        MACOSX_RPATH ON
        INSTALL_RPATH "@loader_path/../lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# ============================================================================
# INSTALLATION INTELLIGENTE PER PYENV
# ============================================================================

if(DEFINED ENV{PYENV_ROOT})
    # Determina site-packages di pyenv
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "
import site
if hasattr(site, 'getsitepackages'):
    print(site.getsitepackages()[0])
else:
    import sys
    print(sys.prefix + '/lib/python' + str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '/site-packages')
"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(PYTHON_SITE_PACKAGES)
        message(STATUS "📁 Installing to pyenv site-packages: ${PYTHON_SITE_PACKAGES}")
        install(TARGETS machine_learning_module DESTINATION ${PYTHON_SITE_PACKAGES})
        
        # Symlink per testing
        add_custom_command(TARGET machine_learning_module POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:machine_learning_module>
            ${CMAKE_BINARY_DIR}/machine_learning_module$<TARGET_FILE_NAME:machine_learning_module>
            COMMENT "Creating symlink for testing"
        )
    else()
        message(WARNING "⚠️  Could not determine pyenv site-packages")
        install(TARGETS machine_learning_module DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
    endif()
else()
    # Installazione normale
    install(TARGETS machine_learning_module DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
endif()

# ============================================================================
# TARGET PER TEST PYTHON MODULE
# ============================================================================

add_custom_target(test_python_module
    COMMAND ${Python3_EXECUTABLE} -c "
import sys
import os
import numpy as np

# Aggiungi directory build al path
build_dir = os.path.join('${CMAKE_BINARY_DIR}', 'pybinding')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
else:
    sys.path.insert(0, '${CMAKE_BINARY_DIR}')

try:
    import machine_learning_module as ml
    
    print('=' * 60)
    print('🧪 TESTING PYTHON BINDINGS')
    print('=' * 60)
    
    print('✅ Module imported successfully')
    print(f'   Version: {ml.__version__}')
    
    # ====================================================================
    # TEST 1: ENUMS
    # ====================================================================
    print('\\n📋 Testing Enums...')
    print(f'   OptimizerType.ADAM = {ml.OptimizerType.ADAM}')
    print(f'   RegularizerType.L2 = {ml.RegularizerType.L2}')
    print(f'   LayerType.DENSE = {ml.LayerType.DENSE}')
    print('   ✅ Enums OK')
    
    # ====================================================================
    # TEST 2: LINEAR REGRESSION
    # ====================================================================
    print('\\n📊 Testing LinearRegression...')
    X = np.random.randn(50, 3)
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 + np.random.randn(50) * 0.1
    
    lr = ml.LinearRegression(learning_rate=0.01, max_iter=100)
    lr.fit(X, y)
    pred = lr.predict(X)
    score = lr.score(X, y)
    print(f'   R² score: {score:.4f}')
    print('   ✅ LinearRegression OK')
    
    # ====================================================================
    # TEST 3: LOGISTIC REGRESSION
    # ====================================================================
    print('\\n📊 Testing LogisticRegression...')
    X2 = np.random.randn(100, 3)
    y2 = (X2[:, 0] + X2[:, 1] > 0).astype(float)
    
    log_reg = ml.LogisticRegression(learning_rate=0.1, max_iter=200)
    log_reg.fit(X2, y2)
    acc = log_reg.score(X2, y2)
    print(f'   Accuracy: {acc:.4f}')
    print('   ✅ LogisticRegression OK')
    
    # ====================================================================
    # TEST 4: NEURAL NETWORK
    # ====================================================================
    print('\\n🧠 Testing NeuralNetwork...')
    nn = ml.NeuralNetwork()
    nn.add_dense_layer(16, activation='relu')
    nn.add_dense_layer(8, activation='relu')
    nn.add_dense_layer(1, activation='sigmoid')
    nn.build(3, 1)
    nn.set_epochs(50)
    nn.set_verbose(False)
    
    X3 = np.random.randn(100, 3)
    y3 = (np.sum(X3[:, :2], axis=1) > 0).astype(float)
    nn.fit(X3, y3)
    score_nn = nn.score(X3, y3)
    print(f'   NN accuracy: {score_nn:.4f}')
    print('   ✅ NeuralNetwork OK')
    
    # ====================================================================
    # TEST 5: LAYERS
    # ====================================================================
    print('\\n🔧 Testing Layers...')
    dense = ml.DenseLayer(8, activation='relu')
    dense.set_input_shape(4)
    dense.initialize_weights()
    print(f'   DenseLayer created: {dense.get_type()}')
    print(f'   Config: {dense.get_config()}')
    print('   ✅ Layers OK')
    
    # ====================================================================
    # TEST 6: LOSS FUNCTIONS
    # ====================================================================
    print('\\n📉 Testing Loss Functions...')
    mse = ml.MeanSquaredErrorLoss()
    y_true = np.random.randn(10)
    y_pred = y_true + 0.1
    loss = mse.compute_vector(y_true, y_pred)
    print(f'   MSE loss: {loss:.4f}')
    print('   ✅ Loss Functions OK')
    
    # ====================================================================
    # TEST 7: OPTIMIZERS
    # ====================================================================
    print('\\n⚡ Testing Optimizers...')
    adam = ml.AdamOptimizer(learning_rate=0.001)
    w = np.random.randn(5, 5)
    g = np.random.randn(5, 5)
    adam.update_weights(w, g)
    print('   Adam update OK')
    print('   ✅ Optimizers OK')
    
    # ====================================================================
    # TEST 8: REGULARIZERS
    # ====================================================================
    print('\\n🔬 Testing Regularizers...')
    l2 = ml.L2Regularizer(strength=0.01)
    w_reg = np.random.randn(5, 5)
    loss_l2 = l2.compute_loss_matrix(w_reg)
    print(f'   L2 loss: {loss_l2:.4f}')
    print('   ✅ Regularizers OK')
    
    # ====================================================================
    # TEST 9: SCALERS
    # ====================================================================
    print('\\n📏 Testing Scalers...')
    scaler = ml.StandardScaler()
    X_scaled = [[1, 2], [3, 4], [5, 6]]
    scaler.fit(X_scaled)
    scaled = scaler.transform(X_scaled)
    print(f'   Scaled data shape: {len(scaled)}x{len(scaled[0])}')
    print('   ✅ Scalers OK')
    
    # ====================================================================
    # TEST 10: MATH UTILS
    # ====================================================================
    print('\\n🧮 Testing MathUtils...')
    sig = ml.MathUtils.sigmoid(0.0)
    print(f'   sigmoid(0) = {sig:.4f}')
    weights_he = ml.MathUtils.he_initialization(10, 5)
    print(f'   He weights shape: {weights_he.shape}')
    print('   ✅ MathUtils OK')
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print('\\n' + '=' * 60)
    print('✅ ALL TESTS PASSED!')
    print('=' * 60)
    print(f'\\n📦 Module: machine_learning_module')
    print(f'📌 Version: {ml.__version__}')
    print('🚀 Ready to use!')
    
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    DEPENDS machine_learning_module
    COMMENT "Testing Python module import and functionality"
    VERBATIM
)

message(STATUS "✅ Python bindings configured")
message(STATUS "  Module: machine_learning_module")
message(STATUS "  Test: make test_python_module")
message(STATUS "  Install: make install")