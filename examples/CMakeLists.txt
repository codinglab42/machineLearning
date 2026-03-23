# examples/CMakeLists.txt

# Directory per gli esempi
set(EXAMPLES_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# ESEMPI C++
# ============================================================================

# 1. Linear Regression Example
add_executable(example_linear_regression
    ${EXAMPLES_SRC_DIR}/cpp/example_linear_regression.cpp
)
target_link_libraries(example_linear_regression PRIVATE
    ML::Library
    ${EIGEN3_LIBRARIES}
)
target_compile_options(example_linear_regression PRIVATE -O2)

# 2. Logistic Regression Example
add_executable(example_logistic_regression
    ${EXAMPLES_SRC_DIR}/cpp/example_logistic_regression.cpp
)
target_link_libraries(example_logistic_regression PRIVATE
    ML::Library
    ${EIGEN3_LIBRARIES}
)
target_compile_options(example_logistic_regression PRIVATE -O2)

# 3. Neural Network XOR Example
add_executable(example_neural_network_xor
    ${EXAMPLES_SRC_DIR}/cpp/example_neural_network_xor.cpp
)
target_link_libraries(example_neural_network_xor PRIVATE
    ML::Library
    ${EIGEN3_LIBRARIES}
)
target_compile_options(example_neural_network_xor PRIVATE -O2)

## 4. Optional: MNIST-like example (if you want to add it later)
## add_executable(neural_network_mnist
#     ${EXAMPLES_SRC_DIR}/cpp/neural_network_mnist.cpp
# )
# target_link_libraries(neural_network_mnist PRIVATE
#     ML::Library
#     ${EIGEN3_LIBRARIES}
# )
# target_compile_options(neural_network_mnist PRIVATE -O2)

# ============================================================================
# INSTALLAZIONE ESEMPI (opzionale)
# ============================================================================
if(CMAKE_INSTALL_PREFIX)
    install(TARGETS 
        example_linear_regression
        #logistic_regression_example
        #neural_network_xor
        RUNTIME DESTINATION bin/examples
    )
endif()

# ============================================================================
# MESSAGGIO DI CONFERMA
# ============================================================================
if(VERBOSE_OUTPUT)
    message(STATUS "✅ C++ examples configured:")
    message(STATUS "   - example_linear_regression")
    message(STATUS "   - logistic_regression_example")
    message(STATUS "   - neural_network_xor")
endif()