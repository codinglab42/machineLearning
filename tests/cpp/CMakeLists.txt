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
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# --- Definizione dei Test ---

# Test per la cache dei layers
add_ml_test(test_cache_deep unit/cache/test_cache_deep.cpp) 
add_ml_test(test_cache_gemini unit/cache/test_cache_gemini.cpp)

# Test Atomici sui Layer (Gradient Checking, Forward/Backward)
add_ml_test(test_layers_deep unit/layers/test_layers_deep.cpp)
add_ml_test(test_layers_gemini unit/layers/test_layers_gemini.cpp)


#c# Test per Math Utils
#add_ml_test(test_math unit/test_math.cpp)

# Test per Linear Regression
add_ml_test(test_linear unit/models/test_linear.cpp)

# Test per Logistic Regression
add_ml_test(test_logistic unit/models/test_logistic.cpp)

# Test per Neural Network
add_ml_test(test_neural unit/models/test_neural.cpp)

message(STATUS "✅ C++ tests configured")
