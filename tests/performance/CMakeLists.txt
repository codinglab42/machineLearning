# tests/performance/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Benchmark Linear Regression
add_executable(benchmark_linear
    benchmark_linear.cpp
)

target_link_libraries(benchmark_linear PRIVATE
    ML::Library
    pthread
)

target_compile_options(benchmark_linear PRIVATE -O3 -march=native)

# Aggiungi come test (opzionale)
add_test(NAME benchmark_linear COMMAND benchmark_linear)



# Benchmark Linear Regression
add_executable(benchmark_logistic
    benchmark_logistic.cpp
)

target_link_libraries(benchmark_logistic PRIVATE
    ML::Library
    pthread
)

target_compile_options(benchmark_logistic PRIVATE -O3 -march=native)

# Aggiungi come test (opzionale)
add_test(NAME benchmark_logistic COMMAND benchmark_logistic)


# Benchmark Neural Network
add_executable(benchmark_neural
    benchmark_neural.cpp
)

target_link_libraries(benchmark_neural PRIVATE
    ML::Library
    pthread
)

target_compile_options(benchmark_neural PRIVATE -O3 -march=native)

# Aggiungi come test (opzionale)
add_test(NAME benchmark_neural COMMAND benchmark_neural)

add_executable(benchmark_sequences benchmark_sequences.cpp)
target_link_libraries(benchmark_sequences PRIVATE ML::Library pthread)
target_compile_options(benchmark_sequences PRIVATE -O3 -march=native)
add_test(NAME benchmark_sequences COMMAND benchmark_sequences)

add_executable(benchmark_inference benchmark_inference.cpp)
target_link_libraries(benchmark_inference PRIVATE ML::Library pthread)
target_compile_options(benchmark_inference PRIVATE -O3 -march=native)
add_test(NAME benchmark_inference COMMAND benchmark_inference)

set_tests_properties(benchmark_linear benchmark_logistic benchmark_neural benchmark_sequences benchmark_inference
    PROPERTIES LABELS "performance")