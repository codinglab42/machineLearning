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