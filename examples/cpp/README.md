# C++ examples

The examples in this directory demonstrate using the library directly from
C++.

## Energy demand forecast

`example_energy_forecast.cpp` generates a reproducible synthetic daily energy
demand series, trains `models::LinearRegression` on its first 80%, and reports
MAE, RMSE, and R2 on the later period. It compares the model with the simple
baseline that predicts tomorrow's demand as today's demand. Features are
standardized with training-period statistics only.

Build and run it from the repository root:

```bash
cmake -S . -B build/examples -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTS=OFF
cmake --build build/examples --target example_energy_forecast
./build/examples/bin/example_energy_forecast
```

The generated data are synthetic and demonstrate the native C++ workflow;
their scores are not an estimate of real-world forecasting performance.
