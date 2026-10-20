# Time-series forecasting demo

This example predicts the next value of a daily series using the previous 28
observations as a sliding input window. The model is a feed-forward neural
network (`ml_core.NeuralNetwork`) trained for regression; the window is
flattened into its 28 lag features, so the example does not rely on the
library's recurrent layers.

The series is synthetic, reproducible, and combines weekly and annual cycles,
a gradual trend, and noise. The final 20% of windows is held out in chronological
order. Input and target scaling uses training-period statistics only. Model
errors are compared with persistence and weekly seasonal baselines. Data are
generated in memory; no CSV is required.

## Run

Use Python 3.11, NumPy, and the `ml_core` Python binding. From the repository
root, if the binding is available in `src/`:

```bash
PYTHONPATH=src python3.11 examples/python/timeseries_forecast/timeseries_forecast_demo.py
```

To change the simulation seed, series length, or training duration:

```bash
PYTHONPATH=src python3.11 examples/python/timeseries_forecast/timeseries_forecast_demo.py --days 2000 --seed 7 --epochs 150
```

The example tests a sliding-window forecasting workflow with the existing
Python binding. Synthetic metrics demonstrate execution only; real forecasting
quality must be checked on representative data and a genuinely future period.
