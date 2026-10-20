# Energy demand forecasting demo

This example uses `ml_core.LinearRegression` to forecast the next day's
electricity demand from a short history of daily demand, calendar cycles, and
a simulated weather forecast.

The script simulates four years of reproducible daily observations, trains on
the earlier 80%, and evaluates on the later 20%. Features are standardized
using training-period statistics only. The result is compared with a simple
persistence baseline that predicts tomorrow's demand as today's demand. Data
are generated in memory; no CSV is required.

## Run

Use Python 3.11, NumPy, and the `ml_core` Python binding. From the repository
root, if the binding is available in `src/`:

```bash
PYTHONPATH=src python3.11 examples/python/energy_forecast/energy_forecast_demo.py
```

To vary the number of simulated days or the random seed:

```bash
PYTHONPATH=src python3.11 examples/python/energy_forecast/energy_forecast_demo.py --days 2000 --seed 7
```

The output reports MAE, RMSE, and R² on the held-out future period, alongside
the baseline metrics. The simulation is designed to demonstrate the workflow,
not to establish forecasting performance on real energy data.
