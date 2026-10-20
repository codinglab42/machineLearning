# Predictive maintenance demo

This runnable example uses `ml_core.LogisticRegression` to predict whether a
machine will fail within the next seven days from daily temperature, vibration,
current, and sensor changes.

The script generates deterministic synthetic data for 20 machines over 365
days. It trains on the earlier period and evaluates on the later period, with a
seven-day gap so training labels cannot use failures from the test period.
Feature scaling is fitted on training data only. No generated CSV is needed or
committed.

## Run

Use Python 3.11, which is the version supported by this library, with NumPy and
the `ml_core` Python binding built or installed. From the repository root, if
the binding is available in `src/`:

```bash
PYTHONPATH=src python3.11 examples/python/predictive_maintenance/predictive_maintenance_demo.py
```

Change the random seed or alert probability threshold with:

```bash
PYTHONPATH=src python3.11 examples/python/predictive_maintenance/predictive_maintenance_demo.py --seed 7 --threshold 0.4
```

The output includes precision, recall, F1, a confusion matrix, false alerts per
100 machine-days, and the share of simulated failures detected in advance.
Accuracy is shown alongside the always-negative baseline because failures are
less common than normal days. Adjusting the threshold changes the trade-off
between missed failures and false alerts.

This example checks that the library can participate in a simple predictive
maintenance workflow. Its synthetic scores are not evidence of real-world
performance; production use requires representative sensor histories,
verified failure labels, and validation on genuinely later operating periods.
