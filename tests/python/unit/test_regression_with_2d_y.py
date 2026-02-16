import pytest
import numpy as np
import machine_learning_module as ml


def test_regression_with_2d_y():
    X = np.random.rand(10, 3)
    y_1d = np.random.rand(10)        # shape (10,)
    y_2d = y_1d.reshape(-1, 1)       # shape (10,1)
    
    lr = ml.LinearRegression()
    
    # Questo funziona
    lr.fit(X, y_1d)
    
    # Questo funziona? Se sì, pybind11 sta facendo conversione automatica
    lr.fit(X, y_2d)