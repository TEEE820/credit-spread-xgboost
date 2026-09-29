import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from importlib import import_module

model = import_module("03_model")


def test_train_baselines_returns_r2():
    X, y = make_regression(n_samples=200, n_features=5, noise=1.0, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    n = 160
    scores = model.train_baselines(X.iloc[:n], y.iloc[:n], X.iloc[n:], y.iloc[n:])
    assert set(scores.keys()) == {"ols", "random_forest"}
    assert all(-10 < v <= 1.0 for v in scores.values())


def test_shap_analysis_returns_importance():
    from xgboost import XGBRegressor
    X, y = make_regression(n_samples=100, n_features=4, noise=0.5, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    m = XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    m.fit(X, y)
    imp = model.shap_analysis(m, X)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert len(imp) == 4
