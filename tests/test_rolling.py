import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from importlib import import_module

rolling = import_module("04_rolling_eval")


def test_eval_metrics_exact():
    df = pd.DataFrame({
        "y_true": [1.0, 2.0, 1.5, 2.5],
        "y_pred": [1.2, 1.8, 1.6, 2.2],
        "y_pred_rw": [1.0, 1.0, 2.0, 1.5],
    })
    m = rolling.eval_metrics(df)
    assert np.isclose(m["rmse"], np.sqrt(np.mean([0.04, 0.04, 0.01, 0.09])))
    assert np.isclose(m["mae"], np.mean([0.2, 0.2, 0.1, 0.3]))
    # 方向准确率：预测变化与实际变化同号的比例（3 个变化全部同号）
    assert np.isclose(m["direction_accuracy"], 3 / 3)
    assert "rw_rmse" in m


def test_expanding_window_no_leakage():
    # 构造简单数据验证：t 时刻预测只使用 <=t 之前的数据（通过窗口切片结构保证）
    X = pd.DataFrame({"f1": np.arange(20.0)}, index=pd.date_range("2020-01-03", periods=20, freq="W-FRI"))
    y = pd.Series(np.arange(20.0), index=X.index)
    pred = rolling.expanding_window_predict(X, y, train_years=0.15, step_weeks=4)
    # train_years*52 = 7.8 -> start=7，预测从第 8 周开始，共 ceil((20-7)/4)=4 个窗口
    assert len(pred) > 0
    assert pred["y_true"].notna().all()
    assert pred["y_pred"].notna().all()
    # 所有预测日期都在首个训练窗之后
    assert pred["date"].min() >= X.index[7]
