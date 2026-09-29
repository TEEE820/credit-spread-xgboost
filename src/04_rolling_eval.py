"""滚动样本外预测：expanding window，5 年训练窗逐步推进。"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from importlib import import_module

model = import_module("03_model")

PROC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


def expanding_window_predict(X, y, train_years=5.0, step_weeks=4):
    """expanding window 滚动：每 4 周重训一次，预测未来 4 周。"""
    rows = []
    dates = y.index
    n = len(y)
    start = int(train_years * 52)
    for t in range(start, n, step_weeks):
        Xtr, ytr = X.iloc[:t], y.iloc[:t]
        Xte, yte = X.iloc[t:t + step_weeks], y.iloc[t:t + step_weeks]
        if len(Xte) == 0:
            break
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.9, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        for i in range(len(Xte)):
            rows.append({
                "date": dates[t + i],
                "y_true": yte.iloc[i],
                "y_pred": pred[i],
                "y_pred_rw": ytr.iloc[-1],  # 随机游走：用上期真实值
            })
    return pd.DataFrame(rows)


def eval_metrics(pred_df):
    err = pred_df["y_true"] - pred_df["y_pred"]
    err_rw = pred_df["y_true"] - pred_df["y_pred_rw"]
    dy, dp = pred_df["y_true"].diff().dropna(), pred_df["y_pred"].diff().dropna()
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "direction_accuracy": float(np.mean(np.sign(dy) == np.sign(dp))),
        "rw_rmse": float(np.sqrt(np.mean(err_rw ** 2))),
    }


def main():
    data = pd.read_csv(os.path.join(PROC, "model_data.csv"), parse_dates=[0], index_col=0)
    rows = []
    for t in model.TARGETS:
        rating = t.replace("spread_", "")
        feats = model.features_for(t)
        df = data.dropna(subset=[t] + feats)
        pred = expanding_window_predict(df[feats], df[t])
        m = eval_metrics(pred)
        m["rating"] = rating
        rows.append(m)
        pred.to_csv(os.path.join(PROC, f"rolling_results_{rating}.csv"), index=False)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(PROC, "rolling_metrics.csv"), index=False)
    print(result.to_string())


if __name__ == "__main__":
    main()
