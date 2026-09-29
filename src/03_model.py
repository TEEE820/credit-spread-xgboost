"""模型训练：OLS / 随机森林基准 + XGBoost 调优 + SHAP 可解释性分析。"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import shap

PROC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

# 目标利差列（品种 x 期限），滞后列 spread_*_lag_N 不属于目标
TARGETS = ["spread_cp_3y", "spread_cp_5y", "spread_bank_3y", "spread_bank_5y"]
# 外部因子 + 目标自身滞后项构成特征（不含其他利差的同期值，避免同期共线性主导）
EXTERNAL_FEATS = ["hs300_ret", "hs300_vol", "y10_level", "term_spread", "shibor3m",
                  "cpi", "ppi", "pmi", "m2_yoy", "sf_rolling12"]


def features_for(target):
    return EXTERNAL_FEATS + [f"{target}_lag_{k}" for k in (1, 2, 3, 4)]


def train_baselines(X_train, y_train, X_test, y_test):
    """基准模型在测试集上的 R2。"""
    scores = {}
    ols = LinearRegression().fit(X_train, y_train)
    scores["ols"] = ols.score(X_test, y_test)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)
    scores["random_forest"] = rf.score(X_test, y_test)
    return scores


def tune_xgboost(X, y):
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
    }
    gs = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring="neg_mean_squared_error", verbose=0,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def shap_analysis(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    imp = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return imp


def main():
    data = pd.read_csv(os.path.join(PROC, "model_data.csv"), parse_dates=[0], index_col=0)
    perf_rows = []
    best_params = None
    for t in TARGETS:
        rating = t.replace("spread_", "")
        feats = features_for(t)
        df = data.dropna(subset=[t] + feats)
        y = df[t]
        X = df[feats]
        split = int(len(df) * 0.7)
        Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
        baselines = train_baselines(Xtr, ytr, Xte, yte)
        if best_params is None:
            # 主利差（cp_3y）完整调优；其余利差复用最优参数，保证参数可比性
            xgb = tune_xgboost(Xtr, ytr)
            best_params = xgb.get_params()
        else:
            xgb = XGBRegressor(**best_params)
            xgb.fit(Xtr, ytr)
        xgb_r2 = xgb.score(Xte, yte)
        perf_rows.append({"rating": rating, **baselines, "xgboost": xgb_r2,
                          "best_params": str(best_params)})
        imp = shap_analysis(xgb, Xte)
        imp.to_csv(os.path.join(PROC, f"shap_importance_{rating}.csv"), index=False)
    pd.DataFrame(perf_rows).to_csv(os.path.join(PROC, "model_performance.csv"), index=False)
    print(pd.DataFrame(perf_rows)[["rating", "ols", "random_forest", "xgboost"]])


if __name__ == "__main__":
    main()
