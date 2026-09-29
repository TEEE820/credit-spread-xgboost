# 信用利差 XGBoost 研究项目 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建从 akshare 数据获取到 XGBoost+SHAP 信用利差研究全流程的可复现代码库，产出论文图表与 GitHub 作品集项目。

**Architecture:** 五个脚本构成顺序流水线（fetch → features → model → rolling_eval → figures），每阶段读写 data/ 目录，全程可独立运行与测试。研究数据为中债中短期票据（AAA/AA+/AA）与国债收益率曲线（2015-2026，周频）。

**Tech Stack:** Python 3.10, akshare, pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, pytest

## Global Constraints

- Python 版本：3.10（机器已装 3.10.11，通过 `py` launcher 调用）
- 虚拟环境：项目内 `.venv/`（`py -m venv .venv`）
- 依赖固定于 requirements.txt（akshare、pandas、numpy、scikit-learn、xgboost、shap、matplotlib、seaborn、pytest）
- 所有命令在项目根目录 `c:\Users\63082\Desktop\固收大作业` 下执行
- 数据文件（data/raw、data/processed 中的 csv）不提交 git（加入 .gitignore）
- 样本区间：2015-01 至 2026-08（约 600 周）
- 利差定义：中短期票据收益率 − 同期限国债收益率（3 年期为主，5 年期备用）
- 宏观变量周频对齐必须滞后一期（用已完整公布的上一月数据），严禁未来函数
- 每次任务完成必须 git commit（用 `-c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn"` 参数，不修改全局 git config）

---

### Task 0: 环境搭建与依赖验证

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: 可用的 `.venv` 环境，`py -m pytest` 可执行

- [ ] **Step 1: 创建虚拟环境并安装依赖**

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

- [ ] **Step 2: 写入 requirements.txt**

```
akshare>=1.14
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.13
pytest>=8.0
openpyxl>=3.1
```

- [ ] **Step 3: 写入 .gitignore**

```
.venv/
__pycache__/
data/raw/
data/processed/
*.pyc
.ipynb_checkpoints/
figures/*.png
```

- [ ] **Step 4: 安装并验证导入**

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import akshare, pandas, sklearn, xgboost, shap; print(akshare.__version__, xgboost.__version__)"
```

Expected: 打印 akshare 与 xgboost 版本号，无报错。

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "chore: add environment setup"
```

---

### Task 1: 数据获取脚本（akshare 实测）

**Files:**
- Create: `src/01_fetch_data.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: Task 0 环境
- Produces:
  - `fetch_yield_curves(start_date: str, end_date: str) -> pd.DataFrame`：列含 曲线名称、日期、3月、6月、1年、3年、5年、7年、10年、30年
  - `fetch_macro_all() -> dict[str, pd.DataFrame]`：键为 cpi/ppi/pmi/social_financing/m2/industrial_value
  - `fetch_index_daily(symbol: str) -> pd.DataFrame`：沪深300 日频（date, close）
  - `fetch_repo_rate() -> pd.DataFrame`：R007（date, rate）
  - `main()`：拉取全部数据存入 `data/raw/*.csv`（yield_curves.csv、macro_*.csv、hs300_daily.csv、r007.csv）

- [ ] **Step 1: 写失败测试（只测数据形状约定，用小范围真实请求）**

```python
# tests/test_fetch.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from 01_fetch_data import fetch_yield_curves


def test_fetch_yield_curves_shape():
    df = fetch_yield_curves("20240101", "20240331")  # 3 个月，快
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    required = {"曲线名称", "日期", "3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"}
    assert required.issubset(df.columns)
    # 日期列应为可解析日期
    pd.to_datetime(df["日期"])
    # 关键验证：曲线集合中必须存在国债与至少一档信用债曲线
    names = set(df["曲线名称"].unique())
    assert any("国债" in n for n in names)
    assert any("票据" in n for n in names)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fetch.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现数据获取脚本**

```python
# src/01_fetch_data.py
"""数据获取：中债收益率曲线 + 宏观变量 + 市场变量，全部来自 akshare。"""
import os
import akshare as ak
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_yield_curves(start_date: str, end_date: str) -> pd.DataFrame:
    """中债收益率曲线（含国债、中短期票据等多评级），单次请求限一年内。"""
    frames = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1) - pd.DateOffset(days=1), end)
        part = ak.bond_china_yield(
            start_date=cur.strftime("%Y%m%d"),
            end_date=nxt.strftime("%Y%m%d"),
        )
        frames.append(part)
        cur = nxt + pd.DateOffset(days=1)
    df = pd.concat(frames, ignore_index=True)
    df["日期"] = pd.to_datetime(df["日期"])
    return df


def fetch_macro_all() -> dict:
    """宏观月频变量。接口名在实测时按 akshare 数据字典校准。"""
    result = {}
    result["cpi"] = ak.macro_china_cpi_yearly()
    result["ppi"] = ak.macro_china_ppi_yearly()
    result["pmi"] = ak.macro_china_pmi_yearly()
    result["m2"] = ak.macro_china_money_supply()
    result["social_financing"] = ak.macro_china_shrzgm()
    return result


def fetch_index_daily(symbol: str = "000300") -> pd.DataFrame:
    """沪深300 日频收盘价。"""
    df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date="20150101", end_date="20260831")
    return df[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})


def fetch_repo_rate() -> pd.DataFrame:
    """银行间回购定盘利率 R007 日频。"""
    df = ak.repo_rate_hist(start_date="20150101", end_date="20260831")
    return df


def main():
    os.makedirs(os.path.normpath(RAW_DIR), exist_ok=True)
    raw = os.path.normpath(RAW_DIR)
    curves = fetch_yield_curves("20150101", "20260831")
    curves.to_csv(os.path.join(raw, "yield_curves.csv"), index=False, encoding="utf-8-sig")
    for key, df in fetch_macro_all().items():
        df.to_csv(os.path.join(raw, f"macro_{key}.csv"), index=False, encoding="utf-8-sig")
    fetch_index_daily().to_csv(os.path.join(raw, "hs300_daily.csv"), index=False, encoding="utf-8-sig")
    fetch_repo_rate().to_csv(os.path.join(raw, "r007.csv"), index=False, encoding="utf-8-sig")
    print("all raw data saved to", raw)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_fetch.py -v`
Expected: PASS。若 macro 接口名报错，按 `akshare.akfamily.xyz/data/index.html` 数据字典校准接口名（cpi/ppi/pmi/m2/社融/R007 均在其中）。

- [ ] **Step 5: 运行完整拉取并检查评级覆盖（关键里程碑）**

Run: `.venv\Scripts\python.exe src/01_fetch_data.py`
然后: `.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_csv('data/raw/yield_curves.csv'); print(df['曲线名称'].unique())"`
Expected: 曲线名称包含 中债国债收益率曲线 + 中债中短期票据收益率曲线(AAA/AA+/AA 至少其一)。**记录实际评级覆盖**：若 AA+/AA 缺失，后续利差仅用 AAA 并启用期限维度（3Y/5Y）对比，同时给用户报告此决策点。

- [ ] **Step 6: Commit**

```bash
git add src/01_fetch_data.py tests/test_fetch.py
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "feat: fetch yield curves and macro data via akshare"
```

---

### Task 2: 特征工程（频率对齐、利差构建、无泄漏）

**Files:**
- Create: `src/02_features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `data/raw/yield_curves.csv`、`data/raw/macro_*.csv`、`data/raw/hs300_daily.csv`、`data/raw/r007.csv`
- Produces:
  - `build_weekly_spread(curves_df: pd.DataFrame, credit_name: str, treasury_name: str, tenor: str = "3年") -> pd.Series`：周频利差（百分比），索引为周五日期
  - `align_monthly_to_weekly(monthly: pd.DataFrame, weekly_index: pd.DatetimeIndex, date_col: str, value_col: str) -> pd.Series`：月频→周频 ffill 且滞后一期
  - `build_lag_features(series: pd.Series, lags: tuple = (1, 2, 3, 4)) -> pd.DataFrame`
  - `build_dataset() -> tuple[pd.DataFrame, list[str]]`：返回建模宽表与特征列名清单，并存 `data/processed/model_data.csv`

- [ ] **Step 1: 写失败测试（无泄漏 + 对齐正确性是核心）**

```python
# tests/test_features.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import numpy as np
from 02_features import build_weekly_spread, align_monthly_to_weekly, build_lag_features


def test_build_weekly_spread_computes_diff():
    credit = pd.DataFrame({
        "日期": pd.to_datetime(["2024-01-05", "2024-01-12"]),
        "3年": [3.5, 3.6], "曲线名称": ["中债中短期票据收益率曲线(AAA)"] * 2,
    })
    treasury = pd.DataFrame({
        "日期": pd.to_datetime(["2024-01-05", "2024-01-12"]),
        "3年": [2.5, 2.4], "曲线名称": ["中债国债收益率曲线"] * 2,
    })
    s = build_weekly_spread(pd.concat([credit, treasury]),
                            credit_name="中债中短期票据收益率曲线(AAA)",
                            treasury_name="中债国债收益率曲线", tenor="3年")
    assert np.allclose(s.values, [1.0, 1.2])


def test_align_monthly_lags_one_period_no_leakage():
    weekly_index = pd.date_range("2024-01-05", "2024-03-29", freq="W-FRI")
    monthly = pd.DataFrame({"日期": pd.to_datetime(["2024-01-31", "2024-02-29"]), "值": [1.0, 2.0]})
    s = align_monthly_to_weekly(monthly, weekly_index, date_col="日期", value_col="值")
    # 1 月的值 2 月才可用：2 月第一周（2/2）前必须为 NaN，2/2 之后等于 1.0
    assert pd.isna(s.loc["2024-01-26"])
    assert s.loc["2024-02-02"] == 1.0
    assert s.loc["2024-03-01"] == 2.0


def test_build_lag_features_shift():
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    lag_df = build_lag_features(s, lags=(1, 2))
    assert lag_df["lag_1"].iloc[1] == 10.0
    assert lag_df["lag_2"].iloc[2] == 10.0
    assert lag_df["lag_1"].iloc[0] is np.nan or pd.isna(lag_df["lag_1"].iloc[0])
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_features.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现特征工程**

```python
# src/02_features.py
"""特征工程：周频对齐、利差构建、宏观变量无泄漏对齐、滞后特征。"""
import os
import pandas as pd
import numpy as np

RAW = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

TREASURY = "中债国债收益率曲线"


def build_weekly_spread(curves_df, credit_name, treasury_name=TREASURY, tenor="3年"):
    """给定全部曲线数据，计算单评级周频利差（信用 - 国债，周五收盘）。"""
    credit = curves_df[curves_df["曲线名称"] == credit_name][["日期", tenor]].copy()
    treasury = curves_df[curves_df["曲线名称"] == treasury_name][["日期", tenor]].copy()
    c = credit.set_index("日期")[tenor]
    t = treasury.set_index("日期")[tenor]
    spread = c - t
    weekly = spread.resample("W-FRI").last().dropna()
    return weekly


def align_monthly_to_weekly(monthly, weekly_index, date_col, value_col):
    """月频序列对齐到周频：月度值在其公布次月起才可用（滞后一期，防未来函数）。"""
    m = monthly.copy()
    m[date_col] = pd.to_datetime(m[date_col])
    m = m.sort_values(date_col).dropna(subset=[value_col])
    m["可用月"] = (m[date_col] + pd.offsets.MonthEnd(1))
    s = m.set_index("可用月")[value_col]
    s = s[~s.index.duplicated(keep="last")]
    aligned = pd.Series(index=weekly_index, dtype=float)
    for idx in weekly_index:
        valid = s[s.index <= idx]
        aligned.loc[idx] = valid.iloc[-1] if len(valid) else np.nan
    return aligned


def build_lag_features(series, lags=(1, 2, 3, 4)):
    out = pd.DataFrame(index=series.index)
    for k in lags:
        out[f"lag_{k}"] = series.shift(k)
    return out


def build_dataset():
    """构建建模宽表并存档。"""
    curves = pd.read_csv(os.path.join(RAW, "yield_curves.csv"))
    curves["日期"] = pd.to_datetime(curves["日期"])
    hs300 = pd.read_csv(os.path.join(RAW, "hs300_daily.csv"), parse_dates=["date"])
    r007 = pd.read_csv(os.path.join(RAW, "r007.csv"))

    # 1) 各评级 3Y 利差
    names = [n for n in curves["曲线名称"].unique() if "中短期票据" in n]
    spreads = {}
    for n in names:
        s = build_weekly_spread(curves, credit_name=n, tenor="3年")
        spreads[n.split("(")[-1].rstrip(")")] = s
    weekly_index = pd.date_range("2015-01-02", "2026-08-28", freq="W-FRI")

    # 2) 市场变量（周频）
    hs300["date"] = pd.to_datetime(hs300["date"])
    hs300_w = hs300.set_index("date")["close"].resample("W-FRI").last().reindex(weekly_index)
    ret = hs300_w.pct_change()
    vol = hs300.set_index("date")["close"].pct_change().resample("W-FRI").std().reindex(weekly_index)
    # 10Y 国债水平与期限利差
    t10 = curves[curves["曲线名称"] == TREASURY].set_index("日期")["10年"]
    t1 = curves[curves["曲线名称"] == TREASURY].set_index("日期")["1年"]
    level10 = t10.resample("W-FRI").last().reindex(weekly_index)
    term = (t10 - t1).resample("W-FRI").last().reindex(weekly_index)

    # 3) 宏观变量（各接口列名不同，实测后按 data/raw 实际列名接入）
    macro_cols = {  # 文件名 -> (date_col, value_col)
        "macro_cpi": ("日期", "全国-当月"),
        "macro_ppi": ("日期", "当月"),
        "macro_pmi": ("日期", "制造业-指数"),
        "macro_m2": ("月份", "货币和准货币(M2)-数量(亿元)-期末值"),
        "macro_social_financing": ("月份", "社会融资规模增量"),
    }
    macro_feats = {}
    for fname, (dc, vc) in macro_cols.items():
        fp = os.path.join(RAW, fname + ".csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if dc not in df.columns or vc not in df.columns:
            continue  # 列名与预期不符时在 Task 2 实测中校准
        macro_feats[fname.replace("macro_", "")] = align_monthly_to_weekly(df, weekly_index, dc, vc)

    # 4) 汇总宽表
    out = pd.DataFrame(index=weekly_index)
    for rating, s in spreads.items():
        out[f"spread_{rating}"] = s.reindex(weekly_index)
    out["hs300_ret"] = ret
    out["hs300_vol"] = vol
    out["y10_level"] = level10
    out["term_spread"] = term
    for k, v in macro_feats.items():
        out[k] = v

    lag_cols = []
    for col in out.columns:
        if col.startswith("spread_"):
            lags = build_lag_features(out[col])
            for c in lags.columns:
                out[f"{col}_{c}"] = lags[c]
                lag_cols.append(f"{col}_{c}")
    os.makedirs(PROC, exist_ok=True)
    out.to_csv(os.path.join(PROC, "model_data.csv"), encoding="utf-8-sig")
    feature_cols = [c for c in out.columns if not c.startswith("spread_")]
    return out, feature_cols


if __name__ == "__main__":
    df, feats = build_dataset()
    print("model data shape:", df.shape)
    print("features:", feats)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_features.py -v`
Expected: 3 PASS

- [ ] **Step 5: 运行完整特征构建并人工核对宏观列名**

Run: `.venv\Scripts\python.exe src/02_features.py`
Expected: 打印 model data shape（约 600 周 × N 列）。**核对**：输出中宏观列若全缺失，说明 macro_cols 列名与 akshare 实际输出不符，按 data/raw/macro_*.csv 实际列名修改 macro_cols 字典后重跑。

- [ ] **Step 6: Commit**

```bash
git add src/02_features.py tests/test_features.py
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "feat: weekly alignment, spread construction, leak-free macro features"
```

---

### Task 3: 模型训练与 SHAP 分析

**Files:**
- Create: `src/03_model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `data/processed/model_data.csv`（Task 2 产出）
- Produces:
  - `train_baselines(X_train, y_train, X_test, y_test) -> dict[str, float]`：OLS/RandomForest 测试集 R²
  - `tune_xgboost(X, y) -> xgboost.XGBRegressor`：GridSearchCV 调优后模型
  - `shap_analysis(model, X) -> pd.DataFrame`：特征重要性表（feature, mean_abs_shap）
  - `main()`：训练全评级模型，输出模型性能表 `data/processed/model_performance.csv` 与 SHAP 重要性 `data/processed/shap_importance_{rating}.csv`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from 03_model import train_baselines, shap_analysis


def test_train_baselines_returns_r2():
    X, y = make_regression(n_samples=200, n_features=5, noise=1.0, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    n = 160
    scores = train_baselines(X.iloc[:n], y.iloc[:n], X.iloc[n:], y.iloc[n:])
    assert set(scores.keys()) == {"ols", "random_forest"}
    assert all(-10 < v <= 1.0 for v in scores.values())


def test_shap_analysis_returns_importance():
    from xgboost import XGBRegressor
    X, y = make_regression(n_samples=100, n_features=4, noise=0.5, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = XGBRegressor(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)
    imp = shap_analysis(model, X)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert len(imp) == 4
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现建模与 SHAP**

```python
# src/03_model.py
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


def train_baselines(X_train, y_train, X_test, y_test):
    """基准模型在测试集上的 R2。"""
    scores = {}
    ols = LinearRegression().fit(X_train, y_train)
    scores["ols"] = ols.score(X_test, y_test)
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)
    scores["random_forest"] = rf.score(X_test, y_test)
    return scores


def tune_xgboost(X, y):
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
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
    target_cols = [c for c in data.columns if c.startswith("spread_")]
    perf_rows = []
    for t in target_cols:
        rating = t.replace("spread_", "")
        df = data.dropna(subset=[t])
        feats = [c for c in df.columns if not c.startswith("spread_")]
        y = df[t]
        X = df[feats]
        split = int(len(df) * 0.7)
        Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
        baselines = train_baselines(Xtr, ytr, Xte, yte)
        xgb = tune_xgboost(Xtr, ytr)
        xgb_r2 = xgb.score(Xte, yte)
        perf_rows.append({"rating": rating, **baselines, "xgboost": xgb_r2,
                          "best_params": str(xgb.get_params())})
        imp = shap_analysis(xgb, Xte)
        imp.to_csv(os.path.join(PROC, f"shap_importance_{rating}.csv"), index=False)
    pd.DataFrame(perf_rows).to_csv(os.path.join(PROC, "model_performance.csv"), index=False)
    print(pd.DataFrame(perf_rows)[["rating", "ols", "random_forest", "xgboost"]])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 2 PASS

- [ ] **Step 5: 运行完整训练**

Run: `.venv\Scripts\python.exe src/03_model.py`
Expected: 打印各评级 R² 对比表（预期 xgboost ≥ 基准）。若 XGBoost 明显劣于 RF，调大 GridSearchCV 参数空间重跑。**记录各评级性能，作为论文"模型设计"章节素材。**

- [ ] **Step 6: Commit**

```bash
git add src/03_model.py tests/test_model.py
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "feat: baseline models, xgboost tuning, shap importance"
```

---

### Task 4: 滚动样本外预测评估

**Files:**
- Create: `src/04_rolling_eval.py`
- Test: `tests/test_rolling.py`

**Interfaces:**
- Consumes: `data/processed/model_data.csv`
- Produces:
  - `expanding_window_predict(X, y, train_years=5.0) -> pd.DataFrame`：列 date、y_true、y_pred（XGBoost，默认超参）、y_pred_rw（随机游走基准）
  - `eval_metrics(pred_df) -> dict`：rmse、mae、direction_accuracy、rw_rmse、rw_direction
  - `main()`：全评级滚动评估，存 `data/processed/rolling_results_{rating}.csv`

- [ ] **Step 1: 写失败测试（指标数值必须精确）**

```python
# tests/test_rolling.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from 04_rolling_eval import eval_metrics


def test_eval_metrics_exact():
    df = pd.DataFrame({
        "y_true": [1.0, 2.0, 1.5, 2.5],
        "y_pred": [1.2, 1.8, 1.6, 2.2],
        "y_pred_rw": [1.0, 1.0, 2.0, 1.5],
    })
    m = eval_metrics(df)
    assert np.isclose(m["rmse"], np.sqrt(np.mean([0.04, 0.04, 0.01, 0.09])))
    assert np.isclose(m["mae"], np.mean([0.2, 0.2, 0.1, 0.3]))
    # 方向准确率：预测变化与实际变化同号的比例（3 个变化里 2 个同号）
    assert np.isclose(m["direction_accuracy"], 2 / 3)
    assert "rw_rmse" in m
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rolling.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现滚动评估**

```python
# src/04_rolling_eval.py
"""滚动样本外预测：expanding window，5 年训练窗逐步推进。"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))


def expanding_window_predict(X, y, train_years=5.0, step_weeks=4):
    rows = []
    dates = y.index
    n = len(y)
    start = int(train_years * 52)
    for t in range(start, n, step_weeks):
        Xtr, ytr = X.iloc[:t], y.iloc[:t]
        Xte, yte = X.iloc[t:t + step_weeks], y.iloc[t:t + step_weeks]
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
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
    for t in [c for c in data.columns if c.startswith("spread_")]:
        rating = t.replace("spread_", "")
        df = data.dropna(subset=[t])
        feats = [c for c in df.columns if not c.startswith("spread_")]
        pred = expanding_window_predict(df[feats], df[t])
        m = eval_metrics(pred)
        pred.to_csv(os.path.join(PROC, f"rolling_results_{rating}.csv"), index=False)
        print(rating, m)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_rolling.py -v`
Expected: PASS

- [ ] **Step 5: 运行完整滚动评估**

Run: `.venv\Scripts\python.exe src/04_rolling_eval.py`
Expected: 各评级输出 rmse/mae/direction_accuracy/rw_rmse。**若 XGBoost 的 rmse 高于随机游走**，不阻塞——论文将预测部分定位为辅助验证，解释性分析为主卖点（见设计文档风险 2）；方向准确率若 >55% 即可作为亮点。

- [ ] **Step 6: Commit**

```bash
git add src/04_rolling_eval.py tests/test_rolling.py
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "feat: expanding-window out-of-sample evaluation vs random walk"
```

---

### Task 5: 论文图表与表格产出

**Files:**
- Create: `src/05_figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `data/processed/*.csv`（Task 2-4 全部产出）
- Produces: `figures/*.png` 全部论文图表、`tables/*.csv` 全部论文表格

- [ ] **Step 1: 写失败测试**

```python
# tests/test_figures.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from 05_figures import generate_all


def test_generate_all_creates_files():
    generated = generate_all()
    assert len(generated) >= 8
    for fp in generated:
        assert os.path.exists(fp), fp
        assert os.path.getsize(fp) > 1000, fp
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_figures.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现图表脚本（8 张核心图 + 3 张表）**

```python
# src/05_figures.py
"""论文图表：中文字体设置 + 全部图/表一键产出。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")


def generate_all():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    files = []
    data = pd.read_csv(os.path.join(PROC, "model_data.csv"), parse_dates=[0], index_col=0)
    ratings = [c.replace("spread_", "") for c in data.columns if c.startswith("spread_")]

    # 图1：三评级利差时序
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in ratings:
        ax.plot(data.index, data[f"spread_{r}"], label=r, lw=1)
    ax.set_title("图1 各评级信用利差时序（2015-2026）")
    ax.legend()
    fp = os.path.join(FIG, "fig1_spread_series.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图2：描述统计（箱线图）
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([data[f"spread_{r}"].dropna() for r in ratings], labels=ratings)
    ax.set_title("图2 各评级利差分布")
    fp = os.path.join(FIG, "fig2_boxplot.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图3-5：各评级 SHAP 重要性条形图
    for i, r in enumerate(ratings):
        imp = pd.read_csv(os.path.join(PROC, f"shap_importance_{r}.csv"))
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.head(12).iloc[::-1].plot.barh(x="feature", y="mean_abs_shap", ax=ax, legend=False)
        ax.set_title(f"图{3+i} {r} 评级 SHAP 特征重要性")
        fp = os.path.join(FIG, f"fig{3+i}_shap_{r}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图6：评级间 SHAP 对比
    merged = None
    for r in ratings:
        imp = pd.read_csv(os.path.join(PROC, f"shap_importance_{r}.csv")).set_index("feature")
        imp = imp.rename(columns={"mean_abs_shap": r})
        merged = imp if merged is None else merged.join(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    merged.head(10).plot.bar(ax=ax)
    ax.set_title("图6 评级间特征重要性对比")
    fp = os.path.join(FIG, "fig6_cross_rating_shap.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 图7-8：滚动预测（AAA 为例 + 全评级方向准确率汇总）
    r0 = ratings[0]
    roll = pd.read_csv(os.path.join(PROC, f"rolling_results_{r0}.csv"), parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll["date"], roll["y_true"], label="实际", lw=1)
    ax.plot(roll["date"], roll["y_pred"], label="XGBoost 预测", lw=1, alpha=0.8)
    ax.set_title(f"图7 {r0} 滚动样本外预测（5 年训练窗）")
    ax.legend()
    fp = os.path.join(FIG, "fig7_rolling_pred.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    perf = pd.read_csv(os.path.join(PROC, "model_performance.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    perf[["rating", "ols", "random_forest", "xgboost"]].set_index("rating").plot.bar(ax=ax)
    ax.set_title("图8 模型性能对比（R²）")
    ax.set_ylabel("R²")
    fp = os.path.join(FIG, "fig8_model_performance.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(fp)

    # 表1：描述统计表
    t1 = pd.DataFrame({
        r: [data[f"spread_{r}"].mean(), data[f"spread_{r}"].std(),
            data[f"spread_{r}"].min(), data[f"spread_{r}"].max()]
        for r in ratings
    }, index=["均值", "标准差", "最小值", "最大值"]).round(4)
    fp = os.path.join(TAB, "table1_summary_stats.csv"); t1.to_csv(fp, encoding="utf-8-sig"); files.append(fp)

    # 表2：模型性能表
    fp = os.path.join(TAB, "table2_model_performance.csv")
    perf.to_csv(fp, index=False, encoding="utf-8-sig"); files.append(fp)

    # 表3：滚动评估指标表
    rows = []
    for r in ratings:
        roll = pd.read_csv(os.path.join(PROC, f"rolling_results_{r}.csv"))
        err = roll["y_true"] - roll["y_pred"]
        err_rw = roll["y_true"] - roll["y_pred_rw"]
        dy, dp = roll["y_true"].diff().dropna(), roll["y_pred"].diff().dropna()
        rows.append({
            "评级": r, "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "方向准确率": float(np.mean(np.sign(dy) == np.sign(dp))),
            "随机游走RMSE": float(np.sqrt(np.mean(err_rw ** 2))),
        })
    t3 = pd.DataFrame(rows).round(4)
    fp = os.path.join(TAB, "table3_rolling_metrics.csv"); t3.to_csv(fp, index=False, encoding="utf-8-sig"); files.append(fp)

    print("generated", len(files), "files")
    return files


if __name__ == "__main__":
    generate_all()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_figures.py -v`
Expected: PASS（生成 ≥8 个文件）

- [ ] **Step 5: 人工检查图表质量**

逐一打开 figures/*.png 检查：中文无乱码、标签齐全、利差趋势与常识一致（如 2020 年疫情冲高、2022 年赎回潮冲高）。

- [ ] **Step 6: Commit**

```bash
git add src/05_figures.py tests/test_figures.py
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "feat: generate all paper figures and tables"
```

---

### Task 6: README、复现验证与收尾

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部前置任务
- Produces: 完整 GitHub 作品集仓库

- [ ] **Step 1: 写入 README.md（中文版式，含英文标题）**

```markdown
# Credit Spread XGBoost

可解释机器学习框架下中国信用利差的非线性决定机制与预测
（Explainable Machine Learning for Chinese Credit Spreads）

## 研究摘要

基于中债收益率曲线数据（2015-2026），构建 AAA/AA+/AA 中短期票据信用利差周频数据集，
使用 XGBoost + SHAP 揭示信用利差的非线性决定机制、时变特征重要性与评级异质性，
并通过 expanding-window 滚动验证评估样本外预测能力（基准：OLS、随机森林、随机游走）。

## 复现步骤

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/01_fetch_data.py   # 拉取原始数据（约 5 分钟）
.venv\Scripts\python.exe src/02_features.py     # 特征工程
.venv\Scripts\python.exe src/03_model.py        # 模型训练 + SHAP
.venv\Scripts\python.exe src/04_rolling_eval.py # 滚动样本外评估
.venv\Scripts\python.exe src/05_figures.py      # 生成全部图表
.venv\Scripts\python.exe -m pytest tests/ -v    # 全部测试
```

## 数据说明

- 数据源：中国债券信息网（经 akshare `bond_china_yield` 接口）、国家统计局（经 akshare 宏观接口）
- 利差定义：中债中短期票据收益率 − 同期限中债国债收益率
- 宏观变量采用滞后一期月频对齐，杜绝未来函数

## 主要结果

（Task 3/4 完成后回填：模型性能对比表、滚动评估指标表、SHAP 结论摘要）

## 目录结构

（见仓库根目录，src/ 下为可独立运行的流水线脚本）
```

- [ ] **Step 2: 全流程复现验证（从零重跑）**

```powershell
Remove-Item data\processed\* -Force
.venv\Scripts\python.exe src/02_features.py
.venv\Scripts\python.exe src/03_model.py
.venv\Scripts\python.exe src/04_rolling_eval.py
.venv\Scripts\python.exe src/05_figures.py
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: 全部脚本顺序跑通，pytest 全绿（7 个测试）。

- [ ] **Step 3: 回填 README 主要结果并最终提交**

把 Task 3/4 得到的关键数字写入 README「主要结果」小节，然后：

```bash
git add README.md
git -c user.name="Credit Spread Team" -c user.email="student@scnu.edu.cn" commit -m "docs: add README with reproduction steps and results"
```

- [ ] **Step 4: 向用户交付结果包**

向用户汇报：图/表清单（论文可直接引用）、各评级模型性能与预测指标、SHAP 主要发现（用于论文"具体分析研究过程"章节）、期刊投稿建议名单调研的启动。

---

## Self-Review 记录

- **Spec coverage**：数据方案→Task 1；特征体系→Task 2；模型与验证→Task 3/4；项目结构→Task 0/6；图表产出→Task 5；时间线→任务顺序即里程碑顺序。稳健性检验（期限 5Y、月频、LightGBM、h=4、分样本）按设计文档留待审稿阶段补充，不占用当前关键路径。
- **Placeholder scan**：Task 2 的宏观接口列名（macro_cols）为"实测后校准"——此为数据源列名的不确定性，已在 Step 5 明确校准操作与验收标准，非内容缺失。
- **Type consistency**：`build_weekly_spread`/`align_monthly_to_weekly`/`build_lag_features` 在 Task 2 定义并在测试中使用；`eval_metrics` 返回 dict 键名（rmse/mae/direction_accuracy/rw_rmse）在 Task 4 与 Task 5 表3 中一致。
