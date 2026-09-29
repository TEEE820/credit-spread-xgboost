"""特征工程：周频对齐、利差构建、宏观变量无泄漏对齐、滞后特征。"""
import os
import pandas as pd
import numpy as np

RAW = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

TREASURY = "中债国债收益率曲线"
CP_CURVE = "中债中短期票据收益率曲线(AAA)"        # 中票 AAA
BANK_CURVE = "中债商业银行普通债收益率曲线(AAA)"    # 商行债 AAA

# 利差序列定义：品种 x 期限
SPREAD_DEFS = {
    "cp_3y": (CP_CURVE, "3年"),
    "cp_5y": (CP_CURVE, "5年"),
    "bank_3y": (BANK_CURVE, "3年"),
    "bank_5y": (BANK_CURVE, "5年"),
}


def build_weekly_spread(curves_df, credit_name, treasury_name=TREASURY, tenor="3年"):
    """给定全部曲线数据，计算单品种周频利差（信用 - 国债，周五收盘）。"""
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
    # 可用时点 = 数据所属月次月月初（月度值公布后次月可用）
    m["可用月"] = (m[date_col].dt.to_period("M") + 1).dt.to_timestamp()
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


def parse_m2_month(x):
    """'2026年08月份' -> '2026-08-01'。"""
    return pd.Timestamp(x.replace("年", "-").replace("月份", "") + "-01")


def parse_sf_month(x):
    """201501 -> 2015-01-31。"""
    s = str(int(x))
    return pd.Timestamp(s[:4] + "-" + s[4:6] + "-01") + pd.offsets.MonthEnd(0)


def build_dataset():
    """构建建模宽表并存档。"""
    curves = pd.read_csv(os.path.join(RAW, "yield_curves.csv"))
    curves["日期"] = pd.to_datetime(curves["日期"])
    hs300 = pd.read_csv(os.path.join(RAW, "hs300_daily.csv"), parse_dates=["date"])
    shibor = pd.read_csv(os.path.join(RAW, "shibor3m.csv"), parse_dates=["date"])

    weekly_index = pd.date_range("2015-01-02", "2026-08-28", freq="W-FRI")

    # 1) 各品种 x 期限利差（周频）
    spreads = {}
    for key, (curve_name, tenor) in SPREAD_DEFS.items():
        spreads[key] = build_weekly_spread(curves, credit_name=curve_name, tenor=tenor)

    # 2) 市场变量（周频，价格缺失用最近交易日值填充后计算）
    hs300_c = hs300.set_index("date")["close"].sort_index()
    hs300_w = hs300_c.resample("W-FRI").last().reindex(weekly_index).ffill()
    ret = hs300_w.pct_change()
    daily_ret = hs300_c.pct_change()
    vol = daily_ret.resample("W-FRI").std().reindex(weekly_index).ffill()

    t10 = curves[curves["曲线名称"] == TREASURY].set_index("日期")["10年"]
    t1 = curves[curves["曲线名称"] == TREASURY].set_index("日期")["1年"]
    level10 = t10.resample("W-FRI").last().reindex(weekly_index).ffill()
    term = (t10 - t1).resample("W-FRI").last().reindex(weekly_index).ffill()

    shibor_w = shibor.set_index("date")["shibor3m"].resample("W-FRI").last().reindex(weekly_index).ffill()

    # 3) 宏观变量（无泄漏：次月可用）
    cpi = pd.read_csv(os.path.join(RAW, "macro_cpi.csv"))
    ppi = pd.read_csv(os.path.join(RAW, "macro_ppi.csv"))
    pmi = pd.read_csv(os.path.join(RAW, "macro_pmi.csv"))
    m2 = pd.read_csv(os.path.join(RAW, "macro_m2.csv"))
    sf = pd.read_csv(os.path.join(RAW, "macro_social_financing.csv"))

    macro_feats = {}
    macro_feats["cpi"] = align_monthly_to_weekly(cpi, weekly_index, "日期", "今值")
    macro_feats["ppi"] = align_monthly_to_weekly(ppi, weekly_index, "日期", "今值")
    macro_feats["pmi"] = align_monthly_to_weekly(pmi, weekly_index, "日期", "今值")
    m2["month_parsed"] = m2["月份"].map(parse_m2_month)
    macro_feats["m2_yoy"] = align_monthly_to_weekly(m2, weekly_index, "month_parsed", "货币和准货币(M2)-同比增长")
    sf["month_parsed"] = sf["月份"].map(parse_sf_month)
    sf = sf.sort_values("month_parsed")
    sf["sf_rolling12"] = sf["社会融资规模增量"].rolling(12).sum()
    macro_feats["sf_rolling12"] = align_monthly_to_weekly(sf, weekly_index, "month_parsed", "sf_rolling12")

    # 4) 汇总宽表
    out = pd.DataFrame(index=weekly_index)
    for key, s in spreads.items():
        out[f"spread_{key}"] = s.reindex(weekly_index)
    out["hs300_ret"] = ret
    out["hs300_vol"] = vol
    out["y10_level"] = level10
    out["term_spread"] = term
    out["shibor3m"] = shibor_w
    for k, v in macro_feats.items():
        out[k] = v

    # 5) 利差滞后特征
    for col in [c for c in out.columns if c.startswith("spread_")]:
        lags = build_lag_features(out[col])
        for c in lags.columns:
            out[f"{col}_{c}"] = lags[c]

    os.makedirs(PROC, exist_ok=True)
    out.to_csv(os.path.join(PROC, "model_data.csv"), encoding="utf-8-sig")
    feature_cols = [c for c in out.columns if not c.startswith("spread_")]
    return out, feature_cols


if __name__ == "__main__":
    df, feats = build_dataset()
    print("model data shape:", df.shape)
    print("features:", feats)
    print(df[["spread_cp_3y", "cpi", "m2_yoy"]].describe())
