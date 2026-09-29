import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import numpy as np
from importlib import import_module

feat = import_module("02_features")


def test_build_weekly_spread_computes_diff():
    credit = pd.DataFrame({
        "日期": pd.to_datetime(["2024-01-05", "2024-01-12"]),
        "3年": [3.5, 3.6], "曲线名称": ["中债中短期票据收益率曲线(AAA)"] * 2,
    })
    treasury = pd.DataFrame({
        "日期": pd.to_datetime(["2024-01-05", "2024-01-12"]),
        "3年": [2.5, 2.4], "曲线名称": ["中债国债收益率曲线"] * 2,
    })
    s = feat.build_weekly_spread(pd.concat([credit, treasury]),
                                 credit_name="中债中短期票据收益率曲线(AAA)",
                                 treasury_name="中债国债收益率曲线", tenor="3年")
    assert np.allclose(s.values, [1.0, 1.2])


def test_align_monthly_lags_one_period_no_leakage():
    weekly_index = pd.date_range("2024-01-05", "2024-03-29", freq="W-FRI")
    monthly = pd.DataFrame({"日期": pd.to_datetime(["2024-01-31", "2024-02-29"]), "值": [1.0, 2.0]})
    s = feat.align_monthly_to_weekly(monthly, weekly_index, date_col="日期", value_col="值")
    # 1 月的值 2 月才可用：2 月第一周（2/2）前必须为 NaN，2/2 之后等于 1.0
    assert pd.isna(s.loc["2024-01-26"])
    assert s.loc["2024-02-02"] == 1.0
    assert s.loc["2024-03-01"] == 2.0


def test_build_lag_features_shift():
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    lag_df = feat.build_lag_features(s, lags=(1, 2))
    assert lag_df["lag_1"].iloc[1] == 10.0
    assert lag_df["lag_2"].iloc[2] == 10.0
    assert pd.isna(lag_df["lag_1"].iloc[0])
