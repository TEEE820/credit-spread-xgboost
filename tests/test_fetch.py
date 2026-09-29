import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from importlib import import_module

fetch = import_module("01_fetch_data")


def test_fetch_yield_curves_shape():
    df = fetch.fetch_yield_curves("20240101", "20240331")  # 3 个月，快
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
