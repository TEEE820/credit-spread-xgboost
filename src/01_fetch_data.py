"""数据获取：中债收益率曲线 + 宏观变量 + 市场变量，全部来自 akshare。"""
import os
import time
import akshare as ak
import pandas as pd

RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


def _retry(fn, retries=3, wait=5):
    """网络请求重试：远端断连时退避重试。"""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} after {type(e).__name__}: {e}")
            time.sleep(wait * (attempt + 1))


def fetch_yield_curves(start_date: str, end_date: str) -> pd.DataFrame:
    """中债收益率曲线（含国债、中短期票据等多评级），单次请求限一年内。"""
    frames = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(years=1) - pd.DateOffset(days=1), end)
        part = _retry(lambda cur=cur, nxt=nxt: ak.bond_china_yield(
            start_date=cur.strftime("%Y%m%d"),
            end_date=nxt.strftime("%Y%m%d"),
        ))
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


def fetch_index_daily(symbol: str = "sh000300") -> pd.DataFrame:
    """沪深300 日频收盘价（新浪源，东财源被限流）。"""
    df = _retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_shibor3m() -> pd.DataFrame:
    """SHIBOR 3 个月拆借利率（替代已失效的 repo_rate_hist 接口）。"""
    df = _retry(lambda: ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="3月"))
    df = df[["报告日", "利率"]].rename(columns={"报告日": "date", "利率": "shibor3m"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    raw = RAW_DIR
    curves = fetch_yield_curves("20150101", "20260831")
    curves.to_csv(os.path.join(raw, "yield_curves.csv"), index=False, encoding="utf-8-sig")
    for key, df in fetch_macro_all().items():
        df.to_csv(os.path.join(raw, f"macro_{key}.csv"), index=False, encoding="utf-8-sig")
    fetch_index_daily().to_csv(os.path.join(raw, "hs300_daily.csv"), index=False, encoding="utf-8-sig")
    fetch_shibor3m().to_csv(os.path.join(raw, "shibor3m.csv"), index=False, encoding="utf-8-sig")
    print("all raw data saved to", raw)


if __name__ == "__main__":
    main()
