"""补拉缺失的沪深300与SHIBOR数据。"""
import os
import importlib.util

spec = importlib.util.spec_from_file_location("fetch", os.path.join(os.path.dirname(__file__), "src", "01_fetch_data.py"))
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

for fn, fname in [
    (fetch.fetch_index_daily, "hs300_daily.csv"),
    (fetch.fetch_shibor3m, "shibor3m.csv"),
]:
    out = os.path.join(fetch.RAW_DIR, fname)
    if os.path.exists(out):
        print(f"{fname} exists, skip")
        continue
    try:
        df = fn()
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{fname} saved, rows={len(df)}")
    except Exception as e:
        print(f"{fname} FAILED: {type(e).__name__}: {e}")
