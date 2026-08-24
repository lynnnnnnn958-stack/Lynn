#!/usr/bin/env python3
"""
step_smallcap_sectors.py — 小盘 ticker→行业(SIC)缓存
=====================================================
组合层风控需要行业集中度上限, 但本地没有小盘行业数据。EDGAR 的 submissions
JSON 自带 sicDescription(标准行业分类)—— 快(每票 1 次调用, 不解析 XML)。

输出: smallcap_sectors.csv (ticker, sic, sector)  给 canyon_pm_desk 做行业集中度。
用法: python3 step_smallcap_sectors.py [--quick N]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd

from canyon_insider_scanner import universe, cikmap
from step_smallcap_form4 import _get, DELAY

ROOT = Path(__file__).parent
OUT = ROOT / "smallcap_sectors.csv"

# SIC 大类 → 可读行业(粗分, 够做集中度)
def _sic_group(sic):
    try:
        s = int(sic)
    except Exception:
        return "Unknown"
    g = [(999, "Agriculture"), (1499, "Mining/Energy"), (1799, "Construction"),
         (3999, "Manufacturing"), (4499, "Transport"), (4999, "Utilities"),
         (5199, "Wholesale"), (5999, "Retail"), (6799, "Finance/RealEstate"),
         (8999, "Services"), (9999, "Public/Other")]
    for hi, name in g:
        if s <= hi:
            return name
    return "Other"


def main():
    tks = universe()
    if "--quick" in sys.argv:
        tks = tks[:int(sys.argv[sys.argv.index("--quick") + 1])]
    cm = cikmap(tks)
    done = {}
    if OUT.exists():
        try:
            prev = pd.read_csv(OUT)
            done = {str(r["ticker"]): r for _, r in prev.iterrows()}
        except Exception:
            pass
    rows = list(done.values())
    todo = [t for t in tks if t in cm and t not in done]
    print(f"sectors: {len(done)} cached · {len(todo)} to fetch", flush=True)
    for k, tk in enumerate(todo, 1):
        try:
            j = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cm[tk]:010d}.json").decode())
            time.sleep(DELAY)
            sic = j.get("sic", ""); desc = j.get("sicDescription", "")
            rows.append({"ticker": tk, "sic": sic, "sic_desc": desc, "sector": _sic_group(sic)})
        except Exception as e:
            print(f"  {tk}: skip ({str(e)[:30]})", flush=True)
        if k % 50 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {k}/{len(todo)} · saved", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n✓ {len(df)} tickers → {OUT.name}")
    if not df.empty:
        print(df["sector"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
