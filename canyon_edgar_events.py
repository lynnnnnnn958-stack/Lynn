#!/usr/bin/env python3
"""
canyon_edgar_events.py — 真·SEC EDGAR 事件流 (8-K + Form 4)
============================================================
免费、官方、近实时。用 EDGAR submissions API 拉每家标普500公司的:
  · 近期 8-K 申报 (真实公司事件) + item 代码 → 映射到手册事件类型
  · 近期 Form 4 (内部人交易活动)
这是把"数据深度"从散户级(yfinance标题)推到半机构级的关键 —— 8-K 本身就是事件。

无需 API key, 只需合规 User-Agent。EDGAR 限速 ~10 req/s。
输出: edgar_events.csv (ticker, n_8k_30d, latest_8k_date, 8k_items, 8k_event_type,
       8k_severity, n_form4_30d, insider_active, note)
缓存: edgar_cik_map.json (ticker→CIK)
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests   # 用 requests(带 certifi 证书), 避开 macOS urllib SSL 验证失败

_SESSION = requests.Session()

ROOT = Path(__file__).parent
UA = "Canyon Research canyon-research@example.com"   # EDGAR 要求带联系方式的 UA
CIK_MAP = ROOT / "edgar_cik_map.json"
LOOKBACK_DAYS = 45

# 8-K item 代码 → 手册事件类型 + 严重度(0-4)
ITEM_MAP = {
    "1.01": ("第二春重估型", 2, "重大协议"),        "1.02": ("企业重大事故型", 2, "协议终止"),
    "1.03": ("企业重大事故型", 4, "破产/接管"),      "1.05": ("企业重大事故型", 4, "网络安全事件"),
    "2.01": ("第二春重估型", 3, "资产收购/处置"),    "2.02": ("行业爆发型", 2, "业绩发布"),
    "2.03": ("企业重大事故型", 2, "重大债务"),       "2.04": ("企业重大事故型", 3, "债务加速"),
    "2.05": ("企业重大事故型", 3, "重组成本"),       "2.06": ("企业重大事故型", 3, "资产减值"),
    "3.01": ("企业重大事故型", 3, "退市警告"),       "4.01": ("企业重大事故型", 2, "更换审计"),
    "4.02": ("企业重大事故型", 4, "财报不可靠/重述"), "5.01": ("第二春重估型", 3, "控制权变更"),
    "5.02": ("第二春重估型", 2, "高管/董事变动"),    "5.03": ("第二春重估型", 1, "章程修订"),
    "7.01": ("行业爆发型", 1, "Reg FD 披露"),        "8.01": ("行业爆发型", 1, "其他事件"),
}


def tickers():
    for f in ("alpha_scores.csv", "regime_ml_scores.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p)
            if "ticker" in d.columns:
                return d["ticker"].astype(str).str.upper().tolist()
    return []


def _get(url, timeout=15):
    r = _SESSION.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.content


def cik_map(tks):
    if CIK_MAP.exists():
        try:
            m = json.load(open(CIK_MAP))
            if len(m) > 100:
                return m
        except Exception:
            pass
    print("  拉取 EDGAR ticker→CIK 映射表 ...")
    try:
        j = json.loads(_get("https://www.sec.gov/files/company_tickers.json").decode())
    except Exception as e:
        print(f"  CIK 映射拉取失败: {e}"); return {}
    m = {}
    for _, v in j.items():
        m[str(v["ticker"]).upper()] = int(v["cik_str"])
    json.dump(m, open(CIK_MAP, "w"))
    return m


def fetch_one(cik, cutoff):
    """拉一家的 submissions, 返回近期 8-K(含items) 与 Form4 计数。"""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    j = json.loads(_get(url).decode())
    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", []); dates = recent.get("filingDate", [])
    items = recent.get("items", [])
    n8k, f4 = 0, 0
    all_items, latest8k = [], ""
    for i, form in enumerate(forms):
        d = dates[i] if i < len(dates) else ""
        if d < cutoff:
            continue
        if form == "8-K":
            n8k += 1
            if not latest8k:
                latest8k = d
            it = items[i] if i < len(items) else ""
            all_items += [x.strip() for x in str(it).split(",") if x.strip()]
        elif form in ("4", "4/A"):
            f4 += 1
    return n8k, latest8k, all_items, f4


def run():
    tks = tickers()
    if not tks:
        print("无 universe"); return pd.DataFrame()
    cm = cik_map(tks)
    if not cm:
        return pd.DataFrame()
    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    rows, done, fail = [], 0, 0
    for tk in tks:
        cik = cm.get(tk)
        if cik is None:
            continue
        try:
            n8k, latest8k, items, f4 = fetch_one(cik, cutoff)
            done += 1
        except Exception:
            fail += 1
            time.sleep(0.15)
            continue
        # item → 事件类型(取严重度最高的)
        et, sev, desc = "", 0, ""
        codes = []
        for it in items:
            code = it.split("Item")[-1].strip() if "Item" in it else it.strip()
            code = "".join(ch for ch in code if ch.isdigit() or ch == ".")[:4]
            if code in ITEM_MAP:
                codes.append(code)
                e, s, dsc = ITEM_MAP[code]
                if s > sev:
                    et, sev, desc = e, s, dsc
        rows.append({
            "ticker": tk, "n_8k_30d": n8k, "latest_8k_date": latest8k,
            "8k_items": ",".join(sorted(set(codes))), "8k_event_type": et,
            "8k_severity": sev, "8k_desc": desc,
            "n_form4_45d": f4, "insider_active": int(f4 >= 2),
            "note": f"{n8k}份8-K/{f4}份Form4/{LOOKBACK_DAYS}天" + (f"·{desc}" if desc else ""),
        })
        time.sleep(0.11)   # 限速
        if done % 100 == 0:
            print(f"    ...{done} 家已拉")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(ROOT / "edgar_events.csv", index=False)
    print(f"  ✓ {done} 家成功 / {fail} 失败 · edgar_events.csv")
    return df


def main():
    print("=" * 62)
    print("SEC EDGAR 真实事件流 — 8-K + Form 4 (标普500)")
    print("=" * 62)
    df = run()
    if df.empty:
        print("  无输出"); return
    has8k = df[df["n_8k_30d"] > 0]
    sev = df[df["8k_severity"] >= 3]
    ins = df[df["insider_active"] == 1]
    print(f"  有近期8-K: {len(has8k)} 家 · 高严重度(≥3)事件: {len(sev)} 家 · 内部人活跃: {len(ins)} 家")
    print("\n  高严重度 8-K 事件(值得关注):")
    for _, r in sev.sort_values("8k_severity", ascending=False).head(10).iterrows():
        print(f"    {r['ticker']:6} 严重度{r['8k_severity']} {r['8k_desc']:12} → {r['8k_event_type']} ({r['latest_8k_date']})")
    print("\n  → edgar_events.csv (喂回事件侦测 + C因子)")


if __name__ == "__main__":
    main()
