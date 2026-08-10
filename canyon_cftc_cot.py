#!/usr/bin/env python3
"""
canyon_cftc_cot.py — CFTC 持仓报告 (商品供需错配信号)
====================================================
免费官方(CFTC Socrata API)。看**投机资金(管理基金)** vs **商业(生产商/贸易商)**
在期货上的持仓极值 —— 这是"商品供需错配型"事件最缺的一块真实数据。

逻辑:
  管理基金净持仓 = m_money_long - m_money_short
  COT指数 = 净持仓在近3年的百分位(0-100)
  · COT指数极低(<25, 投机washout) = 供需错配的经典底部setup(反转向上空间)
  · COT指数极高(>80, 投机拥挤) = 追高风险
  · 商业持仓转多(生产商减空/囤货) = 现货紧俏信号
再把品种映射到标普500相关股(能源/材料/农业) → 给"商品供需错配型"事件加/减 L。

输出: cot_positioning.csv (品种级) + cot_ticker_signal.csv (股票级 cot_boost)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
UA = "Canyon Research canyon-research@example.com"

# 关注品种(name 关键词) → 标普500相关股
COMMODITIES = {
    "CRUDE OIL":     ("原油", ["XOM","CVX","COP","OXY","EOG","DVN","APA","FANG","HES","MRO","SLB","HAL","VLO","MPC","PSX"]),
    "NAT GAS":       ("天然气", ["EQT","LNG","WMB","KMI","OKE","TRGP","CTRA","EXPE"]),
    "GOLD":          ("黄金", ["NEM"]),
    "SILVER":        ("白银", ["NEM"]),
    "COPPER":        ("铜", ["FCX"]),
    "CORN":          ("玉米", ["CF","MOS","ADM","BG","CTVA","NTR","DE"]),
    "WHEAT":         ("小麦", ["CF","MOS","ADM","BG","CTVA","NTR"]),
    "SOYBEAN":       ("大豆", ["ADM","BG","CTVA","CF","MOS"]),
}


def fetch_commodity(name_kw, weeks=160):
    """拉某品种近 weeks 周的 COT。用 name LIKE 过滤。"""
    where = f"upper(commodity_name) like '%{name_kw}%'"
    params = {"$where": where, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": weeks * 6}
    r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    if d.empty:
        return d
    for c in ("m_money_positions_long_all", "m_money_positions_short_all",
              "prod_merc_positions_long", "prod_merc_positions_short"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["report_date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"], errors="coerce")
    # 选主力合约(未平仓最大的那个市场)聚合: 按报告日汇总所有相关合约
    d = d.dropna(subset=["report_date"])
    return d


def run():
    rows, tsig = [], {}
    for kw, (cn, tickers) in COMMODITIES.items():
        try:
            d = fetch_commodity(kw)
            time.sleep(0.3)
        except Exception as e:
            print(f"  {kw} 拉取失败: {e}"); continue
        if d.empty:
            continue
        # 按报告日聚合(多合约求和)
        g = d.groupby("report_date").agg(
            mm_long=("m_money_positions_long_all", "sum"),
            mm_short=("m_money_positions_short_all", "sum"),
            pm_long=("prod_merc_positions_long", "sum"),
            pm_short=("prod_merc_positions_short", "sum"),
        ).sort_index()
        if len(g) < 26:
            continue
        g["mm_net"] = g["mm_long"] - g["mm_short"]
        g["pm_net"] = g["pm_long"] - g["pm_short"]
        net = g["mm_net"]
        cur = float(net.iloc[-1])
        lo, hi = float(net.min()), float(net.max())
        cot_index = float((cur - lo) / (hi - lo) * 100) if hi > lo else 50.0
        # 商业(生产商)净持仓趋势: 近8周变化
        pm_chg = float(g["pm_net"].iloc[-1] - g["pm_net"].iloc[-9]) if len(g) > 9 else 0.0
        # setup 判定
        if cot_index < 25:
            setup, boost = "投机washout·底部setup", +0.4
        elif cot_index > 80:
            setup, boost = "投机拥挤·追高风险", -0.3
        elif pm_chg > 0 and cot_index < 50:
            setup, boost = "商业囤货·现货趋紧", +0.25
        else:
            setup, boost = "中性", 0.0
        rows.append({"commodity": cn, "keyword": kw,
                     "report_date": str(g.index[-1].date()),
                     "mm_net": int(cur), "cot_index": round(cot_index, 1),
                     "pm_net_chg_8w": int(pm_chg), "setup": setup, "cot_boost": boost,
                     "tickers": ",".join(tickers)})
        for t in tickers:
            # 一只股可能对应多个品种, 取最大绝对boost
            if t not in tsig or abs(boost) > abs(tsig[t][0]):
                tsig[t] = (boost, f"{cn}:{setup}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(ROOT / "cot_positioning.csv", index=False)
    ts = pd.DataFrame([{"ticker": t, "cot_boost": b, "cot_note": n} for t, (b, n) in tsig.items()])
    if not ts.empty:
        ts.to_csv(ROOT / "cot_ticker_signal.csv", index=False)
    return df


def main():
    print("=" * 60)
    print("CFTC 持仓报告 — 商品供需错配信号")
    print("=" * 60)
    df = run()
    if df.empty:
        print("  无输出"); return
    print(f"  {'品种':<8}{'COT指数':>8}{'管理基金净':>12}  setup")
    for _, r in df.iterrows():
        print(f"  {r['commodity']:<8}{r['cot_index']:>7.0f} {r['mm_net']:>12,}  {r['setup']}")
    fav = df[df["cot_boost"] > 0]
    print(f"\n  有利setup(底部/囤货): {', '.join(fav['commodity'].tolist()) or '无'}")
    print("  → cot_positioning.csv · cot_ticker_signal.csv (喂回商品供需错配型事件 L)")


if __name__ == "__main__":
    main()
