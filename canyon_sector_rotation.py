#!/usr/bin/env python3
"""
canyon_sector_rotation.py — 行业板块轮动信号
=============================================
用标普500全体按 GICS 板块分组, 判断资金正在"轮入"还是"轮出"哪些板块。
不用抓 ETF —— 直接用本地 495 只成分股价格 + 行业标签聚合, 就是板块信号本身。

每个板块算四个维度:
  1. 动量 (1M / 3M / 6M 等权平均涨幅)
  2. 相对强度 RS —— 板块 3M 涨幅 减 全市场 3M 涨幅 (跑赢/跑输大盘)
  3. 加速度 —— 1M动量 vs 3M动量(月化), >0=正在加速(轮入), <0=在退潮(轮出)
  4. 广度 —— 板块内多少比例的股票站上 50 / 200 日线

综合 → 轮动分 RotationScore → 排名 → 超配/中性/低配 建议。
并与事件系统联动: 标注该板块对应的事件类型 + 是否是当前激活链条。

输出: sector_rotation.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# 板块 → 主导事件类型 (与 canyon_build_pool 的映射一致)
SECTOR_EVENT = {
    "Technology": "行业爆发型", "Communication Services": "行业爆发型",
    "Energy": "商品供需错配型", "Materials": "商品供需错配型",
    "Industrials": "行业爆发型", "Utilities": "行业爆发型",
    "Health Care": "第二春重估型", "Financials": "第二春重估型",
    "Consumer Discretionary": "第二春重估型", "Consumer Staples": "第二春重估型",
    "Real Estate": "第二春重估型",
}


def universe_sectors():
    for f in ("alpha_scores.csv", "regime_ml_scores.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(f)
            if "ticker" in d.columns and "sector" in d.columns:
                return {str(r["ticker"]): str(r["sector"]) for _, r in d.iterrows()}
    return {}


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def _ret(s: pd.Series, days: int):
    s = s.dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1)


def active_chains():
    p = ROOT / "macro_intel_scorecard.json"
    if p.exists():
        try:
            j = json.load(open(p))
            return set(j.get("重点受益链条", []) or []), set(j.get("激活事件池", []) or [])
        except Exception:
            pass
    return set(), set()


def run():
    secmap = universe_sectors()
    px = prices()
    if px.empty or not secmap:
        print("缺价格或行业数据"); return pd.DataFrame()

    # 全市场基准 (全体成分股等权 3M)
    mkt_3m = np.nanmean([_ret(px[t], 63) for t in px.columns if t in secmap])

    benefit_chains, active_pools = active_chains()
    rows = []
    for sec in sorted(set(secmap.values())):
        if sec in ("Broad", "nan", ""):
            continue
        tks = [t for t, s in secmap.items() if s == sec and t in px.columns]
        if len(tks) < 3:
            continue
        m1 = np.nanmean([_ret(px[t], 21) for t in tks])
        m3 = np.nanmean([_ret(px[t], 63) for t in tks])
        m6 = np.nanmean([_ret(px[t], 126) for t in tks])
        rs = m3 - mkt_3m                                   # 相对强度 vs 大盘
        accel = m1 - m3 / 3.0                              # 加速度: 月化1M vs 3M均速
        # 广度
        above50 = above200 = n = 0
        for t in tks:
            s = px[t].dropna()
            if len(s) < 205:
                continue
            n += 1
            p_now = float(s.iloc[-1])
            above50 += p_now > float(s.tail(50).mean())
            above200 += p_now > float(s.tail(200).mean())
        breadth50 = above50 / n if n else 0.0
        breadth200 = above200 / n if n else 0.0
        # 综合轮动分 (z 前先原始加权, 之后统一标准化)
        raw = (rs * 100) * 0.4 + (accel * 100) * 0.3 + (breadth50 - 0.5) * 100 * 0.2 + (m6 * 100) * 0.1
        rows.append({"sector": sec, "n": len(tks),
                     "mom_1m": round(m1, 4), "mom_3m": round(m3, 4), "mom_6m": round(m6, 4),
                     "rel_strength": round(rs, 4), "accel": round(accel, 4),
                     "breadth_50d": round(breadth50, 2), "breadth_200d": round(breadth200, 2),
                     "raw_score": raw,
                     "event_type": SECTOR_EVENT.get(sec, "—"),
                     "macro_aligned": SECTOR_EVENT.get(sec, "") in active_pools})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 标准化为 0-100 轮动分, 排名, 建议
    r = df["raw_score"]
    df["rotation_score"] = ((r - r.min()) / (r.max() - r.min()) * 100).round(1) if r.max() > r.min() else 50.0
    df = df.sort_values("rotation_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    n = len(df)
    def _sig(i):
        if i < max(2, n // 3):   return "超配 · 轮入"
        if i >= n - max(2, n // 3): return "低配 · 轮出"
        return "中性"
    df["signal"] = [(_sig(i)) for i in range(n)]
    # 方向: 加速度 + 广度 判断趋势方向
    df["direction"] = np.where(df["accel"] > 0.01, "▲ 加速", np.where(df["accel"] < -0.01, "▼ 退潮", "→ 平"))
    # 🔥 焦点链条: 既是轮入超配(动量强) 又有宏观主线支撑(macro_aligned) —— 真正值得重点找机会
    df["active_chain"] = df["signal"].str.contains("超配") & df["macro_aligned"]
    df = df.drop(columns=["raw_score", "macro_aligned"])
    df.to_csv(ROOT / "sector_rotation.csv", index=False)
    return df


def main():
    print("=" * 60)
    print("行业板块轮动信号 — 标普500全体聚合")
    print("=" * 60)
    df = run()
    if df.empty:
        print("  无输出"); return
    print(f"  {'板块':<24}{'轮动分':>7} {'信号':<12}{'方向':<8}{'相对强度':>9}{'广度50':>8}")
    for _, r in df.iterrows():
        chain = " 🔥激活" if r["active_chain"] else ""
        print(f"  {r['sector']:<22}{r['rotation_score']:>7} {r['signal']:<12}{r['direction']:<8}"
              f"{r['rel_strength']*100:>+8.1f}%{r['breadth_50d']*100:>7.0f}%{chain}")
    lead = df[df["signal"].str.contains("超配")]["sector"].tolist()
    lag = df[df["signal"].str.contains("低配")]["sector"].tolist()
    print(f"\n  轮入(超配): {', '.join(lead)}")
    print(f"  轮出(低配): {', '.join(lag)}")
    print("\n  → sector_rotation.csv")


if __name__ == "__main__":
    main()
