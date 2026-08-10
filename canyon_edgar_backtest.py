#!/usr/bin/env python3
"""
canyon_edgar_backtest.py — 8-K 事件研究(验证事件层 edge)
=========================================================
华尔街/学术验证事件信号的标准方法: event study。
对每个历史 8-K 备案, 测个股相对市场(全体等权)的**超额收益**(市场调整),
按事件类型 / 严重度聚合, 算平均超额、t统计量、胜率 —— 回答"8-K 事件有没有预测力"。

诚实边界: 只用最近 ~2 年的 8-K(EDGAR submissions.recent 覆盖), 价格用本地 deep history。
无前瞻偏差: 每个事件从"备案次日"开始算前向收益。

输出: edgar_8k_history.csv(缓存原始事件) + edgar_event_study.json(验证结果)
"""
from __future__ import annotations
import json, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from canyon_edgar_events import tickers, cik_map, _get, ITEM_MAP

ROOT = Path(__file__).parent
LOOKBACK_DAYS = 730
HORIZONS = [21, 63]           # 1月 / 3月 前向


def fetch_8k_history(cik, cutoff):
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    j = json.loads(_get(url).decode())
    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", []); dates = recent.get("filingDate", []); items = recent.get("items", [])
    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        d = dates[i] if i < len(dates) else ""
        if d < cutoff:
            continue
        it = items[i] if i < len(items) else ""
        et, sev, codes = "", 0, []
        for raw in str(it).split(","):
            code = "".join(ch for ch in raw.split("Item")[-1] if ch.isdigit() or ch == ".")[:4]
            if code in ITEM_MAP:
                codes.append(code)
                e, s, _ = ITEM_MAP[code]
                if s > sev:
                    et, sev = e, s
        out.append((d, et, sev, "|".join(sorted(set(codes)))))
    return out


def build_history():
    """拉全 8-K 历史 → edgar_8k_history.csv (有缓存则用, 但缓存里可能只是45天事件)"""
    p = ROOT / "edgar_8k_history.csv"
    if p.exists():
        try:
            h = pd.read_csv(p)
            if len(h) > 2000 and "items" in h.columns:   # 完整历史 且 含 item 代码
                return h
        except Exception:
            pass
    tks = tickers(); cm = cik_map(tks)
    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    rows, done = [], 0
    print(f"  拉 {len(tks)} 家的 8-K 历史(近{LOOKBACK_DAYS}天)...")
    for tk in tks:
        cik = cm.get(tk)
        if cik is None:
            continue
        try:
            for d, et, sev, codes in fetch_8k_history(cik, cutoff):
                rows.append({"ticker": tk, "date": d, "event_type": et, "severity": sev, "items": codes})
            done += 1
        except Exception:
            time.sleep(0.15); continue
        time.sleep(0.10)
        if done % 100 == 0:
            print(f"    ...{done} 家")
    h = pd.DataFrame(rows)
    h.to_csv(p, index=False)
    print(f"  ✓ {len(h)} 条 8-K 事件 / {done} 家")
    return h


def event_study(h):
    px = None
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        if (ROOT / f).exists():
            px = pd.read_csv(ROOT / f, index_col=0, parse_dates=True); break
    if px is None:
        return {}
    px = px.sort_index()
    rets = px.pct_change()
    mkt = rets.mean(axis=1)                       # 全体等权 = 市场
    idx = px.index
    HZ = 63                                       # 中性化聚焦 63 天

    # 行业映射 + 行业等权价格指数 (行业中性用)
    secmap = {}
    ap = ROOT / "alpha_scores.csv"
    if ap.exists():
        a = pd.read_csv(ap)
        if "sector" in a.columns:
            secmap = {str(r["ticker"]): str(r["sector"]) for _, r in a.iterrows()}
    sec_idx = {}                                  # sector → 等权价格序列
    for sec in set(secmap.values()):
        cols = [t for t in px.columns if secmap.get(t) == sec]
        if len(cols) >= 3:
            sec_idx[sec] = px[cols].mean(axis=1)
    # 每股 beta (全样本 cov/var vs market)
    mvar = float(mkt.var())
    beta = {}
    for t in px.columns:
        c = rets[t].cov(mkt)
        beta[t] = float(c / mvar) if mvar > 0 else 1.0

    h = h[pd.notna(h["date"])].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])

    recs = []
    for _, r in h.iterrows():
        tk = str(r["ticker"]); d = r["date"]
        if tk not in px.columns:
            continue
        pos = idx.searchsorted(d)
        if pos >= len(idx) - max(HORIZONS) - 1 or pos < 1:
            continue
        start = pos + 1
        p0 = float(px[tk].iloc[start])
        if np.isnan(p0):
            continue
        row = {"event_type": r["event_type"], "severity": int(r["severity"]),
               "items": str(r.get("items", ""))}
        ok = True
        for hor in HORIZONS:
            s_ret = float(px[tk].iloc[start + hor] / p0 - 1)
            m_ret = float(px.iloc[start + hor].mean() / px.iloc[start].mean() - 1)
            if np.isnan(s_ret) or np.isnan(m_ret):
                ok = False; break
            row[f"ab_{hor}"] = s_ret - m_ret       # 市场调整
        if not ok:
            continue
        # 63天 行业中性 + beta中性
        s_ret = float(px[tk].iloc[start + HZ] / p0 - 1)
        m_ret = float(px.iloc[start + HZ].mean() / px.iloc[start].mean() - 1)
        sec = secmap.get(tk, "")
        if sec in sec_idx:
            si = sec_idx[sec]
            sec_ret = float(si.iloc[start + HZ] / si.iloc[start] - 1)
            row["ab_sector"] = s_ret - sec_ret     # 行业中性: 减本行业等权
        row["ab_beta"] = s_ret - beta.get(tk, 1.0) * m_ret   # beta中性: 减 β×市场
        recs.append(row)
    df = pd.DataFrame(recs)
    if df.empty:
        return {}

    def stats(sub, col):
        if col not in sub.columns:
            return None
        x = sub[col].dropna().values
        if len(x) < 20:
            return None
        m = float(np.mean(x)); sd = float(np.std(x))
        t = m / sd * np.sqrt(len(x)) if sd > 0 else 0
        return {"n": len(x), "mean_ab_%": round(m * 100, 2), "t": round(t, 2),
                "hit_%": round(float((x > 0).mean()) * 100, 1)}

    out = {"total_events": len(df), "horizons": HORIZONS,
           "by_event_type": {}, "by_severity": {}, "neutralized_63d": {}, "by_item": {}}
    for et, sub in df.groupby("event_type"):
        if not et:
            continue
        out["by_event_type"][et] = {f"{hh}d": stats(sub, f"ab_{hh}") for hh in HORIZONS}
        out["neutralized_63d"][et] = {
            "market_adj": stats(sub, "ab_63"),
            "sector_neutral": stats(sub, "ab_sector"),
            "beta_neutral": stats(sub, "ab_beta"),
        }
    for sev, sub in df.groupby("severity"):
        out["by_severity"][f"sev{sev}"] = {f"{hh}d": stats(sub, f"ab_{hh}") for hh in HORIZONS}
    # 按 8-K item 细分 (每个 item 单独出现即计入)
    ITEM_DESC = {"1.01": "重大协议", "2.01": "资产收购处置", "2.02": "业绩发布", "2.05": "重组成本",
                 "2.06": "资产减值", "4.02": "财报重述", "5.02": "高管董事变动", "7.01": "RegFD", "8.01": "其他"}
    item_rows = {}
    for _, r in df.iterrows():
        for code in str(r.get("items", "")).split("|"):
            if code:
                item_rows.setdefault(code, []).append(r.get("ab_63"))
    for code, vals in item_rows.items():
        x = np.array([v for v in vals if v is not None and not np.isnan(v)])
        if len(x) >= 30:
            m = float(np.mean(x)); sd = float(np.std(x)); t = m / sd * np.sqrt(len(x)) if sd > 0 else 0
            out["by_item"][code] = {"desc": ITEM_DESC.get(code, code), "n": len(x),
                                    "mean_ab_%": round(m * 100, 2), "t": round(t, 2)}
    return out


def run():
    h = build_history()
    if h.empty:
        print("无 8-K 历史"); return {}
    res = event_study(h)
    if res:
        json.dump(res, open(ROOT / "edgar_event_study.json", "w"), ensure_ascii=False, indent=2)
    return res


def main():
    print("=" * 64)
    print("8-K 事件研究 — 验证事件层 edge (市场调整超额收益)")
    print("=" * 64)
    res = run()
    if not res:
        return
    print(f"  样本 {res['total_events']} 个 8-K 事件 · 前向 {res['horizons']} 天")
    print(f"\n  按事件类型(63天超额, t统计):")
    for et, v in sorted(res["by_event_type"].items(), key=lambda x: -(x[1].get('63d') or {}).get('t', -9)):
        s = v.get("63d")
        if s:
            sig = "✓显著" if abs(s["t"]) >= 2 else "弱" if abs(s["t"]) >= 1 else ""
            print(f"    {et:14} n={s['n']:>4}  超额{s['mean_ab_%']:+.2f}%  t={s['t']:+.2f}  胜率{s['hit_%']:.0f}%  {sig}")
    print(f"\n  ★ 中性化验证(63天, 剔除行业/beta暴露后 edge 是否还在):")
    print(f"    {'事件类型':<14}{'市场调整':>16}{'行业中性':>16}{'beta中性':>16}")
    for et, v in res.get("neutralized_63d", {}).items():
        def fmt(s): return f"{s['mean_ab_%']:+.2f}%(t{s['t']:+.1f})" if s else "—"
        print(f"    {et:<14}{fmt(v.get('market_adj')):>16}{fmt(v.get('sector_neutral')):>16}{fmt(v.get('beta_neutral')):>16}")
    print(f"\n  ★ 按 8-K item 细分(哪类事件最肥, 63天):")
    for code, v in sorted(res.get("by_item", {}).items(), key=lambda x: -x[1]["t"]):
        sig = "✓" if abs(v["t"]) >= 2 else ""
        print(f"    Item {code:<5} {v['desc']:<10} n={v['n']:>4}  超额{v['mean_ab_%']:+.2f}%  t={v['t']:+.2f} {sig}")
    print("\n  → edgar_event_study.json")


if __name__ == "__main__":
    main()
