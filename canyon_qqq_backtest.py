#!/usr/bin/env python3
"""
canyon_qqq_backtest.py — 事件策略 vs 纳指(QQQ) 头对头回测
==========================================================
诚实回答"能不能跑赢纳指"。只用**验证过的 8-K 事件信号**建仓, 对着 QQQ 比。
并展示"建对位置"(顺势入场)对回撤的作用: 同时跑三条线 ——

  ① QQQ 基准
  ② 事件策略(裸): 每周持有"近21天有高价值8-K事件"的股票, 等权
  ③ 事件策略+顺势过滤(建对位置): 同上, 但只买"站上50日线"的 → 看回撤能否压下来

用 edgar_8k_history.csv(验证过的事件) + sp500 价格 + qqq_price.csv。无前瞻: 每个再平衡日
只用当日及之前的信息。事件类型只取验证过显著的(行业爆发/第二春/企业事故), item 偏好高t的。

输出: qqq_backtest.json + 控制台对比
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

VALID_TYPES = {"行业爆发型", "第二春重估型", "企业重大事故型"}   # 已验证显著
GOOD_ITEMS = {"7.01", "2.02", "8.01", "5.03", "1.01", "5.02"}     # 高t的item
EVENT_WINDOW = 21      # 事件后多少交易日内算"在漂移窗口"
REBAL = 5              # 每5个交易日(周)再平衡
MAX_HOLD = 30          # 单portfolio最多持多少只(集中)


def load_prices():
    p = ROOT / "sp500_price_history_deep.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index() if p.exists() else pd.DataFrame()


def load_qqq():
    p = ROOT / "qqq_price.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    col = d.columns[0]
    return d[col].dropna()


def load_pit():
    """PIT 成分股: 返回 (sorted_dates, {date: set(tickers)})。每个历史时点谁真在指数里。"""
    p = ROOT / "sp500_pit_membership.csv"
    if not p.exists():
        return [], {}
    m = pd.read_csv(p)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"])
    d = {dt: set(g["ticker"].astype(str)) for dt, g in m.groupby("date")}
    return sorted(d.keys()), d


def pit_members(pit_dates, pit_map, asof):
    """asof 当日, 用最近一次 ≤asof 的PIT快照的成分股集合。"""
    if not pit_dates:
        return None
    import bisect
    i = bisect.bisect_right(pit_dates, asof) - 1
    if i < 0:
        return None
    return pit_map[pit_dates[i]]


def load_events():
    p = ROOT / "edgar_8k_history.csv"
    if not p.exists():
        return pd.DataFrame()
    e = pd.read_csv(p)
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e = e.dropna(subset=["date"])
    e = e[e["event_type"].isin(VALID_TYPES)]
    # item 偏好: 至少命中一个高t item(有 items 列时)
    if "items" in e.columns:
        e["good"] = e["items"].astype(str).apply(lambda s: any(c in GOOD_ITEMS for c in s.split("|")))
        e = e[e["good"]]
    return e


def backtest(px, qqq, events, trend_filter, max_hold=MAX_HOLD, pit_dates=None, pit_map=None):
    """返回该策略的日净值序列。全 numpy 加速: 价格转矩阵, 用整数位置。
    pit_dates/pit_map 提供则做 PIT 成分股过滤(去前视纳入偏差)。"""
    idx = px.index
    cols = list(px.columns)
    col_i = {c: k for k, c in enumerate(cols)}
    use_pit = bool(pit_dates)
    P = px.values.astype(float)                      # (T, N)
    MA = px.rolling(50).mean().values.astype(float)
    T = len(idx)
    # 每个事件 → (ticker列号, 备案位置)
    ev_by_pos = {}                                   # 备案位置 → set(列号)
    dmin = events["date"].min()
    for _, r in events.iterrows():
        tk = str(r["ticker"])
        if tk not in col_i:
            continue
        pos = idx.searchsorted(r["date"])
        if 0 <= pos < T:
            ev_by_pos.setdefault(pos, set()).add(col_i[tk])

    start_i = max(60, idx.searchsorted(dmin) + 21)
    nav = np.ones(T - start_i)
    held = np.array([], dtype=int)
    for k in range(T - start_i - 1):
        gpos = start_i + k
        if k % REBAL == 0:
            # 近 EVENT_WINDOW 天内有事件的列号
            cand = set()
            for pp in range(max(0, gpos - EVENT_WINDOW), gpos + 1):
                cand |= ev_by_pos.get(pp, set())
            cand = [c for c in cand if not np.isnan(P[gpos, c]) and P[gpos, c] > 0]
            if use_pit:                                   # PIT 过滤: 只留当时真在指数里的
                members = pit_members(pit_dates, pit_map, idx[gpos])
                if members:
                    cand = [c for c in cand if cols[c] in members]
            if trend_filter:
                cand = [c for c in cand if not np.isnan(MA[gpos, c]) and P[gpos, c] > MA[gpos, c]]
            if cand:
                held = np.array(cand[:max_hold], dtype=int)
        # 当日→次日
        if held.size:
            p0 = P[gpos, held]; p1 = P[gpos + 1, held]
            m = (~np.isnan(p0)) & (~np.isnan(p1)) & (p0 > 0)
            r = float(np.mean(p1[m] / p0[m] - 1)) if m.any() else 0.0
        else:
            r = 0.0
        nav[k + 1] = nav[k] * (1 + r)
    return pd.Series(nav, index=idx[start_i:start_i + len(nav)])


def metrics(nav):
    if len(nav) < 20:
        return {}
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    yrs = len(nav) / 252
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1)
    rets = nav.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    dd = float((nav / nav.cummax() - 1).min())
    return {"total_%": round(total * 100, 1), "cagr_%": round(cagr * 100, 1),
            "sharpe": round(sharpe, 2), "max_dd_%": round(dd * 100, 1)}


def run():
    print("  载入数据...", flush=True)
    px = load_prices(); qqq = load_qqq(); events = load_events()
    if px.empty or qqq.empty or events.empty:
        print("缺数据 (价格/QQQ/事件)"); return {}
    # 只保留事件区间前 120 天起的价格(回测只需近 ~2.5 年, 别算16年)
    if not events.empty:
        start = events["date"].min() - pd.Timedelta(days=120)
        px = px[px.index >= start]
    print(f"  价格裁剪到 {px.shape} · 事件 {len(events)} · 开始回测", flush=True)
    pit_dates, pit_map = load_pit()
    nav_bare = backtest(px, qqq, events, trend_filter=False)
    nav_trend = backtest(px, qqq, events, trend_filter=True)
    nav_pit = backtest(px, qqq, events, trend_filter=True, pit_dates=pit_dates, pit_map=pit_map)
    # QQQ 对齐到策略区间
    q = qqq.reindex(nav_bare.index).ffill()
    q_nav = q / q.iloc[0]
    out = {
        "window": f"{nav_bare.index[0].date()} → {nav_bare.index[-1].date()}",
        "QQQ": metrics(q_nav),
        "事件策略_裸": metrics(nav_bare),
        "事件策略_顺势过滤": metrics(nav_trend),
        "事件策略_顺势_PIT成分股": metrics(nav_pit),
    }
    json.dump(out, open(ROOT / "qqq_backtest.json", "w"), ensure_ascii=False, indent=2)
    return out


def main():
    print("=" * 62)
    print("事件策略 vs 纳指(QQQ) — 头对头回测")
    print("=" * 62)
    r = run()
    if not r:
        return
    print(f"  区间: {r['window']}\n")
    print(f"  {'策略':<24}{'总收益':>9}{'年化':>8}{'夏普':>7}{'最大回撤':>9}")
    for name in ["QQQ", "事件策略_裸", "事件策略_顺势过滤", "事件策略_顺势_PIT成分股"]:
        m = r.get(name, {})
        if m:
            print(f"  {name:<20}{m['total_%']:>8.1f}%{m['cagr_%']:>7.1f}%{m['sharpe']:>7.2f}{m['max_dd_%']:>8.1f}%")
    q = r["QQQ"]; e = r.get("事件策略_顺势_PIT成分股") or r["事件策略_顺势过滤"]
    if q and e:
        diff = e["cagr_%"] - q["cagr_%"]
        print(f"\n  ★ PIT成分股版(去前视纳入偏差) vs QQQ: 年化{'跑赢' if diff>0 else '跑输'} {abs(diff):.1f}个点")
    print("\n  → qqq_backtest.json")
    print("  诚实提示: PIT版只从当时真在指数的股票选(去前视纳入偏差), 但仍缺324只退市股价格,")
    print("  残余幸存者偏差未除净; 2年样本、等权、未计成本。参考, 非承诺。")


if __name__ == "__main__":
    main()
