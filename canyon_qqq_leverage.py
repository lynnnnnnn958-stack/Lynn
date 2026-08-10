#!/usr/bin/env python3
"""
canyon_qqq_leverage.py — 集中度 × 杠杆 扫描(找"能睡着觉的激进度")
================================================================
把验证过的事件策略(顺势过滤), 在不同集中度(持股数)和杠杆下跑, 看年化能推多高、
回撤会到多深, 对着 QQQ 比。帮你亲眼选一个"收益够猛、回撤你扛得住"的配置。

杠杆按日收益线性放大, 扣融资成本(年化6%借贷×(L-1))。
诚实警告: 这是 2 年单一牛市样本, 杠杆会同比放大**已知**回撤;
真实的熊市回撤会比这里更深(可能 ×1.5~2)。别把这张表当承诺。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from canyon_qqq_backtest import load_prices, load_qqq, load_events, backtest, metrics

ROOT = Path(__file__).parent
BORROW = 0.06                        # 年化融资成本
CONCENTRATIONS = [5, 10, 20, 30]
LEVERAGES = [1.0, 1.5, 2.0, 2.5, 3.0]


def lever(nav, L):
    """把净值序列的日收益放大 L 倍, 扣融资成本, 返回新净值。"""
    r = nav.pct_change().fillna(0).values
    daily_borrow = BORROW / 252 * (L - 1)
    lr = r * L - daily_borrow
    out = np.cumprod(1 + lr)
    return pd.Series(out, index=nav.index)


def run():
    px = load_prices(); qqq = load_qqq(); events = load_events()
    if px.empty or qqq.empty or events.empty:
        print("缺数据"); return {}
    start = events["date"].min() - pd.Timedelta(days=120)
    px = px[px.index >= start]

    # QQQ 基准
    grid = {}
    base_navs = {}
    for c in CONCENTRATIONS:
        base_navs[c] = backtest(px, qqq, events, trend_filter=True, max_hold=c)
    q = qqq.reindex(base_navs[CONCENTRATIONS[0]].index).ffill()
    q_nav = q / q.iloc[0]
    qm = metrics(q_nav)

    rows = []
    for c in CONCENTRATIONS:
        nav = base_navs[c]
        for L in LEVERAGES:
            m = metrics(lever(nav, L))
            if m:
                m.update({"持股数": c, "杠杆": L})
                rows.append(m)
    out = {"QQQ": qm, "grid": rows,
           "window": f"{base_navs[CONCENTRATIONS[0]].index[0].date()} → {base_navs[CONCENTRATIONS[0]].index[-1].date()}"}
    json.dump(out, open(ROOT / "qqq_leverage_grid.json", "w"), ensure_ascii=False, indent=2)
    return out


def main():
    print("=" * 66)
    print("集中度 × 杠杆 扫描 — 找你能睡着觉的激进度")
    print("=" * 66)
    r = run()
    if not r:
        return
    q = r["QQQ"]
    print(f"  区间: {r['window']}")
    print(f"  ★ QQQ 基准: 年化 {q['cagr_%']}% · 夏普 {q['sharpe']} · 最大回撤 {q['max_dd_%']}%\n")
    print(f"  {'持股':>4}{'杠杆':>6}{'年化':>8}{'夏普':>7}{'最大回撤':>10}  vs QQQ")
    for c in CONCENTRATIONS:
        for row in [x for x in r["grid"] if x["持股数"] == c]:
            beat = row["cagr_%"] - q["cagr_%"]
            far = "🔥远超" if beat >= 8 else "跑赢" if beat > 0 else "跑输"
            dd = row["max_dd_%"]
            warn = " ⚠深回撤" if dd <= -35 else ""
            print(f"  {c:>4}{row['杠杆']:>6.1f}x{row['cagr_%']:>7.1f}%{row['sharpe']:>7.2f}{dd:>9.1f}%  {far}{warn}")
        print()
    print("  诚实警告: 2年单一牛市样本。杠杆放大的是**已知**回撤;真实熊市会更深(可能×1.5~2)。")
    print("  → qqq_leverage_grid.json")


if __name__ == "__main__":
    main()
