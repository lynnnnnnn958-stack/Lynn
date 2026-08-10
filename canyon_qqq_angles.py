#!/usr/bin/env python3
"""
canyon_qqq_angles.py — 诚实测试: 有没有任何配置能真打赢纳指(全 PIT 去偏差)
==========================================================================
选2的核心问题: 去偏差后, 集中 / 科技聚焦 能不能把夏普拉到纳指之上?
只有夏普明显 >QQQ(~1.0) 才有真超额空间。全部用 PIT 成分股(去前视纳入偏差)。

测:
  A. 集中度: top 5 / 10 / 20 事件名(PIT)
  B. 科技聚焦: 只在 科技/通信/AI基建 板块的事件里选(PIT) × 集中10
  C. 高t item聚焦: 只用 RegFD(7.01)/业绩(2.02)/其他(8.01) 这些验证最强的 8-K
每个对着 QQQ 比 CAGR/夏普/回撤。诚实: 仍缺324只退市股价格, 残余偏差未除净。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from canyon_qqq_backtest import load_prices, load_qqq, load_events, load_pit, backtest, metrics

ROOT = Path(__file__).parent
BOOM_SECTORS = {"Technology", "Communication Services", "Utilities"}
BOOM_ALLOW = {"ETN", "EMR", "ROK", "PH", "AME", "HUBB", "PWR", "VRT", "GEV", "JCI",
              "CARR", "TT", "GE", "ANET", "CDW", "GLW", "APH", "TEL"}
HOT_ITEMS = {"7.01", "2.02", "8.01"}


def sector_map():
    d = pd.read_csv(ROOT / "alpha_scores.csv")
    return {str(r["ticker"]): str(r["sector"]) for _, r in d.iterrows()} if "sector" in d.columns else {}


def main():
    print("=" * 70)
    print("诚实测试 — 有没有配置能真打赢纳指(全 PIT 去偏差)")
    print("=" * 70)
    px = load_prices(); qqq = load_qqq(); events = load_events()
    pit_dates, pit_map = load_pit()
    start = events["date"].min() - pd.Timedelta(days=120)
    px = px[px.index >= start]
    secmap = sector_map()

    def bt(ev, hold, label):
        nav = backtest(px, qqq, ev, trend_filter=True, max_hold=hold,
                       pit_dates=pit_dates, pit_map=pit_map)
        m = metrics(nav); m["label"] = label
        return m

    results = []
    # A. 集中度(全事件)
    for h in [5, 10, 20]:
        results.append(bt(events, h, f"集中{h}只(PIT)"))
    # B. 科技聚焦(只在科技/通信/AI基建的事件里选)
    tech_ev = events[events["ticker"].astype(str).apply(
        lambda t: secmap.get(t, "") in BOOM_SECTORS or t in BOOM_ALLOW)]
    for h in [10, 20]:
        results.append(bt(tech_ev, h, f"科技聚焦·集中{h}(PIT)"))
    # C. 高t item聚焦
    if "items" in events.columns:
        hot = events[events["items"].astype(str).apply(
            lambda s: any(c in HOT_ITEMS for c in s.split("|")))]
        results.append(bt(hot, 10, "高t-item·集中10(PIT)"))

    q = qqq.reindex(px.index).ffill()
    # QQQ 对齐到 A[0] 的区间
    ref_nav = backtest(px, qqq, events, trend_filter=True, max_hold=10,
                       pit_dates=pit_dates, pit_map=pit_map)
    q2 = qqq.reindex(ref_nav.index).ffill()
    qm = metrics(q2 / q2.iloc[0])
    print(f"  ★ QQQ 基准: 年化 {qm['cagr_%']}% · 夏普 {qm['sharpe']} · 回撤 {qm['max_dd_%']}%\n")
    print(f"  {'配置':<24}{'年化':>8}{'夏普':>7}{'回撤':>9}  vs QQQ")
    winners = []
    for m in results:
        if not m:
            continue
        beat_ret = m["cagr_%"] - qm["cagr_%"]
        beat_sh = m["sharpe"] - qm["sharpe"]
        tag = "🔥真打赢" if (beat_ret > 2 and beat_sh > 0.1) else "跑赢" if beat_ret > 0 else "跑输"
        if beat_ret > 2 and beat_sh > 0.1:
            winners.append(m["label"])
        print(f"  {m['label']:<24}{m['cagr_%']:>7.1f}%{m['sharpe']:>7.2f}{m['max_dd_%']:>8.1f}%  {tag}")
    # ── 对冲/做空维度: 长事件组合 − h×QQQ(指数空头对冲, 砍回撤/提前避险) ──
    print(f"\n  --- 对冲维度: 集中10(PIT) 长腿 − 指数空头对冲 ---")
    long_ret = ref_nav.pct_change().fillna(0)
    q_ret = (q2 / q2.iloc[0]).pct_change().fillna(0)
    hedge_rows = []
    for h in [0.0, 0.5, 1.0]:
        hr = long_ret - h * q_ret
        hnav = (1 + hr).cumprod()
        m = metrics(hnav)
        label = "纯多头" if h == 0 else f"对冲{int(h*100)}%QQQ" + ("(市场中性)" if h == 1.0 else "")
        beat_sh = m["sharpe"] - qm["sharpe"]
        print(f"  {label:<22}{m['cagr_%']:>7.1f}%{m['sharpe']:>7.2f}{m['max_dd_%']:>8.1f}%  夏普{'↑' if beat_sh>0 else '↓'}{beat_sh:+.2f}")
        m["label"] = label; hedge_rows.append(m)
    # 择时对冲: 用第0层稀缺度/宏观风险信号在risk-off时挂空(简化: 用QQQ跌破50日线时对冲100%)
    q50 = (q2 / q2.iloc[0]).rolling(50).mean()
    qn = q2 / q2.iloc[0]
    risk_off = (qn < q50).reindex(long_ret.index).fillna(False)
    timed = long_ret - np.where(risk_off.values, 1.0, 0.0) * q_ret
    tnav = (1 + timed).cumprod(); tm = metrics(tnav)
    print(f"  {'择时对冲(跌破50线才空)':<20}{tm['cagr_%']:>7.1f}%{tm['sharpe']:>7.2f}{tm['max_dd_%']:>8.1f}%  夏普{tm['sharpe']-qm['sharpe']:+.2f}")

    out = {"QQQ": qm, "configs": results, "winners": winners, "hedged": hedge_rows,
           "timed_hedge": tm}
    json.dump(out, open(ROOT / "qqq_angles.json", "w"), ensure_ascii=False, indent=2)
    print("\n  真打赢纳指的配置(年化+2以上且夏普更高):", winners or "无 — 纯多头没有配置能干净打赢纳指")
    print("  → qqq_angles.json (诚实: 仍缺退市股, 残余偏差未除净)")


if __name__ == "__main__":
    main()
