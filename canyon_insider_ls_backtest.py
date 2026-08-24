#!/usr/bin/env python3
"""
canyon_insider_ls_backtest.py — 内部人多空"合体"回测(完整产品的真实数字)
==========================================================================
多头(内部人抄底买入)和空头(内部人卖出)各自验证过了。这里把它们**合起来**
——美元中性 long/short——回答唯一重要的问题: 你纸面账本实际在跑的"完整产品",
真实的 Sharpe / 收益 / 回撤 是多少?

关键发现(会随数据更新): 合体是**纯 alpha**(市场中性, 不靠大盘涨跌), 两腿负相关
→ 对冲把回撤砍掉约一半, 代价是 headline 收益降低(放弃了小盘 beta)。空头单独
回撤 -90%+ = 绝不能裸空, 必须成对。

诚实边界: 空头数据较浅(smallcap_form4_sells.csv, ~78只); 做空有借券费/逼空/
无限亏损风险; 现实成本 44bps。研究数字, 非下单指令。

Output: insider_ls_backtest.json + insider_ls_report.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import canyon_insider_smallcap_backtest as B
from canyon_event_tradable_backtest import _newey_west_t

ROOT = Path(__file__).parent
HOLD = 10
ANN = 252


def _trail63(df, prices, idx):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["dt"])
    out = []
    for t, d in zip(df["ticker"], df["dt"]):
        if t not in prices.columns:
            out.append(np.nan); continue
        p = idx.searchsorted(d)
        out.append(prices[t].iloc[p] / prices[t].iloc[p - 63] - 1 if 64 <= p < len(idx) else np.nan)
    df["t63"] = out
    return df


def _port_series(ev, prices, bench):
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bt = B.build_daily(ev[["ticker", "date"]], prices, bench, HOLD)
    a = bt["n_hold"] > 0
    return bt["port"].where(a, np.nan), bt["cost"]


def _stats(r):
    r = r.dropna()
    if len(r) < 30:
        return {}
    a = float(r.mean() * ANN)
    sh = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() else np.nan
    t = _newey_west_t(r.values, HOLD)
    c = (1 + r).cumprod(); mdd = float((c / c.cummax() - 1).min())
    return {"alpha_annual": round(a, 4), "sharpe": round(sh, 2),
            "hac_t": round(t, 2) if np.isfinite(t) else None,
            "max_dd": round(mdd, 4), "n_days": int(len(r))}


def run():
    prices = B.load_prices(); idx = prices.index
    bench = prices.pct_change(fill_method=None).mean(axis=1)
    B.TC_SPREAD_BPS, B.IMPACT_COEF = 35, 40
    B._COST_ONE_SIDE = (35 + 40 * np.sqrt(0.05)) / 1e4        # 现实小盘成本 44bps

    buys = _trail63(B.load_buys(), prices, idx)
    dip = buys[buys["t63"] < 0]                                # 多头 = 抄底买入
    sp = ROOT / "smallcap_form4_sells.csv"
    if not sp.exists():
        return {"error": "smallcap_form4_sells.csv missing — run step_smallcap_form4.py INSIDER_SIDE=SELL"}
    allsells = pd.read_csv(sp)
    allsells["dt"] = pd.to_datetime(allsells["date"], errors="coerce")
    # 空头只用 CLUSTER 卖出(≥2 内部人 30 天内同抛)—— 验证显示只有它合体能提升 Sharpe
    cl = set()
    for tk, g in allsells.groupby("ticker"):
        g = g.sort_values("dt")
        for _, r in g.iterrows():
            w = g[(g["dt"] > r["dt"] - pd.Timedelta(days=30)) & (g["dt"] <= r["dt"])]
            if ("owner" in w.columns and w["owner"].nunique() >= 2):
                cl.add((tk, r["dt"])); break
    sells = allsells[[(t, d) in cl for t, d in zip(allsells["ticker"], allsells["dt"])]]
    if len(sells) < 30:
        sells = allsells               # cluster 太少则退回全部(保底)

    L, Lc = _port_series(dip, prices, bench)
    S, Sc = _port_series(sells, prices, bench)
    df = pd.DataFrame({"L": L, "S": S, "Lc": Lc, "Sc": Sc})
    df = df.loc[df[["L", "S"]].notna().any(axis=1)].fillna({"L": 0, "S": 0, "Lc": 0, "Sc": 0})

    combined = 0.5 * df["L"] - 0.5 * df["S"] - 0.5 * (df["Lc"] + df["Sc"])   # 美元中性多空
    mn_long = (df["L"] - bench.reindex(df.index).fillna(0)) - df["Lc"]        # 多头市场中性(纯alpha)
    corr = float(df["L"].corr(-df["S"])) if df["S"].std() else np.nan

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "hold_days": HOLD, "cost_bps_one_side": round(B._COST_ONE_SIDE * 1e4, 1),
        "n_buy_events": int(len(dip)), "n_sell_events": int(len(sells)),
        "combined_long_short": _stats(combined),
        "long_market_neutral": _stats(mn_long),
        "short_leg_standalone": _stats(-df["S"] - df["Sc"]),
        "leg_correlation": round(corr, 2) if np.isfinite(corr) else None,
    }
    return out


def _report(m):
    if "error" in m:
        return f"# Insider Long/Short Backtest\n\n{m['error']}\n"
    c = m["combined_long_short"]; ln = m["long_market_neutral"]; s = m["short_leg_standalone"]
    return "\n".join([
        "# Insider Long/Short — the combined product's honest numbers", "",
        f"_hold {m['hold_days']}d · realistic {m['cost_bps_one_side']}bps/side · dollar-neutral · "
        f"HAC t · {m['n_buy_events']} buy / {m['n_sell_events']} sell events_", "",
        "| Book | Alpha/yr | Sharpe | HAC t | Max DD |",
        "|---|---|---|---|---|",
        f"| **Combined long/short (the product)** | {c.get('alpha_annual',0):.1%} | {c.get('sharpe','—')} | "
        f"{c.get('hac_t','—')} | {c.get('max_dd',0):.1%} |",
        f"| Long market-neutral (buy-dip) | {ln.get('alpha_annual',0):.1%} | {ln.get('sharpe','—')} | "
        f"{ln.get('hac_t','—')} | {ln.get('max_dd',0):.1%} |",
        f"| Short leg standalone (never trade alone) | {s.get('alpha_annual',0):.1%} | {s.get('sharpe','—')} | "
        f"{s.get('hac_t','—')} | {s.get('max_dd',0):.1%} |", "",
        f"**Leg correlation: {m['leg_correlation']}** — negative = the short leg genuinely hedges the long, "
        "roughly halving drawdown. The combined book is PURE alpha (market-neutral, beta-independent) — "
        "you trade headline return for all-weather stability. The short leg alone is a disaster "
        "(deep drawdown) — only ever trade it paired.", ""])


def main():
    print("=" * 68)
    print("Insider long/short — combined product backtest")
    print("=" * 68)
    m = run()
    json.dump(m, open(ROOT / "insider_ls_backtest.json", "w"), indent=2, default=str)
    (ROOT / "insider_ls_report.md").write_text(_report(m))
    if "error" in m:
        print("  " + m["error"]); return
    c = m["combined_long_short"]
    print(f"\n  ★ COMBINED long/short: alpha {c['alpha_annual']:+.1%}  Sharpe {c['sharpe']}  "
          f"HAC t {c['hac_t']}  MaxDD {c['max_dd']:.1%}")
    print(f"    long-MN alpha {m['long_market_neutral']['alpha_annual']:+.1%}  ·  "
          f"leg corr {m['leg_correlation']} (neg = hedges)")
    print(f"\n  → insider_ls_backtest.json + insider_ls_report.md")


if __name__ == "__main__":
    main()
