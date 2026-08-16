#!/usr/bin/env python3
"""
canyon_insider_smallcap_backtest.py — 小盘 Form 4 内部人买入的诚实可交易回测
============================================================================
假设 (a-priori, 文献): 内部人 *开放市场买入* 预测正收益, 在 **小盘** 最强
(覆盖少、机构容量做不了)。cluster buy (多个内部人同期买) 信号最强。

用与 canyon_event_tradable_backtest 同一套诚实控制, 但:
  - 基准 = 小盘 **等权指数** (size-neutral, alpha 不是小盘 beta)
  - 成本更高 (小盘价差宽): 单边 20bps 价差 + 冲击
  - 内部人持仓周期更长 → 同时测 hold=21 / 63 天
  - 篮子 (a-priori, 不是从结果反挑):
      all_buys · cxo_buys(CEO/CFO/总裁) · cluster_buys(30天内≥2个不同内部人)
      · large_buys(金额≥$100k)

诚实控制: t+1 建仓 · 市场中性 · 真实成本 · Newey-West HAC t ·
Deflated Sharpe(多重检验) · IS/OOS。

Output: insider_smallcap_backtest.json + insider_smallcap_report.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# 复用诚实统计工具
from canyon_event_tradable_backtest import _newey_west_t, _deflated_sharpe, _stats

ROOT = Path(__file__).parent
ANN = 252
HOLDS = [21, 63]
CLUSTER_WIN = 30          # 天: cluster buy 判定窗口
LARGE_USD = 100_000       # 大额买入门槛
# 小盘真实成本 (bps) — 明显高于大盘
TC_SPREAD_BPS = 20.0
IMPACT_COEF = 25.0
PARTICIPATION = 0.05
_COST_ONE_SIDE = (TC_SPREAD_BPS + IMPACT_COEF * np.sqrt(PARTICIPATION)) / 10_000


def load_prices():
    df = pd.read_csv(ROOT / "smallcap_price_history.csv", index_col=0, parse_dates=True)
    return df.sort_index()


def load_buys():
    p = ROOT / "smallcap_form4_buys_full.csv"      # prefer deep-history file
    if not p.exists() or p.stat().st_size < 100:
        p = ROOT / "smallcap_form4_buys.csv"
    if not p.exists():
        raise FileNotFoundError("run step_smallcap_form4.py first")
    print(f"  buys source: {p.name}")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("role_cxo", "value"):
        if c not in df.columns:
            df[c] = 0
    return df


def mark_clusters(buys: pd.DataFrame) -> pd.DataFrame:
    """标记 cluster buy: 同 ticker 30 天内 ≥2 个不同内部人买入。
    cluster 事件挂在触发日 (第2个不同内部人买入的那天)。"""
    ev = []
    for tk, g in buys.groupby("ticker"):
        g = g.sort_values("date")
        for i, r in g.iterrows():
            win = g[(g["date"] > r["date"] - pd.Timedelta(days=CLUSTER_WIN)) &
                    (g["date"] <= r["date"])]
            owners = win["owner"].nunique() if "owner" in win.columns else len(win)
            if owners >= 2:
                ev.append({"ticker": tk, "date": r["date"]})
    return pd.DataFrame(ev).drop_duplicates()


def build_daily(events, prices, bench, hold):
    """事件 → 每日等权组合 (t+1 建仓, 持有 hold 天)。返回 port/bench/n_hold/cost。"""
    idx = prices.index
    daily = prices.pct_change(fill_method=None)
    n = len(idx)
    spans = []
    for _, r in events.iterrows():
        tk = r["ticker"]
        if tk not in prices.columns:
            continue
        pos = idx.searchsorted(r["date"])
        start = pos + 1
        end = start + hold
        if start < 1 or end >= n:
            continue
        spans.append((start, end, tk))
    if not spans:
        return pd.DataFrame()
    spans.sort()
    ptr = 0
    hold_until = {}
    rows = []
    for t in range(1, n):
        for tk in [k for k, e in hold_until.items() if e <= t]:
            del hold_until[tk]
        entered = 0
        while ptr < len(spans) and spans[ptr][0] == t:
            _, end, tk = spans[ptr]
            if tk not in hold_until:
                entered += 1
            hold_until[tk] = max(hold_until.get(tk, 0), end)
            ptr += 1
        active = list(hold_until.keys())
        if not active:
            rows.append({"date": idx[t], "port": 0.0, "bench": float(bench.iloc[t]),
                         "n_hold": 0, "_entered": 0})
            continue
        day = daily.iloc[t]
        pr = np.nanmean([day.get(tk, np.nan) for tk in active])
        rows.append({"date": idx[t], "port": 0.0 if np.isnan(pr) else float(pr),
                     "bench": float(bench.iloc[t]), "n_hold": len(active), "_entered": entered})
    bt = pd.DataFrame(rows).set_index("date")
    ent = bt["_entered"].fillna(0)
    denom = bt["n_hold"].replace(0, np.nan)
    bt["cost"] = (2 * ent * _COST_ONE_SIDE / denom).fillna(0.0)
    return bt.drop(columns=["_entered"])


def evaluate(bt, n_trials, label):
    if bt.empty:
        return {"label": label, "error": "no positions"}
    active = bt["n_hold"] > 0
    if active.sum() < 60:
        return {"label": label, "error": f"only {int(active.sum())} active days"}
    first, last = bt.index[active][0], bt.index[active][-1]
    bt = bt.loc[first:last]
    active = bt["n_hold"] > 0
    neutral = (bt["port"] - bt["bench"]).where(active, 0.0)   # 市场中性 vs 小盘等权
    net = neutral - bt["cost"]
    split = bt.index[len(bt) // 2]
    nw_t = _newey_west_t(net.values, max(HOLDS))
    naive_t = float(net.mean() / net.std() * np.sqrt(len(net))) if net.std() else np.nan
    s = _stats(net)
    dsr = _deflated_sharpe(s.get("sharpe", np.nan), len(net), n_trials,
                           skew=float(net.skew()), kurt=float(net.kurtosis() + 3))
    return {
        "label": label,
        "period": f"{bt.index.min().date()} → {bt.index.max().date()}",
        "avg_names_held": round(float(bt["n_hold"].mean()), 1),
        "net_market_neutral": s,
        "alpha_annual_net": round(float(net.mean() * ANN), 4),
        "cost_drag_annual": round(float(bt["cost"].mean() * ANN), 4),
        "naive_t": round(naive_t, 2) if np.isfinite(naive_t) else None,
        "newey_west_t": round(nw_t, 2) if np.isfinite(nw_t) else None,
        "deflated_sharpe_prob": round(dsr, 3) if np.isfinite(dsr) else None,
        "is_sharpe": _stats(net[bt.index < split]).get("sharpe"),
        "oos_sharpe": _stats(net[bt.index >= split]).get("sharpe"),
        "verdict": _verdict(nw_t, dsr),
    }


def _verdict(nw_t, dsr):
    if not np.isfinite(nw_t):
        return "N/A"
    if np.isfinite(dsr) and dsr >= 0.95 and abs(nw_t) >= 2:
        return "TRADABLE EDGE — survives costs, HAC t, multiple-testing"
    if abs(nw_t) >= 2:
        return "MARGINAL — HAC-significant but fails deflated-Sharpe"
    return "NO TRADABLE EDGE — dies after costs + honest standard errors"


def run():
    prices = load_prices()
    buys = load_buys()
    bench = prices.pct_change(fill_method=None).mean(axis=1)   # 小盘等权基准

    baskets = {
        "all_buys": buys[["ticker", "date"]],
        "cxo_buys": buys[buys["role_cxo"] == 1][["ticker", "date"]],
        "large_buys_100k": buys[pd.to_numeric(buys["value"], errors="coerce").fillna(0) >= LARGE_USD][["ticker", "date"]],
        "cluster_buys": mark_clusters(buys),
    }
    n_trials = len(baskets) * len(HOLDS)
    out = {"generated_at": pd.Timestamp.now().isoformat(), "holds": HOLDS,
           "n_trials": n_trials, "cost_bps_one_side": round(_COST_ONE_SIDE * 1e4, 1),
           "n_buys": int(len(buys)), "n_tickers": int(buys["ticker"].nunique()),
           "results": {}}
    for name, ev in baskets.items():
        for h in HOLDS:
            bt = build_daily(ev, prices, bench, h)
            out["results"][f"{name}_h{h}"] = evaluate(bt, n_trials, f"{name}_h{h}")
    return out


def _report(m):
    L = ["# Small-cap Insider-Buy Backtest — honest, costed, multiple-testing-corrected", "",
         f"_{m['n_buys']:,} open-market Form 4 buys · {m['n_tickers']} small-caps · "
         f"market-neutral vs equal-weight small-cap · cost {m['cost_bps_one_side']}bps/side · "
         f"Newey-West HAC t · Deflated Sharpe over {m['n_trials']} trials_", "",
         "| Basket | Hold | Held | Sharpe | Alpha/yr | HAC t | DSR P | OOS Sharpe | Verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in m["results"].items():
        if "error" in r:
            L.append(f"| {name} | | — | error: {r['error']} | | | | | |"); continue
        s = r["net_market_neutral"]
        L.append(f"| {name.rsplit('_h',1)[0]} | {name.rsplit('_h',1)[1]}d | {r['avg_names_held']} | "
                 f"{s.get('sharpe','—')} | {r['alpha_annual_net']:.2%} | **{r.get('newey_west_t','—')}** | "
                 f"**{r.get('deflated_sharpe_prob','—')}** | {r.get('oos_sharpe','—')} | "
                 f"{r['verdict'].split('—')[0].strip()} |")
    L += ["", "**DSR P ≥ 0.95 + HAC t ≥ 2 = real tradable edge.** Below = likely luck/beta.", ""]
    return "\n".join(L)


def main():
    print("=" * 72)
    print("Small-cap insider-buy backtest — costed, HAC, multiple-testing-corrected")
    print("=" * 72)
    m = run()
    json.dump(m, open(ROOT / "insider_smallcap_backtest.json", "w"), indent=2, default=str)
    (ROOT / "insider_smallcap_report.md").write_text(_report(m))
    print(f"  {m['n_buys']:,} buys · {m['n_tickers']} tickers · cost {m['cost_bps_one_side']}bps/side\n")
    for name, r in m["results"].items():
        if "error" in r:
            print(f"  {name:24} {r['error']}"); continue
        s = r["net_market_neutral"]
        print(f"  {name:24} held~{r['avg_names_held']:>5.0f}  Sharpe {s.get('sharpe'):>5}  "
              f"alpha/yr {r['alpha_annual_net']:>+7.2%}  HAC t {r.get('newey_west_t'):>5}  "
              f"DSR {r.get('deflated_sharpe_prob')}  OOS {r.get('oos_sharpe')}")
    print("\n  → insider_smallcap_backtest.json + insider_smallcap_report.md")


if __name__ == "__main__":
    main()
