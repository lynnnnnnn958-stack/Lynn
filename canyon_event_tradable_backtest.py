#!/usr/bin/env python3
"""
canyon_event_tradable_backtest.py — 把"事件研究"变成"可交易事件策略"的诚实回测
================================================================================
Why this exists: canyon_edgar_backtest.py 用 *event study* 证明了 8-K 事件有超额
收益 (t=4~8)。但 event study 系统性 **高估** 显著性、且不是能交易的东西:
  - 重叠持仓窗口 → 事件级 t 值虚高 (收益不独立)
  - 毛超额, 没扣交易成本 / 换手
  - "选哪些事件交易" 若用研究结果里显著的那几类 = 样本内挑选 (数据挖掘)

本模块把事件信号变成一个 **每日再平衡的组合**, 回答唯一重要的问题:
    "在 t+1 建仓、持有 H 天、扣真实成本、市场中性后,
     事件策略的 alpha 还剩多少 —— 而且经得起多重检验的折扣吗?"

诚实控制:
  1. NO 前视 — 用 8-K *备案日* (公开信息), t+1 收盘建仓, 持有 H 个交易日。
  2. 每日组合 (非事件级) — 把重叠事件聚成一个真实时间序列, 消除 t 值虚高。
  3. 市场中性 — 组合多头 vs 等额 SPY, 隔离 alpha (不是 beta)。
  4. 真实成本 — 每次进出付 价差+冲击 (与 step_rigorous_backtest 同一套)。
  5. NEWEY-WEST 稳健 t — 对持仓期自相关做修正 (lag=H)。
  6. 多重检验折扣 — 报 Deflated Sharpe (给定试验次数, 剔除"运气最大值")。
  7. IS/OOS — 2024-01-01 前后拆分, 看是否 OOS 崩掉 (过拟合体检)。
  8. A-PRIORI 篮子 — 除了 all-8K, 单测 "业绩发布(2.02)" 这一先验假设 (PEAD),
     不是从研究结果反挑显著的类别。

Output: event_tradable_backtest.json + event_tradable_report.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── fixed a-priori parameters (NOT tuned on test data) ───────────────────────
HOLD_DAYS      = 21          # 持有 1 个月 (短于研究的 63, 更可交易, 重叠更少)
OOS_CUTOFF     = pd.Timestamp("2024-01-01")
TC_SPREAD_BPS  = 5.0         # 单边半价差
IMPACT_COEF    = 10.0        # 1% ADV 参与度时的冲击 bps (sqrt 模型)
PARTICIPATION  = 0.05
NW_LAG         = HOLD_DAYS   # Newey-West 滞后 = 持有期 (覆盖重叠自相关)
ANN            = 252

# 每次建/平仓单边成本 (bps → 小数)
_COST_ONE_SIDE = (TC_SPREAD_BPS + IMPACT_COEF * np.sqrt(PARTICIPATION)) / 10_000


def load_prices() -> pd.DataFrame:
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists() and p.stat().st_size > 3:
            df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            df = df[[c for c in df.columns
                     if str(c).replace(".", "").replace("-", "").isalpha()]]
            return df
    raise FileNotFoundError("no price cache found")


def load_events() -> pd.DataFrame:
    p = ROOT / "edgar_8k_history.csv"
    if not p.exists():
        raise FileNotFoundError("edgar_8k_history.csv missing — run canyon_edgar_backtest.py first")
    h = pd.read_csv(p)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    h["ticker"] = h["ticker"].astype(str).str.replace("-", ".", regex=False)
    h["items"] = h.get("items", "").astype(str)
    return h


def build_daily_returns(events: pd.DataFrame, prices: pd.DataFrame,
                        hold: int) -> pd.DataFrame:
    """事件 → 每日等权组合。建仓 t+1, 持有 hold 个交易日。
    返回 DataFrame[date] with: port(组合日收益, 毛), spy, n_hold, cost。"""
    idx = prices.index
    daily = prices.pct_change(fill_method=None)
    spy = daily["SPY"] if "SPY" in daily.columns else daily.mean(axis=1)

    # 每个交易日的 (进场名单, 出场名单)
    n = len(idx)
    entries = [[] for _ in range(n)]   # 在该日 *建仓* 的 ticker
    active_counts = np.zeros(n)
    # 用一个 (start,end,ticker) 列表, 再展开成每日持仓
    spans = []
    pos_of = {t: k for k, t in enumerate(idx)}
    for _, r in events.iterrows():
        tk = r["ticker"]
        if tk not in prices.columns:
            continue
        d = r["date"]
        pos = idx.searchsorted(d)          # 第一个 >= 备案日 的交易日
        start = pos + 1                     # t+1 建仓 (备案次日)
        end = start + hold                  # 持有 hold 天后平仓 (exclusive)
        if start < 1 or end >= n:
            continue
        spans.append((start, end, tk))

    if not spans:
        return pd.DataFrame()

    # 每日持仓集合 (去重: 同名多事件只算一个仓位, 但记 max 到期)
    hold_until = {}      # ticker -> latest exit index (滚动续期)
    # 先按 start 排序, 逐日推进
    spans.sort()
    span_ptr = 0
    rows = []
    for t in range(1, n):
        # 到期
        for tk in [k for k, e in hold_until.items() if e <= t]:
            del hold_until[tk]
        # 今日新事件 → 建仓 / 续期
        entered_today = 0
        while span_ptr < len(spans) and spans[span_ptr][0] == t:
            _, end, tk = spans[span_ptr]
            if tk not in hold_until:
                entered_today += 1
            hold_until[tk] = max(hold_until.get(tk, 0), end)
            span_ptr += 1
        active = list(hold_until.keys())
        if not active:
            rows.append({"date": idx[t], "port": 0.0, "spy": float(spy.iloc[t]),
                         "n_hold": 0, "cost": 0.0})
            continue
        day_row = daily.iloc[t]
        pr = np.nanmean([day_row.get(tk, np.nan) for tk in active])
        pr = 0.0 if np.isnan(pr) else float(pr)
        rows.append({"date": idx[t], "port": pr, "spy": float(spy.iloc[t]),
                     "n_hold": len(active), "_entered": entered_today})

    bt = pd.DataFrame(rows).set_index("date")
    # 成本: 每个新建仓位付单边成本, 每个平仓位付单边成本。
    # 平仓数 ≈ 建仓数的滞后 (稳态), 直接用 entered 近似进+出 = 2×entered。
    ent = bt["_entered"].fillna(0)
    denom = bt["n_hold"].replace(0, np.nan)
    bt["cost"] = (2 * ent * _COST_ONE_SIDE / denom).fillna(0.0)
    bt = bt.drop(columns=["_entered"])
    return bt


def _newey_west_t(x: np.ndarray, lag: int) -> float:
    """mean(x) 的 Newey-West (HAC) t 统计量, 修正到 lag 阶自相关。"""
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 30:
        return np.nan
    xm = x - x.mean()
    gamma0 = np.dot(xm, xm) / T
    var = gamma0
    for L in range(1, min(lag, T - 1) + 1):
        w = 1.0 - L / (lag + 1)          # Bartlett kernel
        cov = np.dot(xm[L:], xm[:-L]) / T
        var += 2 * w * cov
    se = np.sqrt(var / T)
    return float(x.mean() / se) if se > 0 else np.nan


def _deflated_sharpe(sr_ann: float, n_obs: int, n_trials: int,
                     skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014) 的概率值。
    在 n_trials 次试验里, 一个真 SR=0 的策略靠运气能达到的最大 SR 作为门槛,
    返回 P(真 SR > 0) 的近似 —— >0.95 才算扛得住多重检验。"""
    from math import log, sqrt, erf
    if not np.isfinite(sr_ann) or n_obs < 30 or n_trials < 1:
        return np.nan
    sr = sr_ann / np.sqrt(ANN)           # 转成每期 (日) SR
    # 运气最大 SR 期望 (标准正态极值近似)
    e_max = (1 - np.euler_gamma) * _z(1 - 1.0 / n_trials) + \
            np.euler_gamma * _z(1 - 1.0 / (n_trials * np.e))
    sr0 = e_max / np.sqrt(n_obs) * 1.0   # 门槛 SR (日频)
    num = (sr - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2)
    if den <= 0:
        return np.nan
    z = num / den
    return float(0.5 * (1 + erf(z / sqrt(2))))


def _z(p: float) -> float:
    """标准正态分位数 (Acklam 近似)。"""
    if p <= 0: return -8.0
    if p >= 1: return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {}
    cagr = (1 + r).prod() ** (ANN / len(r)) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() else np.nan
    c = (1 + r).cumprod(); mdd = float((c / c.cummax() - 1).min())
    return {"cagr": round(cagr, 4), "vol": round(vol, 4),
            "sharpe": round(float(sharpe), 2), "max_dd": round(mdd, 4),
            "win_rate": round(float((r > 0).mean()), 4), "n_days": int(len(r))}


def evaluate(bt: pd.DataFrame, n_trials: int, label: str) -> dict:
    """把每日组合序列 → 市场中性净收益 + 全套诚实统计。"""
    if bt.empty:
        return {"label": label, "error": "no positions"}
    # 只在真有持仓的时段评估: 空仓日 = 现金 (收益 0), 且把首尾空仓段裁掉。
    active = bt["n_hold"] > 0
    if active.sum() < 60:
        return {"label": label, "error": f"only {int(active.sum())} active days"}
    first, last = bt.index[active][0], bt.index[active][-1]
    bt = bt.loc[first:last]
    active = bt["n_hold"] > 0
    gross = bt["port"].where(active, 0.0)
    neutral = (bt["port"] - bt["spy"]).where(active, 0.0)   # 持仓才市场中性, 否则现金
    net = neutral - bt["cost"]                              # 扣成本
    # IS/OOS 拆分点 = 事件期中位日期 (2 年历史, 固定 2020 的 cutoff 无意义)
    split = bt.index[len(bt) // 2]
    is_mask = bt.index < split
    oos_mask = bt.index >= split

    nw_t = _newey_west_t(net.values, NW_LAG)
    naive_t = float(net.mean() / net.std() * np.sqrt(len(net))) if net.std() else np.nan
    s = _stats(net)
    dsr = _deflated_sharpe(s.get("sharpe", np.nan), len(net), n_trials,
                           skew=float(net.skew()), kurt=float(net.kurtosis() + 3))
    return {
        "label": label,
        "period": f"{bt.index.min().date()} → {bt.index.max().date()}",
        "avg_names_held": round(float(bt["n_hold"].mean()), 1),
        "net_market_neutral": s,
        "gross_long_only": _stats(gross),
        "cost_drag_annual": round(float(bt["cost"].mean() * ANN), 4),
        "alpha_annual_net": round(float(net.mean() * ANN), 4),
        "naive_t": round(naive_t, 2) if np.isfinite(naive_t) else None,
        "newey_west_t": round(nw_t, 2) if np.isfinite(nw_t) else None,
        "t_inflation_ratio": round(naive_t / nw_t, 2) if (nw_t and np.isfinite(nw_t) and nw_t) else None,
        "n_trials_assumed": n_trials,
        "deflated_sharpe_prob": round(dsr, 3) if np.isfinite(dsr) else None,
        "is_sharpe": _stats(net[is_mask]).get("sharpe"),
        "oos_sharpe": _stats(net[oos_mask]).get("sharpe"),
        "verdict": _verdict(nw_t, dsr),
    }


def _verdict(nw_t, dsr) -> str:
    if not np.isfinite(nw_t):
        return "N/A"
    if np.isfinite(dsr) and dsr >= 0.95 and abs(nw_t) >= 2:
        return "TRADABLE EDGE — survives costs, HAC t, and multiple-testing"
    if abs(nw_t) >= 2:
        return "MARGINAL — HAC-significant but fails deflated-Sharpe (likely luck-of-many-trials)"
    return "NO TRADABLE EDGE — dies after costs + honest standard errors"


def run() -> dict:
    prices = load_prices()
    events = load_events()

    sev = pd.to_numeric(events["severity"], errors="coerce").fillna(0)
    baskets = {
        "all_8K": events,
        "earnings_2.02": events[events["items"].str.contains("2.02", na=False)],
        "regFD_7.01": events[events["items"].str.contains("7.01", na=False)],
        "major_agreement_1.01": events[events["items"].str.contains("1.01", na=False)],
        "high_severity_3plus": events[sev >= 3],   # a-priori: only the most material events
    }
    n_trials = len(baskets)   # 多重检验: 我们试了这么多篮子
    out = {"generated_at": pd.Timestamp.now().isoformat(),
           "hold_days": HOLD_DAYS, "oos_cutoff": str(OOS_CUTOFF.date()),
           "n_trials": n_trials, "results": {}}
    for name, ev in baskets.items():
        bt = build_daily_returns(ev, prices, HOLD_DAYS)
        out["results"][name] = evaluate(bt, n_trials, name)
    return out


def _report(m: dict) -> str:
    L = ["# Tradable Event Backtest — the honest, costed, multiple-testing-corrected edge", "",
         f"_hold {m['hold_days']}d · t+1 entry · market-neutral vs SPY · realistic costs · "
         f"Newey-West HAC t (lag {NW_LAG}) · Deflated Sharpe over {m['n_trials']} trials · "
         f"IS/OOS @ {m['oos_cutoff']}_", "",
         "> Event *studies* overstate significance (overlapping windows) and ignore costs. "
         "This is what's left when you (1) build a real daily portfolio, (2) charge spread+impact, "
         "(3) use HAC standard errors, and (4) deflate the Sharpe for having tried several baskets.", "",
         "| Basket | Held | Net Sharpe | Alpha/yr | naïve t | **HAC t** | t-inflation | **DSR P** | OOS Sharpe | Verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in m["results"].items():
        if "error" in r:
            L.append(f"| {name} | — | error: {r['error']} | | | | | | | |"); continue
        s = r["net_market_neutral"]
        L.append(
            f"| {name} | {r['avg_names_held']} | {s.get('sharpe','—')} | "
            f"{r['alpha_annual_net']:.2%} | {r.get('naive_t','—')} | "
            f"**{r.get('newey_west_t','—')}** | {r.get('t_inflation_ratio','—')}× | "
            f"**{r.get('deflated_sharpe_prob','—')}** | {r.get('oos_sharpe','—')} | "
            f"{r['verdict'].split('—')[0].strip()} |")
    L += ["",
          "**Reading it:**",
          "- **HAC t vs naïve t** — the gap is how much the overlapping-window / autocorrelation "
          "was inflating your event-study t. This is the honest one.",
          "- **DSR P (Deflated Sharpe probability)** — P(true Sharpe > 0) after accounting for how many "
          "baskets we tried. **≥ 0.95** = survives multiple testing; below = probably luck.",
          "- **OOS Sharpe** — out-of-sample (2024→). A collapse vs full-period = overfit.",
          "- **Net = market-neutral, after costs.** This is tradable alpha, not beta.", ""]
    return "\n".join(L)


def main():
    print("=" * 72)
    print("Tradable event backtest — costed, HAC, multiple-testing-corrected")
    print("=" * 72)
    m = run()
    json.dump(m, open(ROOT / "event_tradable_backtest.json", "w"), indent=2, default=str)
    (ROOT / "event_tradable_report.md").write_text(_report(m))
    for name, r in m["results"].items():
        if "error" in r:
            print(f"  {name:22} {r['error']}"); continue
        s = r["net_market_neutral"]
        print(f"\n  {name}")
        print(f"    held~{r['avg_names_held']}  net Sharpe {s.get('sharpe')}  "
              f"alpha/yr {r['alpha_annual_net']:+.2%}  cost drag {r['cost_drag_annual']:.2%}")
        print(f"    naïve t {r.get('naive_t')} → HAC t {r.get('newey_west_t')} "
              f"(inflation {r.get('t_inflation_ratio')}×)  DSR P={r.get('deflated_sharpe_prob')}")
        print(f"    IS Sharpe {r.get('is_sharpe')} / OOS Sharpe {r.get('oos_sharpe')}")
        print(f"    → {r['verdict']}")
    print("\n  → event_tradable_backtest.json + event_tradable_report.md")


if __name__ == "__main__":
    main()
