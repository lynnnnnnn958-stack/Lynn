#!/usr/bin/env python3
"""
canyon_insider_sizing.py — 右尾彩票 edge 的仓位与风控引擎
==========================================================
内部人抄底 edge 是"正期望的彩票": HAC t=3.7 但胜率仅 ~45%, 且去掉最好 1% 交易日
就归零 —— 收益全靠极少数暴涨的名字。这种 edge 用错仓位,再真也会爆仓。

设计原则(全部由脆弱性测试推导, 不是拍脑袋):
  1. 多买小注 —— 你不知道哪只是那个爆拉的, 所以要持很多只、每只很小。
  2. 永不重押 —— 单只硬上限 MAX_PER_NAME_PCT, 信号少宁可留现金也不集中。
  3. 按流动性封顶 —— 单只 ≤ LIQ_CAP × 该股20日平均成交额, 保证真实成交、
     不把价格冲飞(这也决定了策略的真实容量上限)。
  4. 连亏不撤 —— 右尾策略的反弹恰恰来自暴跌后, 回撤后停手 = 错过反弹。所以
     **不设回撤熔断**, 而是从一开始就按"能扛住长期连亏"来定注码。
  5. 等风险(可选) —— 按波动率反向加权, 免得一只超高波动股主导整个组合风险。

用法: python3 canyon_insider_sizing.py            # 打印当前 watchlist 的建议仓位
      被 canyon_insider_paper.py 调用做真实注码。
输出: insider_sizing_today.csv
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
WATCH = ROOT / "insider_scan_today.csv"
OUT = ROOT / "insider_sizing_today.csv"

# ── 风控参数(可调, 但默认值都为"活下来"而非"最大化") ──────────────────────
SLEEVE_CAPITAL   = 100_000    # 分配给这个策略的资金(纸面默认 $100k)
TARGET_POSITIONS = 20         # 目标持仓数: 多买"彩票"才能可靠中到右尾
MAX_PER_NAME_PCT = 0.08       # 单只硬上限 = 8% of sleeve(永不重押)
LIQ_CAP_PCT      = 0.015      # 单只 ≤ 1.5% × 20日平均成交额(流动性/容量约束)
VOL_TARGET       = True       # True=按波动率反向加权(等风险); False=纯等权
MIN_NAME_USD     = 500        # 太小的注不值得下(成本占比过高)


def _load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _bars(tickers, days=40):
    """Daily bars per ticker from Alpaca → {tk: DataFrame(close,volume)}. For
    20-day avg dollar volume (liquidity) and realized vol (risk weighting)."""
    _load_env()
    key = os.environ.get("ALPACA_KEY_ID"); sec = os.environ.get("ALPACA_KEY_SECRET")
    if not key or not sec:
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        cli = StockHistoricalDataClient(key, sec)
        req = StockBarsRequest(symbol_or_symbols=list(tickers), timeframe=TimeFrame.Day,
                               start=datetime.now() - timedelta(days=days * 2))
        df = cli.get_stock_bars(req).df
        out = {}
        for tk in tickers:
            try:
                out[tk] = df.loc[tk][["close", "volume"]].tail(days)
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"  Alpaca bars unavailable ({str(e)[:40]}) — sizing on equal-weight only")
        return {}


def size_positions(watch: pd.DataFrame, sleeve=SLEEVE_CAPITAL) -> pd.DataFrame:
    """Return watch + a `size_usd` column and the binding constraint per name."""
    if watch is None or watch.empty:
        return watch
    w = watch.copy()
    tks = list(w["ticker"].astype(str))
    bars = _bars(tks)

    adv, vol = {}, {}
    for tk in tks:
        b = bars.get(tk)
        if b is not None and len(b) >= 15:
            adv[tk] = float((b["close"] * b["volume"]).tail(20).mean())      # 20日平均成交额($)
            r = b["close"].pct_change().dropna()
            vol[tk] = float(r.tail(20).std()) if len(r) >= 5 else np.nan

    # 1) 基础注码: 等权 或 按波动率反向(等风险)
    n = len(w)
    base = sleeve / max(TARGET_POSITIONS, n)          # 信号≤目标数时才满仓, 否则每只更小
    if VOL_TARGET and vol:
        inv = {tk: (1.0 / vol[tk] if vol.get(tk) and vol[tk] > 0 else np.nan) for tk in tks}
        med = np.nanmedian([x for x in inv.values() if x and np.isfinite(x)]) or 1.0
        w["_rawsize"] = [base * (min(inv.get(tk, med) / med, 2.0) if inv.get(tk) and np.isfinite(inv.get(tk)) else 1.0)
                         for tk in tks]
    else:
        w["_rawsize"] = base

    # 2) 逐名封顶: min(基础, 单只上限, 流动性上限)
    cap_name = sleeve * MAX_PER_NAME_PCT
    sizes, binds = [], []
    for tk, raw in zip(tks, w["_rawsize"]):
        liq = LIQ_CAP_PCT * adv[tk] if tk in adv else np.inf
        candidates = {"base/equal": raw, "per-name cap": cap_name, "liquidity": liq}
        s = min(candidates.values())
        bind = min(candidates, key=candidates.get)
        if s < MIN_NAME_USD:
            s, bind = 0.0, "too-small/skip"
        sizes.append(round(s, 0)); binds.append(bind)
    sizes = np.array(sizes, dtype=float)
    # 归一化: 总额绝不超过 sleeve(信号多时按比例缩小; 信号少时留现金, 不放大)
    tot = sizes.sum()
    if tot > sleeve and tot > 0:
        sizes = sizes * (sleeve / tot)
        binds = [b if b != "base/equal" else "sleeve-normalized" for b in binds]
    w["size_usd"] = np.round(sizes, 0)
    w["binding_constraint"] = binds
    w["adv_20d_usd"] = [int(adv.get(tk, 0)) for tk in tks]
    w = w.drop(columns=["_rawsize"])
    return w


def main():
    print("=" * 66)
    print("Insider dip-edge · position sizing & risk (right-tail lottery)")
    print("=" * 66)
    if not WATCH.exists() or WATCH.stat().st_size < 20:
        print("  no watchlist yet — run canyon_insider_scanner.py first"); return
    w = pd.read_csv(WATCH)
    w = size_positions(w)
    w.to_csv(OUT, index=False)
    dep = w["size_usd"].sum()
    live = w[w["size_usd"] > 0]
    print(f"\n  Sleeve ${SLEEVE_CAPITAL:,} · target {TARGET_POSITIONS} names · "
          f"per-name cap {MAX_PER_NAME_PCT:.0%} · liquidity cap {LIQ_CAP_PCT:.1%} of 20d ADV\n")
    for _, r in live.sort_values("size_usd", ascending=False).head(25).iterrows():
        tag = "★dip" if r.get("dip") else ""
        print(f"  {str(r['ticker']):6} ${int(r['size_usd']):>6,}  ({r['binding_constraint']:>13})  "
              f"ADV ${int(r['adv_20d_usd']):>12,}  {tag}")
    print(f"\n  deployed ${int(dep):,} of ${SLEEVE_CAPITAL:,}  ({dep/SLEEVE_CAPITAL:.0%}) · "
          f"cash ${int(SLEEVE_CAPITAL-dep):,} · {len(live)} names funded")
    print(f"\n  Risk rules baked in:")
    print(f"   · never > {MAX_PER_NAME_PCT:.0%} in one name (you don't know which pays off)")
    print(f"   · never > {LIQ_CAP_PCT:.1%} of a stock's daily volume (keeps fills real → caps capacity)")
    print(f"   · few signals → hold cash, don't concentrate")
    print(f"   · NO drawdown circuit-breaker: the bounce comes after the fall — pausing misses it")
    print(f"\n  → {OUT.name}")


if __name__ == "__main__":
    main()
