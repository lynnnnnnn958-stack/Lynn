#!/usr/bin/env python3
"""
canyon_insider_calibration.py — 实盘 vs 回测 校准(闭环最后一环)
===============================================================
回测说 Sharpe~1.05 / alpha+24%。但"改了≠好了", "回测≠实盘"。这个模块在纸面账本
平仓后, 把 **真实实现的收益** 拿来和回测预期对照 —— 是骗自己还是真的。

现在(2026-08 建仓, 10天持仓)还没到期 → 报 "pending N 笔"。等平仓累积:
  · 实现胜率 / 每笔均值 / 年化 (多头 vs 空头分开)
  · vs 回测预期 (insider_ls_backtest.json) → 差多少
  · 判定: 实盘掉到不显著 → 诚实埋掉; 守住 → 第一个真凭据

Output: insider_calibration.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MIN_TRADES = 20              # 少于这么多平仓, 只报 pending(样本不足)


def run():
    bp = ROOT / "insider_paper_book.json"
    if not bp.exists():
        return {"status": "no paper book yet"}
    book = json.loads(bp.read_text())
    closed = pd.DataFrame(book.get("closed", []))
    openp = book.get("open", {})
    exp = {}
    try:
        exp = json.loads((ROOT / "insider_ls_backtest.json").read_text()).get("combined_long_short", {})
    except Exception:
        pass

    out = {"as_of": pd.Timestamp.now().isoformat(),
           "open_positions": len(openp), "closed_trades": int(len(closed)),
           "backtest_expected": {"alpha_annual": exp.get("alpha_annual"),
                                 "sharpe": exp.get("sharpe"), "max_dd": exp.get("max_dd")}}
    if len(closed) < MIN_TRADES:
        # 预估首批平仓日(建仓日 + ~14 日历日)
        eta = ""
        if openp:
            first = min(pd.Timestamp(p["entry_date"]) for p in openp.values())
            eta = (first + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        out["status"] = f"PENDING — {len(closed)}/{MIN_TRADES} closed trades; first exits ~{eta}"
        out["note"] = "10-trading-day holds; entries from 2026-08. Calibration activates once trades close."
        return out

    # 有足够平仓 → 真实校准
    r = closed["return"]
    def _leg(mask, name):
        rr = r[mask]
        if len(rr) < 5:
            return None
        return {"n": int(len(rr)), "win_rate": round(float((rr > 0).mean()), 3),
                "avg_return": round(float(rr.mean()), 4),
                "annualized_est": round(float(rr.mean()) * (252 / 10), 4)}   # 10天持仓年化
    is_short = closed.get("side", pd.Series(["LONG"] * len(closed))) == "SHORT"
    out["status"] = "LIVE"
    out["realized_all"] = _leg(pd.Series([True] * len(closed)), "all")
    out["realized_long"] = _leg(~is_short, "long")
    out["realized_short"] = _leg(is_short, "short")
    live_ann = out["realized_all"]["annualized_est"] if out["realized_all"] else None
    bt_ann = exp.get("alpha_annual")
    if live_ann is not None and bt_ann:
        out["live_vs_backtest_ratio"] = round(live_ann / bt_ann, 2)
        out["verdict"] = ("HOLDING — live in line with backtest" if live_ann > 0.5 * bt_ann
                          else "DECAYING — live well below backtest, reconsider")
    return out


def main():
    print("=" * 64)
    print("Insider strategy — live paper vs backtest calibration")
    print("=" * 64)
    m = run()
    json.dump(m, open(ROOT / "insider_calibration.json", "w"), indent=2, default=str)
    print(f"\n  {m.get('status','?')}")
    print(f"  open {m.get('open_positions',0)} · closed {m.get('closed_trades',0)}")
    be = m.get("backtest_expected", {})
    print(f"  backtest expects: alpha {be.get('alpha_annual')}  Sharpe {be.get('sharpe')}")
    if m.get("status") == "LIVE":
        ra = m.get("realized_all", {})
        print(f"  LIVE realized: win {ra.get('win_rate')} · avg/trade {ra.get('avg_return')} · "
              f"annualized {ra.get('annualized_est')}")
        print(f"  live/backtest ratio: {m.get('live_vs_backtest_ratio')} → {m.get('verdict')}")
    print(f"\n  → insider_calibration.json")


if __name__ == "__main__":
    main()
