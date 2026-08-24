#!/usr/bin/env python3
"""
canyon_fwd_ic.py — 事件引擎 FinalEventScore 的前瞻 IC 跟踪(闭环验证)
====================================================================
整个 10 层系统的核心问题: FinalEventScore 到底能不能预测随后的收益? 8-K 事件研究
证明了事件层历史 edge(t=4~8), 但 LIVE 的 FES 组合分需要**前瞻**证明。

review_history.csv 每天存一份 495 只的 FES 快照(2026-07-27 起累积)。这里用它 +
价格, 算每个快照日之后 FWD 天的实现收益, 与当日 FES 的秩相关(Spearman IC)——
这是系统的**真实前瞻战绩**, 不是回测。

诚实: 刚累积 ~1 个月, 样本小 → 报"N 日, IC 初值"; 要 3-6 个月才够定论。守住正 IC
= 系统真在选对; 掉到 0/负 = 诚实承认 FES 没预测力。

Output: fwd_ic.json + fwd_ic_history.csv(逐日追加)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
FWD = 10                      # 前瞻交易日(与事件持仓期一致)


def _prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None


def run():
    hp = ROOT / "review_history.csv"
    if not hp.exists():
        return {"status": "no review_history yet"}
    h = pd.read_csv(hp)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    px = _prices()
    if px is None:
        return {"status": "no price file"}
    idx = px.index
    ret_fwd = {}   # (date) -> forward returns per ticker
    ics = []
    for d, g in h.groupby("date"):
        pos = idx.searchsorted(d)
        if pos < 0 or pos + FWD >= len(idx):
            continue                                   # 未来收益还没到 → 跳过(最近的日)
        p0 = px.iloc[pos]; p1 = px.iloc[pos + FWD]
        g = g.dropna(subset=["FinalEventScore"])
        fes, fwd = [], []
        for _, r in g.iterrows():
            tk = str(r["ticker"])
            if tk in px.columns:
                a, b = p0.get(tk), p1.get(tk)
                if pd.notna(a) and pd.notna(b) and a > 0:
                    fes.append(float(r["FinalEventScore"])); fwd.append(b / a - 1)
        if len(fes) >= 30:
            ic = pd.Series(fes).corr(pd.Series(fwd), method="spearman")
            # top-bottom 五分位差
            df = pd.DataFrame({"fes": fes, "fwd": fwd}).sort_values("fes")
            q = max(len(df) // 5, 1)
            tb = df.tail(q)["fwd"].mean() - df.head(q)["fwd"].mean()
            ics.append({"date": d.strftime("%Y-%m-%d"), "ic": round(float(ic), 4),
                        "top_bottom_spread": round(float(tb), 4), "n": len(fes)})

    out = {"as_of": pd.Timestamp.now().isoformat(), "fwd_days": FWD,
           "snapshots_evaluated": len(ics)}
    if not ics:
        out["status"] = "PENDING — snapshots exist but forward window not elapsed yet"
        return out
    icv = np.array([x["ic"] for x in ics])
    tbv = np.array([x["top_bottom_spread"] for x in ics])
    out["status"] = "LIVE"
    out["ic_mean"] = round(float(icv.mean()), 4)
    out["ic_t"] = round(float(icv.mean() / icv.std() * np.sqrt(len(icv))), 2) if icv.std() else None
    out["ic_hit_rate"] = round(float((icv > 0).mean()), 3)
    out["top_bottom_spread_mean"] = round(float(tbv.mean()), 4)
    out["by_snapshot"] = ics
    out["verdict"] = ("HOLDING — FES has positive forward IC" if icv.mean() > 0.02
                      else "WEAK/NONE — FES not yet predicting forward; keep accumulating")
    # 追加逐日历史
    pd.DataFrame(ics).to_csv(ROOT / "fwd_ic_history.csv", index=False)
    return out


def main():
    print("=" * 64)
    print("Event engine — forward IC of FinalEventScore (live validation)")
    print("=" * 64)
    m = run()
    json.dump(m, open(ROOT / "fwd_ic.json", "w"), indent=2, default=str, ensure_ascii=False)
    print(f"\n  {m.get('status','?')}")
    if m.get("status") == "LIVE":
        print(f"  snapshots {m['snapshots_evaluated']} · fwd {m['fwd_days']}d")
        print(f"  IC mean {m['ic_mean']}  t={m['ic_t']}  hit-rate {m['ic_hit_rate']}")
        print(f"  top-bottom quintile spread {m['top_bottom_spread_mean']:+.2%} per {m['fwd_days']}d")
        print(f"  → {m['verdict']}")
    print(f"\n  → fwd_ic.json + fwd_ic_history.csv")


if __name__ == "__main__":
    main()
