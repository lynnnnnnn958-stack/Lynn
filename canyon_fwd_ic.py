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
# 简单挑战者: 事件类型倾向(2年研究t值序)主导 + 类内动量 tiebreak
TYPEW = {"行业爆发型": 4, "商品供需错配型": 3, "企业重大事故型": 2,
         "战争/地缘冲击型": 1.5, "第二春重估型": 1}


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
    ics, chal = [], []                                 # FES 逐快照IC / 简单挑战者IC
    fes_top, chal_top = [], []                          # 各自 top20% 前瞻收益
    for d, g in h.groupby("date"):
        pos = idx.searchsorted(d)
        if pos < 64 or pos + FWD >= len(idx):
            continue                                   # 需要动量回溯63天 + 前瞻窗口未到则跳过
        p0 = px.iloc[pos]; p1 = px.iloc[pos + FWD]; pm = px.iloc[pos - 63]
        g = g.dropna(subset=["FinalEventScore"])
        recs = []
        for _, r in g.iterrows():
            tk = str(r["ticker"])
            if tk in px.columns:
                a, b, m = p0.get(tk), p1.get(tk), pm.get(tk)
                if pd.notna(a) and pd.notna(b) and pd.notna(m) and a > 0 and m > 0:
                    typew = TYPEW.get(str(r["event_type"]), 1.0)
                    recs.append({"fes": float(r["FinalEventScore"]),
                                 "chal": typew * 100 + (a / m - 1) * 10,   # 事件类型+动量
                                 "fwd": b / a - 1})
        if len(recs) < 30:
            continue
        df = pd.DataFrame(recs)
        q = max(len(df) // 5, 1)
        ic_f = float(df["fes"].corr(df["fwd"], method="spearman"))
        ic_c = float(df["chal"].corr(df["fwd"], method="spearman"))
        ics.append({"date": d.strftime("%Y-%m-%d"), "ic": round(ic_f, 4), "n": len(df)})
        chal.append({"date": d.strftime("%Y-%m-%d"), "ic": round(ic_c, 4)})
        fes_top.append(float(df.nlargest(q, "fes")["fwd"].mean()))
        chal_top.append(float(df.nlargest(q, "chal")["fwd"].mean()))

    out = {"as_of": pd.Timestamp.now().isoformat(), "fwd_days": FWD,
           "snapshots_evaluated": len(ics)}
    if not ics:
        out["status"] = "PENDING — snapshots exist but forward window not elapsed yet"
        return out
    icv = np.array([x["ic"] for x in ics]); cv = np.array([x["ic"] for x in chal])
    out["status"] = "LIVE"
    out["complex_fes"] = {"ic_mean": round(float(icv.mean()), 4),
                          "top20_fwd": round(float(np.mean(fes_top)), 4),
                          "ic_hit_rate": round(float((icv > 0).mean()), 3)}
    out["simple_challenger"] = {"ic_mean": round(float(cv.mean()), 4),
                                "top20_fwd": round(float(np.mean(chal_top)), 4),
                                "ic_hit_rate": round(float((cv > 0).mean()), 3),
                                "definition": "event-type (2yr-t order) + within-type momentum"}
    out["winner"] = ("simple_challenger" if cv.mean() > icv.mean() else "complex_fes")
    out["ic_mean"] = out["complex_fes"]["ic_mean"]     # 向后兼容 IC desk
    out["verdict"] = (f"SIMPLE WINS so far ({out['simple_challenger']['ic_mean']} vs {out['complex_fes']['ic_mean']}) "
                      "— complexity not yet earning its keep; keep accumulating"
                      if out["winner"] == "simple_challenger"
                      else f"COMPLEX holds ({out['complex_fes']['ic_mean']} vs {out['simple_challenger']['ic_mean']})")
    pd.DataFrame([{**a, "chal_ic": b["ic"]} for a, b in zip(ics, chal)]).to_csv(ROOT / "fwd_ic_history.csv", index=False)
    return out


def main():
    print("=" * 64)
    print("Event engine — forward IC of FinalEventScore (live validation)")
    print("=" * 64)
    m = run()
    json.dump(m, open(ROOT / "fwd_ic.json", "w"), indent=2, default=str, ensure_ascii=False)
    print(f"\n  {m.get('status','?')}")
    if m.get("status") == "LIVE":
        cf = m["complex_fes"]; sc = m["simple_challenger"]
        print(f"  snapshots {m['snapshots_evaluated']} · fwd {m['fwd_days']}d\n")
        print(f"  {'':22}{'前瞻IC':>10}{'top20%前瞻':>14}{'胜率':>8}")
        print(f"  {'① 复杂 FES(10层)':20}{cf['ic_mean']:>+10.4f}{cf['top20_fwd']:>+13.2%}{cf['ic_hit_rate']:>8}")
        print(f"  {'② 简单挑战者':20}{sc['ic_mean']:>+10.4f}{sc['top20_fwd']:>+13.2%}{sc['ic_hit_rate']:>8}")
        print(f"\n  → {m['verdict']}")
    print(f"\n  → fwd_ic.json + fwd_ic_history.csv")


if __name__ == "__main__":
    main()
