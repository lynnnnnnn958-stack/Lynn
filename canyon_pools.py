#!/usr/bin/env python3
"""
canyon_pools.py — 第4层 功能分层 (自动)
========================================
手册第4层: 把标普500全体按"在组合里扮演的角色"分到四个功能池。
这不同于第7层的事件打分(那是打分排序); 功能分层回答"这只股现在该被当成什么用":

  核心储备池 (Core Reserve)        —— 组合压舱石: 防御型/成熟趋势型 + 执行分稳健。
                                      不追事件, 提供 beta 与稳定性, 稀缺期也可留。
  利润发动机储备池 (Engine Reserve) —— 预备役: FinalEventScore 高 + 执行过滤≥0.6 +
                                      逼近利润发动机门槛(L/M 接近3)。一旦前瞻确认(C)到位即可升入发动机。
  主题链条观察池 (Theme Watch)     —— 有事件信号(新闻侦测命中)但执行/结构未就位, 挂单观察。
  回收观察池 (Recycle Watch)       —— 衰退期/趋势破坏/执行分低: 逻辑走坏, 待剔除或反向观察。

输入: event_candidates.csv(第7层结果) + lifecycle_style.csv(第2层) + auto_event_candidates.csv(侦测)
输出: functional_pools.csv (ticker, func_pool, reason, FinalEventScore, ExecutionFilter, lifecycle, style, detect_confidence)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def load(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def run():
    cand = load("event_candidates.csv")
    if cand.empty:
        print("缺 event_candidates.csv (先跑 canyon_event_system.py)"); return pd.DataFrame()
    life = load("lifecycle_style.csv")
    auto = load("auto_event_candidates.csv")

    life_m = {str(r["ticker"]): r for _, r in life.iterrows()} if not life.empty else {}
    conf_m = {str(r["ticker"]): float(r.get("detect_confidence", 0)) for _, r in auto.iterrows()} \
        if not auto.empty else {}

    # FES 分位 (相对排名, 只跟标普500自身比)
    fes = cand["FinalEventScore"].astype(float)
    q_hi = fes.quantile(0.90)      # 前10%
    q_mid = fes.quantile(0.60)

    rows = []
    for _, r in cand.iterrows():
        tk = str(r["ticker"])
        f = float(r["FinalEventScore"]); ef = float(r.get("ExecutionFilter", 0))
        L, M = float(r.get("L", 0)), float(r.get("M", 0))
        lc = life_m.get(tk)
        lifecycle = str(lc["lifecycle"]) if lc is not None else ""
        style = str(lc["style"]) if lc is not None else ""
        conf = conf_m.get(tk, 0.0)

        # ---- 分层决策 (优先级从上到下) ----
        if lifecycle == "衰退期" or ef < 0.45:
            fp, reason = "回收观察池", (f"衰退期·长期趋势走坏(EF{ef:.2f})" if lifecycle == "衰退期"
                                     else f"执行过滤过低 EF{ef:.2f} 逻辑未就位")
        elif f >= q_hi and ef >= 0.60 and L >= 2.8 and M >= 2.6:
            fp, reason = "利润发动机储备池", f"FES前10%({f:.0f})+EF{ef:.2f}+逼近门槛(L{L:.1f}/M{M:.1f})"
        elif conf >= 0.45:
            fp, reason = "主题链条观察池", f"新闻事件信号(侦测置信{conf:.2f}),执行待就位"
        elif style == "防御型" or (lifecycle in ("成熟期", "成长期") and style == "趋势型" and ef >= 0.55):
            fp, reason = "核心储备池", f"{lifecycle}/{style}·EF{ef:.2f} 压舱石"
        elif f >= q_mid and ef >= 0.55:
            fp, reason = "利润发动机储备池", f"FES中上({f:.0f})+EF{ef:.2f} 预备役"
        else:
            fp, reason = "主题链条观察池", f"{lifecycle or '—'}/{style or '—'} 中性观察"

        rows.append({"ticker": tk, "func_pool": fp, "reason": reason,
                     "FinalEventScore": round(f, 1), "ExecutionFilter": round(ef, 2),
                     "event_type": r.get("event_type", ""), "lifecycle": lifecycle,
                     "style": style, "detect_confidence": round(conf, 2)})
    df = pd.DataFrame(rows)
    order = {"利润发动机储备池": 0, "核心储备池": 1, "主题链条观察池": 2, "回收观察池": 3}
    df["_o"] = df["func_pool"].map(order)
    df = df.sort_values(["_o", "FinalEventScore"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    df.to_csv(ROOT / "functional_pools.csv", index=False)
    return df


def main():
    print("=" * 56)
    print("第4层 功能分层 — 标普500全体角色归属")
    print("=" * 56)
    df = run()
    if df.empty:
        return
    for fp in ["利润发动机储备池", "核心储备池", "主题链条观察池", "回收观察池"]:
        sub = df[df["func_pool"] == fp]
        print(f"\n  {fp} ({len(sub)}只):")
        for _, r in sub.head(6).iterrows():
            print(f"    {r.ticker:6} {r.event_type:12} FES{r.FinalEventScore:6.0f}  {r.reason}")
    print("\n  → functional_pools.csv")


if __name__ == "__main__":
    main()
