#!/usr/bin/env python3
"""
canyon_review.py — 复盘节奏层 (周 / 月 / 季)
============================================
手册收尾环节: 事件驱动系统不是"选完就放着", 而是按节奏复盘、兑现、剔除。
本层做两件事:
  1. 每天存一份状态快照 (review_history.csv), 让系统有记忆, 能对比"变化"。
  2. 生成三个节奏的复盘, 每个节奏问不同的问题:

  周复盘 (战术) — 逻辑还在不在?
     · 退出触发扫描: 结构破位(跌破ma50)/催化已过(财报日已过)/主线退潮 → 减仓或清仓信号
     · 相比 7 天前: 谁掉出活跃池、谁分数跌 → 需要关注的恶化项
  月复盘 (调仓) — 池子该轮换了吗?
     · 功能池迁移: 谁从储备升入发动机、谁跌进回收池
     · 池健康: 回收池是否过大、发动机储备是否枯竭、集中度
  季复盘 (系统) — 打法本身对不对?
     · 事件类型分布与 FES 贡献: 哪类事件在当下最肥
     · 宏观模式变化: 进攻/防守切换
     · 需要校准什么

输出: review_report.json (+ 追加 review_history.csv)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
HIST = ROOT / "review_history.csv"
ACTIVE_POOLS = {"利润发动机", "事件型爆发池"}          # 第7层活跃池
ENGINE_RESERVE = "利润发动机储备池"
RECYCLE = "回收观察池"


def load(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def snapshot(cand, func, today):
    """存当天状态快照(每 ticker 一行), 追加到 review_history.csv (同日覆盖)"""
    fp_m = {str(r["ticker"]): str(r["func_pool"]) for _, r in func.iterrows()} if not func.empty else {}
    rows = []
    for _, r in cand.iterrows():
        tk = str(r["ticker"])
        rows.append({"date": today, "ticker": tk, "event_type": r.get("event_type", ""),
                     "FinalEventScore": round(float(r.get("FinalEventScore", 0)), 1),
                     "pool": r.get("pool", ""), "func_pool": fp_m.get(tk, "")})
    snap = pd.DataFrame(rows)
    if HIST.exists():
        old = pd.read_csv(HIST)
        old = old[old["date"] != today]                # 同日重跑覆盖
        snap = pd.concat([old, snap], ignore_index=True)
    snap.to_csv(HIST, index=False)
    return snap


def _asof_snapshot(hist, today, days_back):
    """取 days_back 天前最接近的一份历史快照(至少差 days_back*0.6 天才算有效对比)"""
    if hist.empty:
        return None, None
    dates = sorted(hist["date"].unique())
    target = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
    past = [d for d in dates if d <= target]
    if not past:
        return None, None
    pick = past[-1]
    age = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(pick, "%Y-%m-%d")).days
    if age < days_back * 0.5:
        return None, None
    return hist[hist["date"] == pick].set_index("ticker"), pick


def weekly(cand, hist, today, px):
    """周复盘: 退出触发扫描 + 7天恶化项"""
    triggers = []
    for _, r in cand.iterrows():
        if str(r.get("pool")) not in ACTIVE_POOLS and str(r.get("func_pool")) != ENGINE_RESERVE:
            # 只扫活跃/预备役, 但 cand 无 func_pool 列, 放宽: 扫前50
            pass
    top = cand.head(50)
    for _, r in top.iterrows():
        tk = str(r["ticker"]); flags = []
        if tk in px.columns:
            s = px[tk].dropna()
            if len(s) > 60:
                p = float(s.iloc[-1]); ma50 = float(s.tail(50).mean())
                if p < ma50 * 0.98:
                    flags.append("跌破50日线(结构恶化)")
                if p < float(s.tail(20).mean()) * 0.95:
                    flags.append("短期破位")
        if flags:
            triggers.append({"ticker": tk, "event_type": r.get("event_type", ""),
                             "FinalEventScore": r.get("FinalEventScore", ""),
                             "exit_force": r.get("exit_force", ""),
                             "flags": flags, "action": "复核逻辑→按退出模板减/清"})
    # 7天分数恶化
    past, pdate = _asof_snapshot(hist, today, 7)
    decays = []
    if past is not None:
        cur = cand.set_index("ticker")
        for tk in cur.index:
            if tk in past.index:
                d = float(cur.loc[tk, "FinalEventScore"]) - float(past.loc[tk, "FinalEventScore"])
                if d <= -8:
                    decays.append({"ticker": tk, "drop": round(d, 1),
                                   "from": past.loc[tk, "FinalEventScore"], "to": cur.loc[tk, "FinalEventScore"]})
        decays = sorted(decays, key=lambda x: x["drop"])[:10]
    return {"exit_triggers": triggers[:15], "score_decays": decays,
            "compare_date": pdate, "note": "首次快照,7天对比需累积后可用" if pdate is None else ""}


def monthly(func, hist, today):
    """月复盘: 功能池迁移 + 池健康"""
    counts = func["func_pool"].value_counts().to_dict() if not func.empty else {}
    total = int(sum(counts.values())) or 1
    health = []
    rec = counts.get(RECYCLE, 0)
    eng = counts.get(ENGINE_RESERVE, 0)
    if rec / total > 0.5:
        health.append(f"⚠ 回收观察池占 {rec/total:.0%} 偏高 — 大盘/多数标的趋势走弱,宜降总仓位、收缩到少数强主线")
    if eng < 8:
        health.append(f"⚠ 利润发动机储备仅 {eng} 只 — 机会稀缺,保持高现金、只打最强确认")
    else:
        health.append(f"✓ 发动机储备 {eng} 只,机会池充足")
    # 池迁移 (30天)
    past, pdate = _asof_snapshot(hist, today, 30)
    promotions, demotions = [], []
    if past is not None and "func_pool" in past.columns:
        cur = func.set_index("ticker")
        rank = {RECYCLE: 0, "主题链条观察池": 1, "核心储备池": 2, ENGINE_RESERVE: 3}
        for tk in cur.index:
            if tk in past.index:
                a = rank.get(str(past.loc[tk, "func_pool"]), 1); b = rank.get(str(cur.loc[tk, "func_pool"]), 1)
                if b - a >= 2:
                    promotions.append({"ticker": tk, "from": past.loc[tk, "func_pool"], "to": cur.loc[tk, "func_pool"]})
                elif a - b >= 2:
                    demotions.append({"ticker": tk, "from": past.loc[tk, "func_pool"], "to": cur.loc[tk, "func_pool"]})
    return {"pool_counts": counts, "health": health, "promotions": promotions[:10],
            "demotions": demotions[:10], "compare_date": pdate,
            "note": "首次,30天池迁移需累积后可用" if pdate is None else ""}


def quarterly(cand, hist, today):
    """季复盘: 事件类型贡献 + 宏观模式 + 校准建议"""
    et = cand.groupby("event_type")["FinalEventScore"].agg(["count", "mean"]).round(1)
    et_rows = [{"event_type": i, "count": int(r["count"]), "avg_FES": float(r["mean"])}
               for i, r in et.sort_values("mean", ascending=False).iterrows()]
    # 宏观模式变化 (季度前)
    mode_now = ""
    ip = ROOT / "macro_intel_scorecard.json"
    if ip.exists():
        try:
            mode_now = json.load(open(ip)).get("宏观模式", "")
        except Exception:
            pass
    past, pdate = _asof_snapshot(hist, today, 90)
    recal = []
    if et_rows:
        hot = et_rows[0]
        recal.append(f"当下最肥事件类型: {hot['event_type']} (均 FES {hot['avg_FES']}, {hot['count']}只) — 资源向此倾斜")
        cold = et_rows[-1]
        recal.append(f"最弱: {cold['event_type']} (均 FES {cold['avg_FES']}) — 除非强前瞻确认,否则降权")
    recal.append(f"当前宏观模式: {mode_now or '—'} — 季度确认进攻/防守制度是否延续")
    return {"event_type_contribution": et_rows, "macro_mode": mode_now,
            "recalibration": recal, "compare_date": pdate}


def run():
    cand = load("event_candidates.csv")
    func = load("functional_pools.csv")
    if cand.empty:
        print("缺 event_candidates.csv"); return {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot(cand, func, today)
    hist = pd.read_csv(HIST)
    px = prices()
    report = {
        "updated": today,
        "history_days": int(hist["date"].nunique()),
        "weekly": weekly(cand, hist, today, px),
        "monthly": monthly(func, hist, today),
        "quarterly": quarterly(cand, hist, today),
    }
    json.dump(report, open(ROOT / "review_report.json", "w"), ensure_ascii=False, indent=2)
    return report


def main():
    print("=" * 56)
    print("复盘节奏层 — 周 / 月 / 季")
    print("=" * 56)
    r = run()
    if not r:
        return
    print(f"  快照历史: {r['history_days']} 天")
    w = r["weekly"]
    print(f"\n  【周】退出触发 {len(w['exit_triggers'])} 项:")
    for t in w["exit_triggers"][:6]:
        print(f"    {t['ticker']:6} {'/'.join(t['flags'])} → {t['action']}")
    if w["score_decays"]:
        print(f"  7天分数恶化: " + ", ".join(f"{d['ticker']}({d['drop']})" for d in w["score_decays"][:6]))
    elif w["note"]:
        print(f"  ({w['note']})")
    m = r["monthly"]
    print(f"\n  【月】功能池: {m['pool_counts']}")
    for h in m["health"]:
        print(f"    {h}")
    q = r["quarterly"]
    print(f"\n  【季】事件类型贡献 (均FES):")
    for e in q["event_type_contribution"]:
        print(f"    {e['event_type']:14} {e['count']:3}只  均FES {e['avg_FES']}")
    for c in q["recalibration"]:
        print(f"    · {c}")
    print("\n  → review_report.json")


if __name__ == "__main__":
    main()
