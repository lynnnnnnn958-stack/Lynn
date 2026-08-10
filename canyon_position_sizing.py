#!/usr/bin/env python3
"""
canyon_position_sizing.py — 第6层 仓位构建与风控 (事件驱动)
===========================================================
把事件候选 + L0稀缺度 + 手册双轨, 变成"具体该持多少仓位"的组合, 带机构级风控。
复用旧风控框架证过的数学(逆波动加权 / 回撤敞口乘数 / 硬上限), 但由信念(FES)驱动。

流程:
  1. 总进攻预算 ← L0机会稀缺度带(稀缺20-45%/一般45-70%/丰富70-100%)
                    × 宏观过滤(MacroFilter) × 回撤敞口乘数(drawdown_control_state)
  2. 可动用标的 ← 利润发动机/事件爆发池; 若空(机会稀缺)则取利润发动机储备池 top-K 作"试探仓"
  3. 每只原始权重 = 信念(FES分位) × 执行质量(EF) × 逆波动(风险平价)
  4. 双轨(第6层): 利润发动机轨 单票上限高(14%), 普通轨低(7%)
  5. 硬约束: 单票上限(按轨) · 单事件类型≤45% · 单行业≤35% · 合计≤总进攻预算 → 其余现金
  6. 组合体检: 投入/现金比 · 组合波动估计 · 集中度

输出: position_plan_event.csv + position_plan_summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

CAP_ENGINE = 0.14      # 利润发动机轨 单票上限
CAP_NORMAL = 0.07      # 普通轨 单票上限
CAP_EVENT_TYPE = 0.45  # 单一事件类型上限
CAP_SECTOR = 0.35      # 单一行业上限
PROBE_SCALE = 0.5      # 机会稀缺、只有储备池时的试探仓折扣


def load(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def sector_map():
    d = load("alpha_scores.csv")
    if not d.empty and "sector" in d.columns:
        return {str(r["ticker"]): str(r["sector"]) for _, r in d.iterrows()}
    return {}


def ann_vol(px, tk):
    if tk in px.columns:
        s = px[tk].dropna()
        if len(s) > 60:
            return float(s.pct_change().dropna().tail(126).std() * np.sqrt(252))
    return 0.30


def _portfolio_vol(plan: pd.DataFrame, px: pd.DataFrame) -> float:
    """真实组合年化波动 √(wΣw): 用持仓名的价格历史算协方差(含真实相关性)。"""
    tks = [t for t in plan["ticker"] if t in px.columns]
    if len(tks) < 2:
        return float((plan["w"] * plan["ann_vol"]).sum())
    rets = px[tks].pct_change().dropna().tail(252)
    if len(rets) < 60:
        return float((plan["w"] * plan["ann_vol"]).sum()) * 0.85
    cov = rets.cov().values * 252                          # 年化协方差
    w = plan.set_index("ticker").loc[tks, "w"].values
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0)))


VALID_EDGE_TYPES = {"行业爆发型", "第二春重估型", "企业重大事故型"}   # 8-K历史证过有真edge
CONC_N = 10            # 集中持股数
CONC_CAP = 0.16       # 集中模式单票上限
CONC_SECTOR_CAP = 0.45


def concentrated_portfolio(cand, px, secmap, mf):
    """集中冲锋清单: 8-12 只最高信念的**验证过事件类型**标的, 冲击跑赢纳指。
    权重 = 信念(FES分位)×逆波动, 单票≤16%, 满仓(×宏观)。回测显示 ~10只/不杠杆最优。"""
    # 优先真·行业爆发型(AI/科技,验证过t最高),不足再用其他验证过类型补
    boom = cand[cand["event_type"] == "行业爆发型"].sort_values("FinalEventScore", ascending=False)
    rest = cand[cand["event_type"].isin(VALID_EDGE_TYPES - {"行业爆发型"})].sort_values("FinalEventScore", ascending=False)
    c = pd.concat([boom.head(CONC_N), rest]).head(CONC_N).copy()
    if c.empty:
        return pd.DataFrame(), {}
    fes = c["FinalEventScore"].astype(float)
    fes_pct = (fes - fes.min()) / (fes.max() - fes.min() + 1e-9) * 0.5 + 0.5
    rows = []
    for i, (_, r) in enumerate(c.iterrows()):
        tk = str(r["ticker"]); v = ann_vol(px, tk)
        raw = float(fes_pct.iloc[i]) * (0.25 / max(v, 0.12))
        rows.append({"ticker": tk, "event_type": r.get("event_type", ""),
                     "sector": secmap.get(tk, "—"), "FES": round(float(r["FinalEventScore"]), 1),
                     "EF": round(float(r.get("ExecutionFilter", 0.6)), 2), "ann_vol": round(v, 3), "raw": raw})
    p = pd.DataFrame(rows)
    deploy = float(np.clip(0.90 * mf, 0.5, 1.0))          # 集中=高信念=满仓(略受宏观约束)
    p["w"] = p["raw"] / p["raw"].sum() * deploy
    for _ in range(6):
        over = p["w"] > CONC_CAP
        if over.any():
            excess = float((p.loc[over, "w"] - CONC_CAP).sum()); p.loc[over, "w"] = CONC_CAP
            room = ~over
            if room.any() and excess > 1e-6:
                p.loc[room, "w"] += p.loc[room, "raw"] / p.loc[room, "raw"].sum() * excess
        else:
            break
    for sec, g in p.groupby("sector"):           # 行业上限
        if g["w"].sum() > CONC_SECTOR_CAP:
            p.loc[g.index, "w"] = g["w"] / g["w"].sum() * CONC_SECTOR_CAP
    p["w"] = p["w"].clip(upper=CONC_CAP)
    p = p.sort_values("w", ascending=False).reset_index(drop=True)
    p["weight_pct"] = (p["w"] * 100).round(1)
    pvol = _portfolio_vol(p, px)
    inv = float(p["w"].sum())
    summ = {"n": len(p), "invested_pct": round(inv * 100, 1), "cash_pct": round((1 - inv) * 100, 1),
            "portfolio_vol_est": round(pvol, 3),
            "top_sector": p.groupby("sector")["w"].sum().idxmax(),
            "top_sector_pct": round(float(p.groupby("sector")["w"].sum().max()) * 100, 1),
            "note": "集中冲锋清单·目标跑赢纳指·仅验证过事件类型·回测参考非承诺"}
    p[["ticker", "event_type", "sector", "weight_pct", "FES", "EF", "ann_vol"]].to_csv(
        ROOT / "concentrated_portfolio.csv", index=False)
    json.dump(summ, open(ROOT / "concentrated_summary.json", "w"), ensure_ascii=False, indent=2)
    return p, summ


def scarcity_band(n_active):
    if n_active <= 2:   return 0.20, 0.45, "机会稀缺"
    if n_active <= 5:   return 0.45, 0.70, "机会一般"
    return 0.70, 1.00, "机会丰富"


def drawdown_multiplier():
    p = ROOT / "drawdown_control_state.json"
    if p.exists():
        try:
            m = json.load(open(p)).get("drawdown_exposure_multiplier")
            if m is not None:
                return float(m)
        except Exception:
            pass
    return 1.0


def run():
    cand = load("event_candidates.csv")
    if cand.empty:
        print("缺 event_candidates.csv"); return {}
    func = load("functional_pools.csv")
    px = prices()
    secmap = sector_map()

    n_pe = int((cand["pool"] == "利润发动机").sum())
    n_ev = int((cand["pool"] == "事件型爆发池").sum())
    lo, hi, band = scarcity_band(n_pe + n_ev)
    mf = float(cand["MacroFilter"].iloc[0]) if "MacroFilter" in cand.columns and len(cand) else 0.9
    ddm = drawdown_multiplier()

    # 可动用标的: 真正入池的(利润发动机/事件爆发池) = 全仓资格; 储备池 = 试探仓资格
    active = cand[cand["pool"].isin(["利润发动机", "事件型爆发池"])].copy()
    reserve_names = set(func[func["func_pool"] == "利润发动机储备池"]["ticker"].astype(str)) if not func.empty else set()
    reserve = cand[cand["ticker"].astype(str).isin(reserve_names) & ~cand["ticker"].isin(active["ticker"])].copy()
    reserve = reserve.sort_values("FinalEventScore", ascending=False)

    # 总进攻预算: 稀缺带中值 × 宏观 × 回撤乘数
    base = (lo + hi) / 2
    total_budget = float(np.clip(base * mf * ddm, 0.05, 1.0))

    # 先放全仓资格的; 若不足以填到带下沿, 用储备池试探仓补(单票折扣)
    sel = active.copy(); sel["is_probe"] = False
    if len(active) < 6 and not reserve.empty:
        need = max(0, 6 - len(active))
        add = reserve.head(max(need, 4)).copy(); add["is_probe"] = True
        sel = pd.concat([sel, add], ignore_index=True)
    if sel.empty:
        sel = cand.sort_values("FinalEventScore", ascending=False).head(6).copy(); sel["is_probe"] = True
    probe = bool(sel["is_probe"].all())

    # 每只原始权重 = FES分位 × EF × 逆波动
    fes = sel["FinalEventScore"].astype(float)
    fes_pct = (fes - fes.min()) / (fes.max() - fes.min() + 1e-9) * 0.6 + 0.4   # 0.4-1.0
    rows = []
    for i, (_, r) in enumerate(sel.iterrows()):
        tk = str(r["ticker"])
        ef = float(r.get("ExecutionFilter", 0.6))
        v = ann_vol(px, tk)
        track = str(r.get("position_track", "普通仓位轨"))
        isp = bool(r["is_probe"])
        raw = float(fes_pct.iloc[i]) * ef * (0.25 / max(v, 0.12)) * (0.6 if isp else 1.0)
        rows.append({"ticker": tk, "track": track, "event_type": r.get("event_type", ""),
                     "sector": secmap.get(tk, "—"), "FES": round(float(r["FinalEventScore"]), 1),
                     "EF": round(ef, 2), "ann_vol": round(v, 3), "is_probe": isp, "raw": raw})
    plan = pd.DataFrame(rows)
    if plan.empty:
        return {}

    # 归一到总预算, 再套硬上限 + 迭代重分配
    plan["w"] = plan["raw"] / plan["raw"].sum() * total_budget
    for _ in range(6):
        # 单票上限(按轨); 试探仓单独折扣
        plan["cap"] = np.where(plan["track"].str.contains("利润发动机"), CAP_ENGINE, CAP_NORMAL)
        plan["cap"] = np.where(plan["is_probe"], plan["cap"] * PROBE_SCALE, plan["cap"])
        over = plan["w"] > plan["cap"]
        if over.any():
            excess = float((plan.loc[over, "w"] - plan.loc[over, "cap"]).sum())
            plan.loc[over, "w"] = plan.loc[over, "cap"]
            room = ~over
            if room.any() and excess > 1e-6:
                add = plan.loc[room, "raw"] / plan.loc[room, "raw"].sum() * excess
                plan.loc[room, "w"] = plan.loc[room, "w"] + add
        else:
            break
    # 事件类型 / 行业 上限
    for col, cap in [("event_type", CAP_EVENT_TYPE), ("sector", CAP_SECTOR)]:
        for key, grp in plan.groupby(col):
            tot = grp["w"].sum()
            if tot > cap:
                plan.loc[grp.index, "w"] = grp["w"] / tot * cap
    plan["w"] = plan["w"].clip(upper=plan["cap"])

    invested = float(plan["w"].sum())
    cash = max(0.0, 1.0 - invested)
    # 组合波动: 用真实相关性协方差矩阵 √(wΣw) (从价格历史算), 而非假设平相关
    port_vol = _portfolio_vol(plan, px)
    plan = plan.sort_values("w", ascending=False).reset_index(drop=True)
    plan["weight_pct"] = (plan["w"] * 100).round(1)
    plan["rationale"] = plan.apply(
        lambda r: f"{r['track'].replace('仓位','')}·信念FES{r['FES']:.0f}·EF{r['EF']}·波动{r['ann_vol']:.0%}"
                  + ("·试探仓" if r["is_probe"] else "·正式仓"), axis=1)

    out_cols = ["ticker", "track", "weight_pct", "event_type", "sector", "FES", "EF", "ann_vol", "rationale"]
    plan[out_cols].to_csv(ROOT / "position_plan_event.csv", index=False)

    summary = {
        "band": band, "probe_mode": probe,
        "total_budget_pct": round(total_budget * 100, 1),
        "invested_pct": round(invested * 100, 1),
        "cash_pct": round(cash * 100, 1),
        "macro_filter": round(mf, 2),
        "drawdown_multiplier": round(ddm, 2),
        "portfolio_vol_est": round(port_vol, 3),
        "n_positions": len(plan),
        "n_engine_track": int(plan["track"].str.contains("利润发动机").sum()),
        "top_sector": plan.groupby("sector")["w"].sum().idxmax() if len(plan) else "—",
        "top_sector_pct": round(float(plan.groupby("sector")["w"].sum().max()) * 100, 1) if len(plan) else 0,
    }
    json.dump(summary, open(ROOT / "position_plan_summary.json", "w"), ensure_ascii=False, indent=2)
    conc, conc_summ = concentrated_portfolio(cand, px, secmap, mf)
    return {"plan": plan, "summary": summary, "conc": conc, "conc_summary": conc_summ}


def main():
    print("=" * 62)
    print("第6层 仓位构建与风控 — 事件驱动")
    print("=" * 62)
    r = run()
    if not r:
        return
    s = r["summary"]; plan = r["plan"]
    print(f"  L0机会稀缺度: {s['band']}" + ("(试探仓模式)" if s["probe_mode"] else ""))
    print(f"  总进攻预算 {s['total_budget_pct']}% (宏观×{s['macro_filter']} 回撤×{s['drawdown_multiplier']})")
    print(f"  → 实际投入 {s['invested_pct']}% · 现金 {s['cash_pct']}% · 组合波动估计 {s['portfolio_vol_est']:.0%}")
    print(f"  {s['n_positions']} 个仓位({s['n_engine_track']}个发动机轨) · 最大行业 {s['top_sector']} {s['top_sector_pct']}%")
    print(f"\n  {'标的':<7}{'轨':<6}{'仓位':>7}  {'事件类型':<12}{'理由'}")
    for _, p in plan.iterrows():
        tr = "发动机" if "利润发动机" in p["track"] else "普通"
        print(f"  {p['ticker']:<7}{tr:<6}{p['weight_pct']:>6.1f}%  {p['event_type']:<12}{p['rationale']}")
    print("\n  → position_plan_event.csv · position_plan_summary.json")
    conc = r.get("conc"); cs = r.get("conc_summary", {})
    if conc is not None and len(conc):
        print("\n" + "=" * 62)
        print(f"  ★ 集中冲锋清单(冲击跑赢纳指) — {cs.get('n')}只 · 投入{cs.get('invested_pct')}% · "
              f"组合波动{cs.get('portfolio_vol_est',0)*100:.0f}% · 最大行业{cs.get('top_sector')} {cs.get('top_sector_pct')}%")
        print("=" * 62)
        print(f"  {'标的':<7}{'仓位':>7}  {'事件类型':<12}{'行业':<12}FES")
        for _, p in conc.iterrows():
            print(f"  {p['ticker']:<7}{p['weight_pct']:>6.1f}%  {p['event_type']:<12}{str(p['sector']):<12}{p['FES']:.0f}")
        print("\n  → concentrated_portfolio.csv · concentrated_summary.json")


if __name__ == "__main__":
    main()
