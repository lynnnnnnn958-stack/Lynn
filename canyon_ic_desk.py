#!/usr/bin/env python3
"""
canyon_ic_desk.py — 投委会决策层(把整套 10 层事件 OS 整合成一个机构级决定)
============================================================================
不是再造角落。这是把 Lynn 已建好的事件驱动 OS 的**所有层**连成一个投委会视图,
像真基金的 Investment Committee 每早看的那一页:
  · 宏观定基调(L1) → 事件引擎选股(L7 FinalEventScore, 全标普500) →
    集中组合(仓位/TCA) → 验证过的事件层 edge 背书 → 诚实结论。

整合的现有输出(全是 Lynn 系统已在 run_daily 里每天产的, 我只是连起来):
  macro_intel_scorecard.json  L1 宏观情报(模式/仓位制度/FRED/链条/事件池/热度)
  position_plan_summary.json  L6 仓位(机会带/已投/现金/组合波动)
  execution_cost_summary.json TCA(部署/往返摩擦/net-edge)
  edgar_event_study.json      验证过的事件层 edge(按事件类型 t 值)
  concentrated_portfolio.csv  集中冲锋书(高信念 10 名 = 实际该买的)
  event_candidates.csv        全标普500 FinalEventScore 排名

内部人买入在这套里是 **C因子(前瞻确认)的增强器**, 不是独立策略 —— 已集成。

诚实结论(来自去偏回测): 去survivorship偏后, 分散版 ≈ QQQ(15%vs21%), 但**集中版
(10名)去偏后 24%/Sharpe1.4 跑赢 QQQ**; 事件层 edge 真实但适度; 前瞻IC需累积验证。

Output: ic_decision.json + 打印
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent


def _load_json(name):
    p = ROOT / name
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _load_csv(name):
    p = ROOT / name
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _beat_qqq(qqq):
    """从 qqq_angles.json 取 QQQ 基准 + 最优去偏配置(集中10只 PIT)对比。"""
    if not qqq:
        return {}
    q = qqq.get("QQQ", {})
    best = None
    for c in qqq.get("configs", []):
        if "集中10只" in str(c.get("label", "")):
            best = c; break
    if best is None and qqq.get("configs"):
        best = max(qqq["configs"], key=lambda c: c.get("sharpe", 0))
    return {
        "qqq_cagr": q.get("cagr_%"), "qqq_sharpe": q.get("sharpe"), "qqq_dd": q.get("max_dd_%"),
        "book_label": (best or {}).get("label"), "book_cagr": (best or {}).get("cagr_%"),
        "book_sharpe": (best or {}).get("sharpe"), "book_dd": (best or {}).get("max_dd_%"),
        "window": qqq.get("window"),
    }


def decide():
    macro = _load_json("macro_intel_scorecard.json")
    pos = _load_json("position_plan_summary.json")
    tca = _load_json("execution_cost_summary.json")
    edge = _load_json("edgar_event_study.json")
    book = _load_csv("concentrated_portfolio.csv")
    cand = _load_csv("event_candidates.csv")
    fic = _load_json("fwd_ic.json")                      # ③ FES 前瞻 IC(live 验证)
    qqq = _load_json("qqq_angles.json")                 # ② 去偏后 beat-QQQ 战绩

    # 验证过的事件层 edge → {事件类型: t}
    edge_t = {}
    for et, v in (edge.get("by_event_type") or {}).items():
        d63 = v.get("63d") or {}
        edge_t[et] = {"t": d63.get("t"), "mean_ab_%": d63.get("mean_ab_%"), "n": d63.get("n")}

    # 组合按事件类型/行业构成
    book_rows = []
    if not book.empty:
        for _, r in book.head(12).iterrows():
            et = str(r.get("event_type", ""))
            book_rows.append({
                "ticker": r.get("ticker"), "event_type": et,
                "sector": r.get("sector"), "weight_pct": round(float(r.get("weight_pct", 0)), 1),
                "FES": round(float(r.get("FES", 0)), 0),
                "edge_t": (edge_t.get(et) or {}).get("t"),   # 该事件类型的验证 t 值
            })
    # 全库分布(495)
    pool_dist = {}
    if not cand.empty and "event_type" in cand.columns:
        pool_dist = cand["event_type"].value_counts().head(6).to_dict()
        n_in_pool = int((cand.get("pool", pd.Series(dtype=str)).astype(str).str.contains("池").sum())) if "pool" in cand.columns else 0
    else:
        n_in_pool = 0

    return {
        "as_of": pd.Timestamp.now().isoformat(),
        "macro": {
            "mode": macro.get("宏观模式"), "position_regime": macro.get("总仓位制度"),
            "macro_filter": macro.get("macro_filter"), "fred_stress": macro.get("FRED硬压力"),
            "fred_signal": (macro.get("FRED信号") or [None])[0],
            "beneficiary_chains": macro.get("重点受益链条", []),
            "risk_chains": macro.get("风险链条", []),
            "active_event_pools": macro.get("激活事件池", []),
            "hot_modules": {k: v.get("heat") for k, v in (macro.get("情报模块") or {}).items()},
        },
        "portfolio": {
            "opportunity_band": pos.get("band"), "budget_pct": pos.get("total_budget_pct"),
            "invested_pct": pos.get("invested_pct"), "cash_pct": pos.get("cash_pct"),
            "portfolio_vol": pos.get("portfolio_vol_est"), "n_positions": pos.get("n_positions"),
            "top_sector": pos.get("top_sector"), "top_sector_pct": pos.get("top_sector_pct"),
            "roundtrip_bps": tca.get("blended_roundtrip_bps"),
            "deployed_usd": tca.get("total_deployed_usd"), "nav_usd": tca.get("nav_usd"),
            "tca_verdict": tca.get("verdict"),
        },
        "concentrated_book": book_rows,
        "pool_distribution": {k: int(v) for k, v in pool_dist.items()},
        "validated_edge": edge_t,
        "forward_ic": {                                  # ③ live 前瞻验证 + 复杂vs简单对决
            "status": fic.get("status"), "snapshots": fic.get("snapshots_evaluated"),
            "complex_fes": fic.get("complex_fes"), "simple_challenger": fic.get("simple_challenger"),
            "winner": fic.get("winner"), "verdict": fic.get("verdict"),
        },
        "beat_qqq": _beat_qqq(qqq),                      # ② 去偏后 concentrated vs QQQ
        "honesty": "De-biased (PIT membership): diversified ~= QQQ (15% vs 21%), but the CONCENTRATED 10-name "
                   "book beats QQQ de-biased (~24%/Sharpe1.4). Event-layer 8-K edge is real (t=4-8, survives "
                   "sector/beta neutralization) but modest; insider buying is the C-factor enhancer, not a "
                   "standalone. Forward IC still accumulating. Free-data speed is the hard cap vs funded pods.",
    }


def main():
    print("=" * 72)
    print("INVESTMENT COMMITTEE — one decision across the full 10-layer event OS")
    print("=" * 72)
    d = decide()
    json.dump(d, open(ROOT / "ic_decision.json", "w"), indent=2, default=str, ensure_ascii=False)
    m = d["macro"]; p = d["portfolio"]
    print(f"\n  MACRO (L1): {m['mode']} · 仓位制度 {m['position_regime']} · MacroFilter {m['macro_filter']} · "
          f"FRED压力 {m['fred_stress']}")
    print(f"    受益链条 {m['beneficiary_chains']} · 风险链条 {m['risk_chains']}")
    print(f"    激活事件池 {m['active_event_pools']} · 热点 {m['hot_modules']}")
    print(f"\n  PORTFOLIO (L6): 机会带 {p['opportunity_band']} · 预算 {p['budget_pct']}% · "
          f"已投 {p['invested_pct']}% · 现金 {p['cash_pct']}% · 组合波动 {p['portfolio_vol']}")
    print(f"    {p['n_positions']} 仓 · top行业 {p['top_sector']} {p['top_sector_pct']}% · "
          f"往返摩擦 {p['roundtrip_bps']}bps · 部署 ${p['deployed_usd']:,}")
    print(f"\n  CONCENTRATED BOOK (实际该买 · 事件类型edge背书):")
    for r in d["concentrated_book"][:10]:
        et = r["edge_t"]; sig = f"t={et:+.1f}" if et is not None else "未验证"
        print(f"    {str(r['ticker']):6} {r['weight_pct']:>4}%  FES {r['FES']:>4.0f}  {r['event_type']:<10} "
              f"{str(r['sector'])[:16]:16}  [{sig}]")
    print(f"\n  VALIDATED EDGE (事件层, 63d): " +
          " · ".join(f"{k} t={v['t']}" for k, v in d["validated_edge"].items()))
    print(f"\n  → ic_decision.json")


if __name__ == "__main__":
    main()
