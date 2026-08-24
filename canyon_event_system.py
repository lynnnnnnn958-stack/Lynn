#!/usr/bin/env python3
"""
canyon_event_system.py — 美股主动投资系统 (事件驱动) 打分引擎
================================================================
按《美股主动投资系统项目手册 v1.0》实现核心决策数学:
  第0层 机会稀缺度开关   → 总进攻仓位区间
  第5层 个股执行分        → ExecutionFilter (0-1)
  第7层 EventScore        → wL·L + wN·N + wM·M + wP·P + wC·C  (各因子0-4)
        FinalEventScore   = EventScore × ExecutionFilter × MacroFilter
  第9层 利润发动机入池门槛
  第6层 仓位轨 (普通 / 利润发动机)
  第10层 分型退出模板

判断类输入 (L/N/M/P/C、催化清晰度、逻辑确认度、失效清晰度) 来自 event_pool.csv,
由使用者按手册标准 0-4 打分;可自动从价格算的 (结构质量/量价确认/技术赔率/阶段匹配/
波动) 由 auto_execution_components() 填充。宏观过滤 (MacroFilter) 从宏观评分卡取。

输出: event_candidates.csv (排序后的候选 + 池归类 + 建议仓位 + 退出模板)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── 第7层: 六种事件类型的权重 (L,N,M,P,C) 与主持有窗口 (手册原表) ─────────────────
EVENT_TYPES = {
    "行业爆发型":       {"L": 32, "N": 24, "M": 18, "P": 16, "C": 10, "hold": "3-12个月"},
    "商品供需错配型":   {"L": 22, "N": 30, "M": 22, "P": 18, "C": 8,  "hold": "2-3个月"},
    "第二春重估型":     {"L": 26, "N": 20, "M": 28, "P": 14, "C": 12, "hold": "3-6个月(强到12)"},
    "战争/地缘冲击型":  {"L": 26, "N": 28, "M": 18, "P": 18, "C": 10, "hold": "1-3个月(少数6)"},
    "自然灾变型":       {"L": 20, "N": 30, "M": 20, "P": 20, "C": 10, "hold": "1-4个月"},
    "企业重大事故型":   {"L": 18, "N": 26, "M": 22, "P": 24, "C": 10, "hold": "1-3个月(少数6)"},
}

# ── 第5层: 执行分因子权重 (和=100) ────────────────────────────────────────────
L5_WEIGHTS = {
    "催化清晰度": 14, "逻辑确认度": 8,
    "结构质量": 17, "量价确认": 14,
    "技术赔率": 20, "阶段匹配度": 12,
    "失效清晰度": 10, "波动容忍匹配": 5,
}
L5_MANUAL = ["催化清晰度", "逻辑确认度", "失效清晰度"]           # 判断类, 来自 pool
L5_AUTO   = ["结构质量", "量价确认", "技术赔率", "阶段匹配度", "波动容忍匹配"]  # 可从价格算

# ── 第10层: 分型退出模板 (手册原表) ───────────────────────────────────────────
EXIT_TEMPLATES = {
    "第二春重估型":     {"trade": 40, "logic": 60, "window": "3-6个月", "principle": "防止卖飞",
                        "force": "关键验证失败/逻辑被证伪/结构持续恶化"},
    "商品供需错配型":   {"trade": 70, "logic": 30, "window": "2-3个月", "principle": "防止恋战",
                        "force": "供给冲击缓和/价格钝化/地缘降温"},
    "战争/地缘冲击型":  {"trade": 75, "logic": 25, "window": "1-3个月", "principle": "预期打满就兑现",
                        "force": "停火缓和/风险溢价回落/冲高回落"},
    "行业爆发型":       {"trade": 35, "logic": 65, "window": "3-12个月", "principle": "不能用商品思维",
                        "force": "行业增速放缓/主线退潮/结构恶化"},
    "自然灾变型":       {"trade": 65, "logic": 35, "window": "1-4个月", "principle": "冲击缓和即减",
                        "force": "供给恢复/重建完成/风险回落"},
    "企业重大事故型":   {"trade": 60, "logic": 40, "window": "1-3个月", "principle": "替代兑现即减",
                        "force": "竞争格局重定/替代逻辑走完"},
}


# ── 第0层: 机会稀缺度开关 ─────────────────────────────────────────────────────
def scarcity_switch(n_high_quality: int, overheated: bool = False) -> dict:
    if n_high_quality <= 2 or overheated:
        return {"state": "机会稀缺", "attack_pos": "20%-45%",
                "action": "只做利润发动机级机会;允许高现金"}
    if n_high_quality <= 5:
        return {"state": "机会一般", "attack_pos": "45%-70%",
                "action": "核心池 + 少数爆发池,严格筛选"}
    return {"state": "机会丰富", "attack_pos": "70%-100%",
            "action": "核心池与利润发动机池同步打开"}


# ── 第5层执行分 → ExecutionFilter ─────────────────────────────────────────────
def execution_score(row: pd.Series) -> float:
    """加权 0-1 各因子 → 0-100 分。缺失按 0.5 中性处理。"""
    s = 0.0
    for f, w in L5_WEIGHTS.items():
        v = row.get(f, np.nan)
        v = 0.5 if pd.isna(v) else float(np.clip(v, 0, 1))
        s += w * v
    return round(s, 1)


_LIFECYCLE = None
def _lifecycle_map():
    """第2层 生命周期/风格 → {ticker: {阶段匹配基准, 波动容忍, lifecycle, style}}"""
    global _LIFECYCLE
    if _LIFECYCLE is None:
        _LIFECYCLE = {}
        p = ROOT / "lifecycle_style.csv"
        if p.exists():
            try:
                for _, r in pd.read_csv(p).iterrows():
                    _LIFECYCLE[str(r["ticker"])] = dict(
                        阶段匹配基准=float(r.get("阶段匹配基准", 0.6)),
                        波动容忍=float(r.get("波动容忍", 0.7)),
                        lifecycle=str(r.get("lifecycle", "")), style=str(r.get("style", "")))
            except Exception:
                pass
    return _LIFECYCLE


def auto_execution_components(prices: pd.DataFrame, ticker: str, asof=None) -> dict:
    """从价格自动估算可量化的第5层因子 (0-1)。asof=None 用最新。"""
    if ticker not in prices.columns:
        return {}
    s = prices[ticker].dropna()
    if asof is not None:
        s = s[s.index <= asof]
    if len(s) < 260:
        return {}
    px = float(s.iloc[-1])
    hi252 = float(s.tail(252).max()); lo252 = float(s.tail(252).min())
    ma50 = float(s.tail(50).mean()); ma200 = float(s.tail(200).mean())
    rets = s.pct_change().dropna()
    vol = float(rets.tail(63).std() * np.sqrt(252))
    # 结构质量: 站上均线 + 距52周高不远
    struct = 0.5 * (px > ma50) + 0.3 * (ma50 > ma200) + 0.2 * (px / hi252)
    # 量价确认(无量, 用趋势斜率近似): 近20日动量为正
    mom20 = px / float(s.iloc[-21]) - 1
    volp = float(np.clip(0.5 + mom20 * 5, 0, 1))
    # 技术赔率: 距52周低越远、越接近高点空间越小 → 用回撤深度(有反弹空间=赔率好)
    drawdown = (px / hi252) if hi252 else 1.0
    odds = float(np.clip(1.2 - drawdown, 0, 1))          # 越接近高点赔率越低
    # 阶段匹配: 在上升结构(px>ma200)且未极度超买
    stage = float(np.clip(0.5 * (px > ma200) + 0.5 * (1 - (px / hi252)), 0, 1))
    # 波动容忍: 中等波动最佳
    voltol = float(np.clip(1 - abs(vol - 0.35) / 0.5, 0, 1))
    # 第2层融合: 生命周期修正阶段匹配, 行为风格修正波动容忍 (各 50/50 融合)
    lc = _lifecycle_map().get(ticker)
    if lc:
        stage = float(np.clip(0.5 * stage + 0.5 * lc["阶段匹配基准"], 0, 1))
        voltol = float(np.clip(0.5 * voltol + 0.5 * lc["波动容忍"], 0, 1))
    return {"结构质量": round(float(np.clip(struct, 0, 1)), 3),
            "量价确认": round(volp, 3), "技术赔率": round(odds, 3),
            "阶段匹配度": round(stage, 3), "波动容忍匹配": round(voltol, 3)}


# ── 第7层 EventScore / FinalEventScore ────────────────────────────────────────
def event_score(row: pd.Series) -> float:
    et = row.get("event_type")
    w = EVENT_TYPES.get(et)
    if not w:
        return 0.0
    return float(sum(w[k] * float(np.clip(row.get(k, 0), 0, 4)) for k in ("L", "N", "M", "P", "C")))


# 事件类型经验目标收益(持有窗口内, 用于净edge / CostFactor)
EVENT_TARGET = {
    "行业爆发型": 0.35, "商品供需错配型": 0.22, "第二春重估型": 0.28,
    "战争/地缘冲击型": 0.20, "自然灾变型": 0.18, "企业重大事故型": 0.20,
}
_COST = None
def _cost_map():
    """每只的最低往返摩擦(与订单无关部分: 2×(半价差+滑点), 转成小数收益)。"""
    global _COST
    if _COST is None:
        _COST = {}
        p = ROOT / "execution_cost_estimates.csv"
        if p.exists():
            try:
                d = pd.read_csv(p)
                for _, r in d.iterrows():
                    sp = float(pd.to_numeric(r.get("spread_bps", 25), errors="coerce") or 25)
                    sl = float(pd.to_numeric(r.get("slippage_bps", 3), errors="coerce") or 3)
                    rt = 2 * (sp / 2 + sl + 1.0)          # 往返 bps (含1bp佣金)
                    _COST[str(r["ticker"])] = rt / 1e4    # → 小数
            except Exception:
                pass
    return _COST


def cost_factor(row: pd.Series) -> float:
    """净edge反哺: 扣掉往返摩擦后, edge 相对目标收益还剩多少 → 0.6~1.0 的折扣。
    大盘股月级持有几乎不受影响; 高价差/低流动性且薄edge的候选被自动降级。"""
    tk = str(row.get("ticker", ""))
    rt = _cost_map().get(tk)
    if rt is None:
        return 1.0
    target = EVENT_TARGET.get(row.get("event_type"), 0.25)
    net_ratio = (target - rt) / target if target > 0 else 1.0   # 净edge占目标比例
    return float(np.clip(0.6 + net_ratio * 0.4, 0.6, 1.0))


_EDGE = None
def _edge_map():
    """从 8-K 事件研究(行业中性)得每种事件类型的"验证过edge乘数"。
    已证显著(t≥2)→ 按超额收益给小幅加成(1.0~1.15); 未证事件类型 → 0.97 小幅折价。"""
    global _EDGE
    if _EDGE is None:
        _EDGE = {}
        p = ROOT / "edgar_event_study.json"
        if p.exists():
            try:
                j = json.load(open(p)).get("neutralized_63d", {})
                for et, v in j.items():
                    sn = v.get("sector_neutral") or v.get("market_adj")
                    if sn and sn.get("n", 0) >= 50 and abs(sn.get("t", 0)) >= 2:
                        ret = float(sn["mean_ab_%"])           # 行业中性超额(%)
                        _EDGE[et] = float(np.clip(1.0 + (ret / 3.0) * 0.15, 1.0, 1.15))
            except Exception:
                pass
    return _EDGE


def edge_factor(row: pd.Series) -> float:
    """验证过edge反哺: 事件类型经 8-K 历史证过 → 加成; 未证 → 0.97 折价(诚实惩罚未验证)。"""
    em = _edge_map()
    if not em:
        return 1.0
    return em.get(row.get("event_type"), 0.97)


def final_event_score(row: pd.Series, macro_filter: float) -> dict:
    es = event_score(row)
    exec_s = execution_score(row)
    ef = exec_s / 100.0
    cf = cost_factor(row)
    edge = edge_factor(row)
    fes = es * ef * macro_filter * cf * edge
    return {"EventScore": round(es, 1), "ExecutionScore": exec_s,
            "ExecutionFilter": round(ef, 3), "MacroFilter": round(macro_filter, 3),
            "CostFactor": round(cf, 3), "EdgeFactor": round(edge, 3), "FinalEventScore": round(fes, 1)}


# ── 入池判定 (第7/8/9层) ──────────────────────────────────────────────────────
def classify(row: pd.Series, ef: float) -> str:
    L, N, M, P, C = (float(np.clip(row.get(k, 0), 0, 4)) for k in ("L", "N", "M", "P", "C"))
    strong_fwd = C >= 3
    # 利润发动机池
    if L >= 3 and N >= 3 and M >= 3 and P >= 3 and ef >= 0.70 and strong_fwd:
        return "利润发动机"
    # 事件型爆发池: 必要 L≥3,N≥2,P≥2,EF≥0.60; 充分 ≥3因子≥3
    n_ge3 = sum(x >= 3 for x in (L, N, M, P, C))
    if L >= 3 and N >= 2 and P >= 2 and ef >= 0.60 and n_ge3 >= 3:
        return "事件型爆发池"
    return "观察/不入池"


def position_plan(pool: str) -> dict:
    """第6层仓位轨。"""
    if pool == "利润发动机":
        return {"track": "利润发动机轨", "initial": "4%-6%", "confirm1": "8%-12%",
                "confirm2": "12%-18%", "high_conf": "18%-25%"}
    return {"track": "普通仓位轨", "initial": "1%-3%(观察)/3%-5%(试探)",
            "confirm1": "6%-10%", "confirm2": "10%-15%", "high_conf": "15%-20%"}


# ── 宏观过滤: 从现有宏观评分卡取 (缺则中性0.85) ────────────────────────────────
def macro_filter() -> float:
    # 优先用第1层宏观情报评分卡 (canyon_macro_intel.py)
    p = ROOT / "macro_intel_scorecard.json"
    if p.exists():
        try:
            mf = json.load(open(p)).get("macro_filter")
            if mf is not None:
                return float(mf)
        except Exception:
            pass
    for f in ("macro_regime_outlook.json", "macro_signals.json"):
        p = ROOT / f
        if p.exists():
            try:
                j = json.load(open(p))
                comp = j.get("composite", {}) if isinstance(j, dict) else {}
                bp = comp.get("bear_prob")
                if bp is not None:
                    return float(np.clip(1 - float(bp) / 100 * 0.6, 0.4, 1.0))
            except Exception:
                pass
    # HMM regime fallback
    hp = ROOT / "hmm_regime_daily.csv"
    if hp.exists():
        try:
            r = pd.read_csv(hp).iloc[-1]
            return 1.0 if str(r.get("regime", "")).upper() == "BULL" else 0.6
        except Exception:
            pass
    return 0.85


# ── 宏观链条 → 选股倾斜(L1 受益/风险链条真正驱动 L7 选股)──────────────────────
def _macro_chains():
    """从 L1 情报评分卡读 受益链条/风险链条(GICS 行业名)。"""
    p = ROOT / "macro_intel_scorecard.json"
    ben, risk = set(), set()
    if p.exists():
        try:
            j = json.load(open(p))
            ben = {str(x).strip() for x in (j.get("重点受益链条") or [])}
            risk = {str(x).strip() for x in (j.get("风险链条") or [])}
        except Exception:
            pass
    return ben, risk


def _sector_map():
    """ticker → GICS 行业(用于链条匹配)。"""
    for f in ("alpha_scores.csv", "event_candidates.csv"):
        p = ROOT / f
        if p.exists():
            try:
                d = pd.read_csv(p)
                if "sector" in d.columns and "ticker" in d.columns:
                    return {str(t): str(s) for t, s in zip(d["ticker"], d["sector"])}
            except Exception:
                pass
    return {}


def _chain_tilt(sector, ben, risk):
    """受益链条 ×1.10, 风险链条 ×0.90, 否则 1.0。返回 (乘子, 标签)。"""
    if sector in ben:
        return 1.10, "受益链条+"
    if sector in risk:
        return 0.90, "风险链条−"
    return 1.0, ""


# ── 主流程 ────────────────────────────────────────────────────────────────────
def run(pool_csv="event_pool.csv") -> pd.DataFrame:
    pp = ROOT / pool_csv
    if not pp.exists():
        return pd.DataFrame()
    pool = pd.read_csv(pp)
    mf = macro_filter()
    ben, risk = _macro_chains()                          # L1 宏观链条
    secmap = _sector_map()
    prices = None
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        if (ROOT / f).exists():
            prices = pd.read_csv(ROOT / f, index_col=0, parse_dates=True); break

    rows = []
    for _, r in pool.iterrows():
        r = r.copy()
        if prices is not None:
            for k, v in auto_execution_components(prices, str(r["ticker"])).items():
                if pd.isna(r.get(k, np.nan)):
                    r[k] = v
        sec = secmap.get(str(r["ticker"]), str(r.get("sector", "")))
        tilt, tilt_lbl = _chain_tilt(sec, ben, risk)     # 宏观链条 → 个股倾斜
        sc = final_event_score(r, mf * tilt)             # 个股专属宏观过滤(标量×链条倾斜)
        pool_cls = classify(r, sc["ExecutionFilter"])
        et = r.get("event_type")
        exitt = EXIT_TEMPLATES.get(et, {})
        rows.append({
            "ticker": r["ticker"], "event_type": et,
            **{k: r.get(k) for k in ("L", "N", "M", "P", "C")},
            **sc, "pool": pool_cls,
            "hold_window": EVENT_TYPES.get(et, {}).get("hold", ""),
            "exit_trade%": exitt.get("trade"), "exit_logic%": exitt.get("logic"),
            "exit_force": exitt.get("force"),
            "position_track": position_plan(pool_cls)["track"],
            "macro_chain": tilt_lbl,                          # 宏观链条倾斜(受益+/风险−)
            "note": (str(r.get("note", "")) + (f" ·{tilt_lbl}" if tilt_lbl else "")),
        })
    df = pd.DataFrame(rows).sort_values("FinalEventScore", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / "event_candidates.csv", index=False)
    return df


def main():
    print("=" * 60)
    print("Canyon 事件驱动主动投资系统 — 打分引擎")
    print("=" * 60)
    df = run()
    if df.empty:
        print("  event_pool.csv 不存在或为空 — 请先建股票池 (见 event_pool_template)")
        return
    mf = macro_filter()
    n_pe = int((df["pool"] == "利润发动机").sum())
    n_ev = int((df["pool"] == "事件型爆发池").sum())
    sw = scarcity_switch(n_pe + n_ev)
    print(f"  宏观过滤 MacroFilter = {mf:.2f}")
    print(f"  机会稀缺度: {sw['state']} → 建议总进攻仓位 {sw['attack_pos']}")
    print(f"    ({sw['action']})")
    print(f"  利润发动机 {n_pe} 只 · 事件型爆发池 {n_ev} 只 · 共 {len(df)} 只候选")
    print("\n  Top 候选 (按 FinalEventScore):")
    for _, r in df.head(8).iterrows():
        print(f"    {r['ticker']:6} {r['event_type'] or '-':12} FES={r['FinalEventScore']:>6}  "
              f"{r['pool']}  hold={r['hold_window']}")
    print(f"\n  → event_candidates.csv")


if __name__ == "__main__":
    main()
