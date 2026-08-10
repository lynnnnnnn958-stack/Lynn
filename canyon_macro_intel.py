#!/usr/bin/env python3
"""
canyon_macro_intel.py — 第1层 宏观情报评分卡 (全系统总开关)
===========================================================
按《美股主动投资系统手册》第1层。自动从新闻 + 板块轮动生成宏观评分卡,减少人工:
  · 六大情报模块扫描 (战争/地缘、关税/制裁、流动性/利率、行业CapEx、商品供需、事故/灾变)
    → 每个模块 0-4 热度分 (新闻提及频率 × 情绪)
  · 板块轮动 → 重点受益链条 (领涨) / 风险链条 (落后)
  · 输出: 宏观模式、风格约束、总仓位制度、重点受益链条、风险链条、监控优先级
  · 激活对应事件池 (哪些事件类型现在值得重点找机会)

输出: macro_intel_scorecard.json  (供作战台/引擎读取)
注: 关键词启发式为自动初判,精细判断 (是否升级/通道受阻) 仍由人按手册确认。
"""
from __future__ import annotations
import json, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── 六大情报模块 → 关键词 (英文新闻) + 触发的事件类型 ─────────────────────────────
MODULES = {
    "战争/地缘风险": {
        "kw": ["war", "military", "missile", "strike", "invasion", "conflict", "sanction on",
               "geopolit", "tension", "defense", "troop", "ceasefire", "attack", "nato", "iran",
               "russia", "ukraine", "israel", "taiwan", "houthi", "strait"],
        "event": "战争/地缘冲击型", "risk_on": False},
    "关税/制裁/监管": {
        "kw": ["tariff", "sanction", "export control", "ban on", "trade war", "restrict",
               "antitrust", "regulat", "ustr", "bis ", "commerce department", "blacklist"],
        "event": "第二春重估型", "risk_on": False},
    "流动性/利率/美元": {
        "kw": ["fed", "interest rate", "rate cut", "rate hike", "inflation", "cpi", "pce",
               "payroll", "yield", "treasury", "liquidity", "dollar", "recession", "soft landing"],
        "event": None, "risk_on": None},
    "行业CapEx/景气": {
        "kw": ["capex", "capital expenditure", "data center", "ai ", "artificial intelligence",
               "chip", "semiconductor", "cloud", "power", "grid", "cooling", "guidance raise",
               "record demand", "backlog", "orders surge", "expansion", "fab"],
        "event": "行业爆发型", "risk_on": True},
    "商品供需/运输": {
        "kw": ["oil", "crude", "opec", "natural gas", "lng", "fertilizer", "tanker", "freight",
               "shipping", "supply cut", "production cut", "inventory", "commodity", "copper",
               "uranium", "shortage", "price surge"],
        "event": "商品供需错配型", "risk_on": None},
    "事故/灾变/供应链": {
        "kw": ["hurricane", "earthquake", "flood", "wildfire", "outage", "recall", "explosion",
               "accident", "shutdown", "disaster", "power grid", "plant fire", "disruption",
               "force majeure", "contamination"],
        "event": "自然灾变型", "risk_on": None},
}


def load_news_items():
    p = ROOT / "stock_news.json"
    if not p.exists():
        return []
    try:
        j = json.load(open(p))
    except Exception:
        return []
    items = []
    news = j.get("news", j) if isinstance(j, dict) else j
    if isinstance(news, dict):
        for _tk, lst in news.items():
            if isinstance(lst, list):
                items.extend(lst)
    elif isinstance(news, list):
        items = news
    return items


def scan_module(items, kws):
    hits, tone = 0, 0.0
    for it in items:
        text = (str(it.get("title", "")) + " " + str(it.get("summary", ""))).lower()
        m = sum(1 for k in kws if k in text)
        if m:
            hits += 1
            t = str(it.get("market_tone", "")).lower()
            tone += (1 if "bull" in t or "pos" in t else -1 if "bear" in t or "neg" in t else 0)
    return hits, tone


def heat_score(hits, total):
    """0-4 热度 (手册关键词热度量化规则)."""
    if total == 0 or hits == 0:
        return 0
    frac = hits / total
    if frac < 0.02: return 1
    if frac < 0.05: return 2
    if frac < 0.10: return 3
    return 4


def chains():
    p = ROOT / "sector_rotation_scores.csv"
    lead, lag = [], []
    if p.exists():
        s = pd.read_csv(p)
        if "rotation_label" in s.columns:
            lead = s[s["rotation_label"] == "LEADER"]["theme"].tolist()[:5]
            lag = s[s["rotation_label"] == "LAGGARD"]["theme"].tolist()[:5]
        elif "rotation_score" in s.columns:
            s = s.sort_values("rotation_score", ascending=False)
            lead = s.head(4)["theme"].tolist(); lag = s.tail(3)["theme"].tolist()
    return lead, lag


def fred_stress():
    """FRED 硬宏观压力(0-4): 曲线倒挂 + 信用利差走阔 + 初请上升。客观锚, 不靠新闻。
    返回 (stress 0-4, 描述列表)。"""
    p = ROOT / "fred_macro_data.csv"
    if not p.exists():
        return None, []
    try:
        f = pd.read_csv(p)
    except Exception:
        return None, []
    def last(col):
        if col not in f.columns:
            return None, None
        v = pd.to_numeric(f[col], errors="coerce").dropna()
        if v.empty:
            return None, None
        return float(v.iloc[-1]), (float(v.iloc[-63]) if len(v) > 63 else float(v.iloc[0]))
    stress, notes = 0.0, []
    curve, curve_p = last("T10Y2Y")
    if curve is not None:
        if curve < 0: stress += 1.5; notes.append(f"收益率曲线倒挂({curve:.2f})")
        elif curve < 0.2: stress += 0.6; notes.append(f"曲线趋平({curve:.2f})")
    hy, hy_p = last("BAMLH0A0HYM2")
    if hy is not None:
        if hy > 5.5: stress += 1.5; notes.append(f"信用利差高企({hy:.1f}%)")
        elif hy > 4.0: stress += 0.8; notes.append(f"信用利差走阔({hy:.1f}%)")
        elif hy_p and hy > hy_p * 1.2: stress += 0.5; notes.append("信用利差快速上行")
    claims, claims_p = last("ICSA")
    if claims is not None and claims_p and claims > claims_p * 1.15:
        stress += 0.8; notes.append("初请失业金快速上升")
    return float(np.clip(stress, 0, 4)), notes


def macro_mode():
    """从 HMM / 宏观 composite 判断风险偏好模式 → 总仓位制度. FRED 硬数据校正。"""
    base = None
    for f in ("macro_regime_outlook.json",):
        p = ROOT / f
        if p.exists():
            try:
                bp = json.load(open(p)).get("composite", {}).get("bear_prob")
                if bp is not None:
                    bp = float(bp)
                    if bp < 25: base = ("进攻", "85%-100%", 1.0)
                    elif bp < 50: base = ("常规", "60%-85%", 0.85)
                    elif bp < 70: base = ("防守", "40%-60%", 0.65)
                    else: base = ("危机", "30%-50%", 0.5)
            except Exception:
                pass
    if base is None:
        hp = ROOT / "hmm_regime_daily.csv"
        if hp.exists():
            try:
                r = pd.read_csv(hp).iloc[-1]
                if str(r.get("regime", "")).upper() == "BULL":
                    base = ("常规偏进攻", "70%-95%", 0.9)
            except Exception:
                pass
    if base is None:
        base = ("常规", "60%-85%", 0.85)
    # FRED 硬压力校正: 压力≥2 时下调一档风险偏好(客观数据否决过度乐观)
    st, _ = fred_stress()
    if st is not None and st >= 2:
        mode, pos, mf = base
        return f"{mode}(FRED降档)", pos, float(round(mf * 0.8, 2))
    return base


def _macro_mode_legacy():
    """从 HMM / 宏观 composite 判断风险偏好模式 → 总仓位制度."""
    for f in ("macro_regime_outlook.json",):
        p = ROOT / f
        if p.exists():
            try:
                bp = json.load(open(p)).get("composite", {}).get("bear_prob")
                if bp is not None:
                    bp = float(bp)
                    if bp < 25: return "进攻", "85%-100%", 1.0
                    if bp < 50: return "常规", "60%-85%", 0.85
                    if bp < 70: return "防守", "40%-60%", 0.65
                    return "危机", "30%-50%", 0.5
            except Exception:
                pass
    hp = ROOT / "hmm_regime_daily.csv"
    if hp.exists():
        try:
            r = pd.read_csv(hp).iloc[-1]
            if str(r.get("regime", "")).upper() == "BULL":
                return "常规偏进攻", "70%-95%", 0.9
        except Exception:
            pass
    return "常规", "60%-85%", 0.85


def main():
    items = load_news_items()
    total = max(len(items), 1)
    modules_out = {}
    for name, cfg in MODULES.items():
        hits, tone = scan_module(items, cfg["kw"])
        h = heat_score(hits, total)
        modules_out[name] = {"heat": h, "hits": hits, "tone": round(tone, 1),
                             "event_type": cfg["event"]}
    # 激活: 相对择优, 只取热度最高的前 3 个模块(且 heat≥3), 反映"当下真正主导的主线"
    # (新闻覆盖扩大后, 绝对阈值会让几乎所有模块都过线; 择优才有区分度)
    ranked = sorted(modules_out.items(), key=lambda x: (-x[1]["heat"], -x[1]["hits"]))
    active_events = []
    for name, v in ranked[:3]:
        if v["heat"] >= 3 and v["event_type"]:
            active_events.append(v["event_type"])
    lead, lag = chains()
    mode, pos, mf = macro_mode()
    fred_st, fred_notes = fred_stress()

    out = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "news_scanned": len(items),
        "宏观模式": mode,
        "总仓位制度": pos,
        "macro_filter": round(mf, 2),
        "FRED硬压力": round(fred_st, 1) if fred_st is not None else None,
        "FRED信号": fred_notes or (["曲线正常·信用利差平稳·就业稳健"] if fred_st is not None else []),
        "重点受益链条": lead,
        "风险链条": lag,
        "情报模块": modules_out,
        "激活事件池": sorted(set(active_events)),
        "监控优先级": [n for n, v in sorted(modules_out.items(), key=lambda x: -x[1]["heat"]) if v["heat"] >= 2],
    }
    json.dump(out, open(ROOT / "macro_intel_scorecard.json", "w"), ensure_ascii=False, indent=2)

    print("=" * 56)
    print("第1层 宏观情报评分卡")
    print("=" * 56)
    print(f"  扫描新闻 {len(items)} 条")
    print(f"  宏观模式: {mode}  → 总仓位 {pos}  (MacroFilter {mf})")
    print(f"  重点受益链条: {', '.join(lead) or '—'}")
    print(f"  风险链条: {', '.join(lag) or '—'}")
    print("  情报热度 (0-4):")
    for n, v in modules_out.items():
        bar = "█" * v["heat"] + "·" * (4 - v["heat"])
        print(f"    {n:14} {bar}  {v['heat']}  ({v['hits']}条 tone={v['tone']:+.0f})")
    if active_events:
        print(f"  → 激活事件池: {', '.join(sorted(set(active_events)))}")
    print("  → macro_intel_scorecard.json")


if __name__ == "__main__":
    main()
