#!/usr/bin/env python3
"""
canyon_event_detect.py — 第3层 + 事件自动侦测 (v2 精准版)
=========================================================
让系统"自己发掘": 从新闻自动发现符合六种事件类型的机会,并预填 L/N/M/P/C 初判。
v2 相对 v1 的准确度改进:
  1. 加权关键词签名   —— 高辨识度词(opec/missile/data center)权重高, 泛词(power/orders)权重低
  2. 否定/条件语境过滤 —— "no war"/"avoid recall"/"denies" 不计入
  3. 主体匹配要求      —— 事件必须真的是该 ticker 的主线(命中标题 / matched_terms / 公司名), 否则降权
  4. 近期性加权        —— 近 7 天新闻权重 1.0, 越旧越衰减
  5. 结构化情绪        —— 用 impact_score / market_tone / bullish_reasons 而非裸字符串
  6. 去重              —— 同一故事多家转发只算一次
  7. 置信度            —— 只有 confidence≥阈值 才算"系统确信发现", 低置信标注"待人工确认"

输出: auto_event_candidates.csv (含 detect_confidence 列)
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# 事件类型关键词签名: (词, 权重). 权重越高辨识度越强(越不可能误判)
EVENT_KW = {
    "战争/地缘冲击型": [
        ("missile", 3), ("airstrike", 3), ("invasion", 3), ("nato", 3), ("troop", 3),
        ("sanction", 2), ("geopolit", 2), ("military", 2), ("conflict", 2), ("drone strike", 3),
        ("strait", 2), ("ceasefire", 2), ("defense budget", 3), ("war", 1), ("attack", 1),
    ],
    "商品供需错配型": [
        ("opec", 3), ("crude", 2), ("lng", 3), ("uranium", 3), ("tanker", 3), ("freight rate", 3),
        ("production cut", 3), ("supply cut", 3), ("inventory draw", 3), ("refining margin", 3),
        ("fertilizer", 3), ("copper", 2), ("crack spread", 3), ("oil price", 2), ("gas price", 2),
        ("shortage", 2), ("oil", 1), ("gas", 1),
    ],
    "行业爆发型": [
        ("data center", 3), ("artificial intelligence", 3), ("semiconductor", 3), ("gpu", 3),
        ("hyperscaler", 3), ("backlog", 2), ("record demand", 3), ("capex cycle", 3),
        ("liquid cooling", 3), ("grid buildout", 3), ("ai ", 2), ("chip", 2), ("cloud", 1),
        ("orders", 1), ("power demand", 2), ("bookings", 2),
    ],
    "第二春重估型": [
        ("turnaround", 3), ("restructur", 2), ("spin off", 3), ("spin-off", 3), ("guidance raise", 3),
        ("raised guidance", 3), ("new product", 2), ("new customer", 2), ("reinvent", 2),
        ("pivot", 2), ("relaunch", 2), ("undervalued", 2), ("discount", 1), ("re-rating", 3),
        ("margin expansion", 3), ("cost cutting", 2),
    ],
    "自然灾变型": [
        ("hurricane", 3), ("earthquake", 3), ("wildfire", 3), ("flood", 2), ("catastrophe", 3),
        ("natural disaster", 3), ("storm", 2), ("outage", 2), ("drought", 2),
    ],
    "企业重大事故型": [
        ("recall", 3), ("explosion", 3), ("contamination", 3), ("data breach", 3), ("fraud", 3),
        ("sec probe", 3), ("investigation", 2), ("lawsuit", 1), ("shutdown", 2), ("accident", 2),
        ("plant fire", 3), ("safety failure", 3), ("regulatory probe", 3),
    ],
}
EVENT_L_BASE = {"行业爆发型": 3.2, "战争/地缘冲击型": 3.0, "第二春重估型": 2.8,
                "商品供需错配型": 2.8, "自然灾变型": 2.5, "企业重大事故型": 2.4}
# 否定语境: 若关键词前后出现这些词, 该命中作废
NEGATORS = ["no ", "not ", "avoid", "without", "denies", "deny", "unlikely", "no sign",
            "ruled out", "ease", "eases", "easing", "resolve", "de-escalat", "averted", "no evidence"]


def load_news():
    p = ROOT / "stock_news.json"
    if not p.exists():
        return {}
    j = json.load(open(p))
    news = j.get("news", {}) if isinstance(j, dict) else {}
    return news if isinstance(news, dict) else {}


# 行业 → 该行业"合理"的事件类型 (行业合理性门, 防泛词把金融/防御股误判为行业爆发型)
SECTOR_PLAUSIBLE = {
    "Technology":             {"行业爆发型", "第二春重估型", "企业重大事故型"},
    "Communication Services": {"行业爆发型", "第二春重估型", "企业重大事故型"},
    "Energy":                 {"商品供需错配型", "战争/地缘冲击型", "自然灾变型", "企业重大事故型"},
    "Materials":              {"商品供需错配型", "行业爆发型", "自然灾变型", "企业重大事故型"},
    "Industrials":            {"行业爆发型", "战争/地缘冲击型", "第二春重估型", "企业重大事故型"},
    "Utilities":              {"行业爆发型", "自然灾变型", "第二春重估型"},
    "Health Care":            {"第二春重估型", "企业重大事故型", "行业爆发型"},
    "Financials":             {"第二春重估型", "战争/地缘冲击型", "企业重大事故型"},
    "Real Estate":            {"第二春重估型", "自然灾变型", "战争/地缘冲击型"},
    "Consumer Discretionary": {"第二春重估型", "行业爆发型", "企业重大事故型"},
    "Consumer Staples":       {"第二春重估型", "自然灾变型", "企业重大事故型"},
}


def sector_map():
    for f in ("alpha_scores.csv", "regime_ml_scores.csv"):
        p = ROOT / f
        if p.exists():
            d = pd.read_csv(p)
            if "ticker" in d.columns and "sector" in d.columns:
                return {str(r["ticker"]): str(r["sector"]) for _, r in d.iterrows()}
    return {}


def catalyst_map():
    """N(催化临近度) 数据: {ticker: (density1-4, note)}。
    同时给两类活催化剂记分:
      · 临近财报 (days_until 越小密度越高)
      · 刚兑现的强超预期 (days_until 近期为负 且 surprise 大 → 财报后漂移 PEAD 窗口)"""
    p = ROOT / "earnings_calendar.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    if "ticker" not in d.columns or "days_until" not in d.columns:
        return {}
    surp_col = "surprise_pct_last" if "surprise_pct_last" in d.columns else None
    out = {}
    for _, r in d.iterrows():
        tk = str(r["ticker"])
        du = pd.to_numeric(r.get("days_until"), errors="coerce")
        if pd.isna(du):
            continue
        du = float(du)
        surp = float(pd.to_numeric(r.get(surp_col), errors="coerce") or 0) if surp_col else 0.0
        if du >= 0:                                       # 临近财报
            dens = 4 if du <= 5 else 3 if du <= 21 else 2 if du <= 45 else 1
            note = f"财报{int(du)}天后"
        elif du >= -10 and surp >= 5:                     # 刚爆强超预期 → 漂移窗口
            dens = 4 if surp >= 15 else 3
            note = f"财报后漂移(超预期{surp:.0f}%,{int(-du)}天前)"
        elif du >= -10 and surp >= 2:
            dens = 2; note = f"财报后({int(-du)}天前)"
        else:
            dens = 1; note = "无临近催化"
        out[tk] = (dens, note)
    return out


def prices():
    for f in ("sp500_price_history_deep.csv", "sp500_price_cache.csv"):
        p = ROOT / f
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    return pd.DataFrame()


def _negated(text: str, kw: str) -> bool:
    """关键词是否处在否定语境 (词前 40 字符内出现否定词)"""
    for m in re.finditer(re.escape(kw), text):
        pre = text[max(0, m.start() - 40):m.start()]
        if not any(neg in pre for neg in NEGATORS):
            return False   # 至少一次非否定命中 → 有效
    return True            # 所有命中都被否定


def _recency(pub: str, now: datetime) -> float:
    """近期性权重: 近7天=1.0, 30天≈0.5, 越旧越低"""
    try:
        d = datetime.strptime(str(pub)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age = (now - d).days
        return float(np.clip(1.2 * np.exp(-age / 22.0), 0.15, 1.0))
    except Exception:
        return 0.5


def detect():
    news = load_news()
    catmap = catalyst_map()
    px = prices()
    secmap = sector_map()
    now = datetime.now(timezone.utc)
    rows = []
    for tk, items in news.items():
        if not isinstance(items, list) or not items:
            continue
        # 去重: 同标题只留一条 (多家转发)
        seen, uniq = set(), []
        for it in items:
            key = re.sub(r"\W+", "", str(it.get("title", "")).lower())[:60]
            if key and key not in seen:
                seen.add(key); uniq.append(it)
        if not uniq:
            continue

        # 逐事件类型加权打分 (含否定过滤 + 主体匹配 + 近期性)
        et_score = {et: 0.0 for et in EVENT_KW}
        et_hits = {et: 0 for et in EVENT_KW}
        et_anchor = {et: False for et in EVENT_KW}   # 是否命中过高辨识度锚词(权重≥3)
        matched_any = False
        for it in uniq:
            title = str(it.get("title", "")).lower()
            body = title + " " + str(it.get("summary", "")).lower()
            rec = _recency(it.get("published", ""), now)
            # 主体匹配: ticker 或公司词出现在标题 / matched_terms 非空 → 该事件确属本股
            subj = 1.0 if (tk.lower() in title or it.get("matched_terms")) else 0.55
            for et, kws in EVENT_KW.items():
                for kw, w in kws:
                    if kw in body and not _negated(body, kw):
                        # 标题命中比正文命中更能代表主线
                        loc = 1.3 if kw in title else 1.0
                        et_score[et] += w * rec * subj * loc
                        et_hits[et] += 1
                        if w >= 3:
                            et_anchor[et] = True
                        matched_any = True
        if not matched_any:
            continue
        et, sc = max(et_score.items(), key=lambda x: x[1])
        hit = et_hits[et]
        anchored = et_anchor[et]                       # 该类型是否有硬锚词支撑
        if sc <= 0:
            continue

        # 结构化情绪 (impact_score + market_tone + reasons)
        imps = [float(it.get("impact_score", 0) or 0) for it in uniq]
        impact = float(np.mean(imps)) if imps else 0.0
        tones = [str(it.get("market_tone", "")).upper() for it in uniq]
        pos = sum(t == "POSITIVE" for t in tones) + sum(bool(it.get("bullish_reasons")) for it in uniq)
        neg = sum(t == "NEGATIVE" for t in tones) + sum(bool(it.get("bearish_reasons")) for it in uniq)
        narrative = abs(pos - neg) / max(len(uniq), 1)

        # 第3层信号
        kw_heat = float(np.clip(sc / 6.0, 0, 4))                 # 加权热度→0-4
        cat_density, cat_note = catmap.get(tk, (1, "无临近催化"))
        catalyst = cat_note

        # 预填 L/N/M/P/C (0-4)
        L = float(np.clip(EVENT_L_BASE[et] + narrative * 0.6 + (impact / 6.0), 0, 4))
        N = float(cat_density)
        M = 2.0
        if tk in px.columns:
            s = px[tk].dropna()
            if len(s) > 60:
                dd = float(s.iloc[-1] / s.tail(252).max())
                M = float(np.clip(1 + (1 - dd) * 4, 0, 4))       # 回撤越深 M 越高(反弹空间)
        # P 映射纯度: 加权分越集中于单一事件类型 + 主体命中 → 越纯
        share = sc / max(sum(et_score.values()), 1e-9)           # 该类型占全部事件分的比例
        P = float(np.clip(1.5 + share * 2.5, 0, 4))
        C = float(np.clip(1.5 + (pos - neg) * 0.5 + impact / 8.0, 0, 4))

        # 侦测置信度: 加权分强度 × 事件类型集中度 × 情绪一致性 × 近期覆盖
        confidence = float(np.clip(
            (min(sc, 12) / 12.0) * 0.45 + share * 0.30 + min(narrative, 1) * 0.15 +
            (0.10 if hit >= 3 else 0.04), 0, 1))
        # 无高辨识度锚词(只撞泛词 ai/cloud/orders 等) → 封顶 0.5, 防金融/防御股被泛词误判为行业爆发型
        if not anchored:
            confidence = min(confidence, 0.5)
        # 行业合理性门: 事件类型与该股所属行业不符 → 大幅降权(如金融股被判行业爆发型)
        sec = secmap.get(tk, "")
        plausible = SECTOR_PLAUSIBLE.get(sec)
        sector_ok = (plausible is None) or (et in plausible)
        if not sector_ok:
            confidence = min(confidence * 0.45, 0.45)
        conf_tag = ("高" if confidence >= 0.6 else "中" if confidence >= 0.38 else "低(待人工确认)")
        if not anchored:
            conf_tag += "·仅泛词"
        if not sector_ok:
            conf_tag += f"·行业存疑({sec})"

        rows.append({
            "ticker": tk, "event_type": et,
            "L": round(L, 1), "N": round(N, 1), "M": round(M, 1), "P": round(P, 1), "C": round(C, 1),
            "detect_confidence": round(confidence, 2), "confidence_tag": conf_tag,
            "关键词热度": round(kw_heat, 1), "叙事强度": round(narrative * 4, 1),
            "催化密度": cat_density, "next_earnings_days": catalyst,
            "n_news": len(uniq), "impact": round(impact, 1), "event_share": round(share, 2),
            "催化清晰度": round(float(np.clip(0.4 + cat_density * 0.12, 0, 1)), 2),
            "逻辑确认度": round(float(np.clip(0.4 + narrative * 0.5, 0, 1)), 2),
            "失效清晰度": 0.6,
            "note": f"侦测:{et} 加权分{sc:.1f}/{hit}命中/{len(uniq)}条·置信{conf_tag}",
        })
    df = pd.DataFrame(rows)
    df = merge_edgar(df, px)                              # 并入真实 8-K 事件
    if not df.empty:
        df = df.sort_values(["detect_confidence", "L"], ascending=[False, False]).reset_index(drop=True)
        df.to_csv(ROOT / "auto_event_candidates.csv", index=False)
    return df


def merge_edgar(df: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """并入 SEC EDGAR 8-K: 真实备案的重大事件比关键词更客观。
    已在 df 的 → 提置信+(高严重度时)采用 8-K 事件类型; 不在的 → 新增 8-K 候选。"""
    p = ROOT / "edgar_events.csv"
    if not p.exists():
        return df
    try:
        e = pd.read_csv(p)
    except Exception:
        return df
    e = e[pd.to_numeric(e.get("8k_severity", 0), errors="coerce").fillna(0) >= 2]
    if e.empty:
        return df
    existing = set(df["ticker"].astype(str)) if not df.empty else set()
    add = []
    for _, r in e.iterrows():
        tk = str(r["ticker"]); sev = float(r["8k_severity"]); et = str(r.get("8k_event_type", "") or "")
        if not et:
            continue
        if tk in existing:                               # 已有新闻侦测 → 8-K 确认, 提置信
            i = df.index[df["ticker"].astype(str) == tk][0]
            df.at[i, "detect_confidence"] = float(np.clip(df.at[i, "detect_confidence"] + 0.12 + sev * 0.03, 0, 0.95))
            if sev >= 3:                                 # 高严重度 8-K → 采用其客观事件类型
                df.at[i, "event_type"] = et
            df.at[i, "note"] = str(df.at[i, "note"]) + f"·8-K确认(严重度{sev:.0f}:{r.get('8k_desc','')})"
            df.at[i, "confidence_tag"] = "高·8-K" if df.at[i, "detect_confidence"] >= 0.6 else df.at[i, "confidence_tag"]
        else:                                            # 纯 8-K 事件 → 新候选
            M = 2.0
            if tk in px.columns:
                s = px[tk].dropna()
                if len(s) > 60:
                    dd = float(s.iloc[-1] / s.tail(252).max()); M = float(np.clip(1 + (1 - dd) * 4, 0, 4))
            conf = float(np.clip(0.45 + sev * 0.11, 0, 0.85))
            add.append({
                "ticker": tk, "event_type": et,
                "L": round(float(np.clip(2.4 + sev * 0.3, 0, 4)), 1), "N": 4.0, "M": round(M, 1),
                "P": 3.0, "C": 1.5, "detect_confidence": round(conf, 2),
                "confidence_tag": ("高·8-K" if conf >= 0.6 else "中·8-K"),
                "关键词热度": 0.0, "叙事强度": 0.0, "催化密度": 4, "next_earnings_days": None,
                "n_news": 0, "impact": 0.0, "event_share": 1.0,
                "催化清晰度": round(float(np.clip(0.5 + sev * 0.1, 0, 1)), 2), "逻辑确认度": 0.6, "失效清晰度": 0.6,
                "note": f"8-K事件:{r.get('8k_desc','')}→{et}(严重度{sev:.0f},{r.get('latest_8k_date','')})",
            })
    if add:
        df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
    return df


def main():
    print("=" * 60)
    print("第3层 + 事件自动侦测 v2 — 加权/否定/主体匹配/置信度")
    print("=" * 60)
    df = detect()
    if df.empty:
        print("  无新闻或未命中任何事件类型")
        return
    hi = df[df["detect_confidence"] >= 0.6]
    md = df[(df["detect_confidence"] >= 0.38) & (df["detect_confidence"] < 0.6)]
    print(f"  扫描 {df['n_news'].sum()} 条新闻 → {len(df)} 只带事件信号")
    print(f"  高置信 {len(hi)} · 中置信 {len(md)} · 低置信 {len(df)-len(hi)-len(md)}")
    for et in df["event_type"].unique():
        sub = df[df["event_type"] == et]
        names = ", ".join(f"{r.ticker}(L{r.L},置信{r.detect_confidence})" for _, r in sub.head(5).iterrows())
        print(f"    {et:14} {len(sub)}只: {names}")
    print("\n  → auto_event_candidates.csv (高置信自动并入; 低置信标注待人工确认)")


if __name__ == "__main__":
    main()
