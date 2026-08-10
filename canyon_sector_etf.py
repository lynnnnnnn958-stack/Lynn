#!/usr/bin/env python3
"""
canyon_sector_etf.py — 板块龙头 ETF 指标
=========================================
和 canyon_sector_rotation.py(成分股聚合)互补: 这里直接用板块龙头 ETF 的价格,
算每个可交易 ETF 的动量/相对强度/趋势 —— 更直观、可直接买卖。

11 个 SPDR 行业 ETF + 半导体 SMH/SOXX, 以 SPY 为相对强度基准。
每个 ETF: 最新价, 1M/3M/6M 动量, 相对强度(vs SPY 3M), 站上50/200日线, 趋势方向。

输出: sector_etf_indicators.csv
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

ETFS = {
    "XLK": "科技 Technology", "XLF": "金融 Financials", "XLE": "能源 Energy",
    "XLV": "医疗 Health Care", "XLI": "工业 Industrials", "XLP": "必需消费 Staples",
    "XLU": "公用 Utilities", "XLRE": "地产 Real Estate", "XLB": "材料 Materials",
    "XLC": "通信 Communication", "XLY": "可选消费 Discretionary",
    "SMH": "半导体 Semis", "SOXX": "半导体 Semis(SOXX)",
}
BENCH = "SPY"


def fetch(tickers):
    import yfinance as yf
    data = yf.download(tickers, period="1y", interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data[["Close"]].rename(columns={"Close": tickers[0]})
    return px.dropna(how="all")


def _ret(s, days):
    s = s.dropna()
    if len(s) <= days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1)


def run():
    tickers = list(ETFS) + [BENCH]
    try:
        px = fetch(tickers)
    except Exception as e:
        print(f"  ETF 抓取失败: {e}")
        return pd.DataFrame()
    if px.empty or BENCH not in px.columns:
        print("  无 ETF 价格数据"); return pd.DataFrame()

    spy3 = _ret(px[BENCH], 63)
    rows = []
    for tk, name in ETFS.items():
        if tk not in px.columns:
            continue
        s = px[tk].dropna()
        if len(s) < 130:
            continue
        m1, m3, m6 = _ret(s, 21), _ret(s, 63), _ret(s, 126)
        rs = (m3 - spy3) if (m3 == m3 and spy3 == spy3) else np.nan
        p_now = float(s.iloc[-1])
        a50 = p_now > float(s.tail(50).mean())
        a200 = p_now > float(s.tail(200).mean())
        accel = m1 - m3 / 3.0
        direction = "▲ 加速" if accel > 0.01 else "▼ 退潮" if accel < -0.01 else "→ 平"
        trend = "多头(站上50&200)" if (a50 and a200) else "偏多(站上50)" if a50 else "偏空(跌破50)"
        rows.append({"etf": tk, "name": name, "price": round(p_now, 2),
                     "mom_1m": round(m1, 4), "mom_3m": round(m3, 4), "mom_6m": round(m6, 4),
                     "rel_strength": round(rs, 4) if rs == rs else np.nan,
                     "above_50d": bool(a50), "above_200d": bool(a200),
                     "direction": direction, "trend": trend})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("rel_strength", ascending=False, na_position="last").reset_index(drop=True)
    df["rank"] = df.index + 1
    df.attrs["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df.to_csv(ROOT / "sector_etf_indicators.csv", index=False)
    return df


def main():
    print("=" * 62)
    print("板块龙头 ETF 指标 — 可交易工具视角")
    print("=" * 62)
    df = run()
    if df.empty:
        print("  无输出"); return
    print(f"  {'ETF':<6}{'板块':<22}{'价格':>8}{'1M':>7}{'3M':>7}{'相对强度':>9}  趋势")
    for _, r in df.iterrows():
        rs = r["rel_strength"]
        print(f"  {r['etf']:<6}{r['name']:<20}{r['price']:>8.2f}"
              f"{r['mom_1m']*100:>+6.1f}%{r['mom_3m']*100:>+6.1f}%"
              f"{(rs*100 if rs==rs else 0):>+8.1f}%  {r['direction']} · {r['trend']}")
    print("\n  → sector_etf_indicators.csv")


if __name__ == "__main__":
    main()
