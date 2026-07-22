#!/usr/bin/env python3
"""
Canyon v9 Step 115 - Paper NAV, return attribution, and slippage model.

Research-only. No broker connection. No live orders.

Outputs:
  paper_nav_curve.csv
  live_nav_manual_template.csv
  live_nav_curve.csv
  return_attribution_by_ticker.csv
  return_attribution_by_signal.csv
  slippage_model_report.csv
  live_nav_attribution_slippage_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    MODEL_ACCOUNT_VALUE,
    ROOT,
    clean_ticker,
    df_to_markdown,
    load_current_book,
    load_liquidity_proxy,
    read_csv_safe,
    today_str,
    write_markdown_report,
)


OUT_NAV = ROOT / "paper_nav_curve.csv"
OUT_LIVE_TEMPLATE = ROOT / "live_nav_manual_template.csv"
OUT_LIVE_NAV = ROOT / "live_nav_curve.csv"
OUT_TICKER = ROOT / "return_attribution_by_ticker.csv"
OUT_SIGNAL = ROOT / "return_attribution_by_signal.csv"
OUT_SLIPPAGE = ROOT / "slippage_model_report.csv"
OUT_MD = ROOT / "live_nav_attribution_slippage_report.md"


def build_nav_curve() -> pd.DataFrame:
    frames = []
    for fname in ["portfolio_nav.csv", "paper_sim_nav.csv"]:
        df = read_csv_safe(ROOT / fname)
        if not df.empty and {"date", "nav"}.issubset(df.columns):
            tmp = df[["date", "nav"]].copy()
            tmp["source_file"] = fname
            frames.append(tmp)
    if not frames:
        nav = pd.DataFrame([{"date": today_str(), "nav": 100.0, "source_file": "seed"}])
    else:
        nav = pd.concat(frames, ignore_index=True)
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna(subset=["date", "nav"]).drop_duplicates("date", keep="last")
    nav = nav.sort_values("date").reset_index(drop=True)
    nav["daily_return"] = nav["nav"].pct_change(fill_method=None).fillna(0.0)
    nav["hwm"] = nav["nav"].cummax()
    nav["drawdown_pct"] = nav["nav"] / nav["hwm"] - 1.0
    nav["cumulative_return_pct"] = nav["nav"] / float(nav["nav"].iloc[0]) - 1.0
    return nav


def build_live_nav_template() -> pd.DataFrame:
    return pd.DataFrame([{
        "date": today_str(),
        "account_equity": "",
        "cash": "",
        "gross_exposure": "",
        "net_exposure": "",
        "deposit_withdrawal": "",
        "notes": "Manual only. No broker connection. Fill this file as live_nav_manual.csv if real account NAV should be tracked.",
    }])


def build_live_nav_curve() -> pd.DataFrame:
    manual = read_csv_safe(ROOT / "live_nav_manual.csv")
    cols = [
        "date", "account_equity", "nav", "daily_return", "hwm",
        "drawdown_pct", "cumulative_return_pct", "cash", "gross_exposure",
        "net_exposure", "deposit_withdrawal", "notes", "source_file",
    ]
    if manual.empty or "date" not in manual.columns:
        return pd.DataFrame(columns=cols)

    live = manual.copy()
    live["date"] = pd.to_datetime(live["date"], errors="coerce")
    live["account_equity"] = pd.to_numeric(live.get("account_equity", np.nan), errors="coerce")
    if "nav" in live.columns:
        live["nav"] = pd.to_numeric(live["nav"], errors="coerce")
    else:
        base = live["account_equity"].dropna().iloc[0] if not live["account_equity"].dropna().empty else np.nan
        live["nav"] = live["account_equity"] / base * 100.0 if np.isfinite(base) and base > 0 else np.nan
    live = live.dropna(subset=["date", "nav"]).drop_duplicates("date", keep="last").sort_values("date")
    if live.empty:
        return pd.DataFrame(columns=cols)
    live["daily_return"] = live["nav"].pct_change(fill_method=None).fillna(0.0)
    live["hwm"] = live["nav"].cummax()
    live["drawdown_pct"] = live["nav"] / live["hwm"] - 1.0
    live["cumulative_return_pct"] = live["nav"] / float(live["nav"].iloc[0]) - 1.0
    for col in ["cash", "gross_exposure", "net_exposure", "deposit_withdrawal", "notes"]:
        if col not in live.columns:
            live[col] = ""
    live["source_file"] = "live_nav_manual.csv"
    return live[[c for c in cols if c in live.columns]].reset_index(drop=True)


def build_ticker_attribution() -> pd.DataFrame:
    positions = read_csv_safe(ROOT / "paper_sim_positions.csv")
    trades = read_csv_safe(ROOT / "paper_sim_trades.csv")
    journal = read_csv_safe(ROOT / "trade_journal.csv")
    rows = []

    if not positions.empty and "ticker" in positions.columns:
        for _, row in positions.iterrows():
            ticker = clean_ticker(row.get("ticker", ""))
            mv = pd.to_numeric(pd.Series([row.get("market_value", np.nan)]), errors="coerce").iloc[0]
            pnl = pd.to_numeric(pd.Series([row.get("unrealised_pnl", np.nan)]), errors="coerce").iloc[0]
            pnl_pct = pd.to_numeric(pd.Series([row.get("unrealised_pct", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "ticker": ticker,
                "source": "paper_sim_positions.csv",
                "status": "OPEN",
                "market_value": mv,
                "realized_pnl": 0.0,
                "unrealized_pnl": pnl,
                "total_pnl": pnl,
                "pnl_pct": pnl_pct / 100.0 if np.isfinite(pnl_pct) and abs(pnl_pct) > 1.5 else pnl_pct,
            })

    trade_sources = [(trades, "paper_sim_trades.csv"), (journal, "trade_journal.csv")]
    for df, source in trade_sources:
        if df.empty or "ticker" not in df.columns:
            continue
        for _, row in df.iterrows():
            ticker = clean_ticker(row.get("ticker", ""))
            pnl = pd.to_numeric(pd.Series([row.get("pnl", np.nan)]), errors="coerce").iloc[0]
            pnl_pct = pd.to_numeric(pd.Series([row.get("pnl_pct", np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(pnl_pct) and abs(pnl_pct) > 1.5:
                pnl_pct = pnl_pct / 100.0
            rows.append({
                "ticker": ticker,
                "source": source,
                "status": str(row.get("status", "CLOSED")),
                "market_value": 0.0,
                "realized_pnl": pnl,
                "unrealized_pnl": 0.0,
                "total_pnl": pnl,
                "pnl_pct": pnl_pct,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = df.groupby("ticker", as_index=False).agg({
        "market_value": "sum",
        "realized_pnl": "sum",
        "unrealized_pnl": "sum",
        "total_pnl": "sum",
        "pnl_pct": "mean",
        "source": lambda x: ";".join(sorted(set(map(str, x)))),
    })
    total_pnl = float(out["total_pnl"].abs().sum())
    out["pnl_contribution_pct"] = out["total_pnl"] / total_pnl if total_pnl > 0 else 0.0
    return out.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def build_signal_attribution() -> pd.DataFrame:
    pnl = read_csv_safe(ROOT / "pnl_attribution.csv")
    if not pnl.empty and {"signal", "attributed_pnl_pct"}.issubset(pnl.columns):
        out = pnl.groupby("signal", as_index=False).agg({
            "attributed_pnl_pct": "sum",
            "ticker": "nunique",
        })
        out = out.rename(columns={"ticker": "ticker_count"})
        total = float(out["attributed_pnl_pct"].abs().sum())
        out["signal_contribution_share"] = out["attributed_pnl_pct"] / total if total > 0 else 0.0
        out["source_file"] = "pnl_attribution.csv"
        return out.sort_values("attributed_pnl_pct", ascending=False).reset_index(drop=True)

    book = load_current_book(prefer_filtered=True)
    alpha = read_csv_safe(ROOT / "alpha_scores.csv")
    if book.empty or alpha.empty or "ticker" not in alpha.columns:
        return pd.DataFrame()
    sig_cols = [c for c in alpha.columns if c.startswith("sig_")]
    if not sig_cols:
        return pd.DataFrame()
    alpha["ticker"] = alpha["ticker"].apply(clean_ticker)
    merged = book[["ticker", "weight"]].merge(alpha[["ticker"] + sig_cols], on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    w = merged["weight"].astype(float)
    w = w / max(float(w.sum()), 1e-12)
    for col in sig_cols:
        score = float((pd.to_numeric(merged[col], errors="coerce").fillna(50.0) * w).sum())
        rows.append({
            "signal": col.replace("sig_", ""),
            "weighted_signal_score": score,
            "score_vs_neutral": score - 50.0,
            "ticker_count": len(merged),
            "source_file": "alpha_scores.csv; current book",
        })
    return pd.DataFrame(rows).sort_values("score_vs_neutral", ascending=False).reset_index(drop=True)


def build_slippage_model() -> pd.DataFrame:
    book = load_current_book(prefer_filtered=True)
    if book.empty:
        return pd.DataFrame()
    tickers = book["ticker"].apply(clean_ticker).tolist()
    liq = load_liquidity_proxy(tickers)
    rows = []
    for _, row in book.iterrows():
        ticker = clean_ticker(row["ticker"])
        weight = float(row.get("weight", 0.0))
        adv = np.nan
        label = "MISSING"
        if not liq.empty and "ticker" in liq.columns:
            match = liq[liq["ticker"] == ticker]
            if not match.empty:
                adv = pd.to_numeric(pd.Series([match.iloc[0].get("avg_20d_dollar_volume", np.nan)]), errors="coerce").iloc[0]
                label = str(match.iloc[0].get("liquidity_label", "UNKNOWN"))
        notional = weight * MODEL_ACCOUNT_VALUE
        participation = notional / adv if np.isfinite(adv) and adv > 0 else np.nan
        if not np.isfinite(participation):
            slippage_bps = 35.0
            status = "MISSING_DATA_REVIEW"
        else:
            slippage_bps = 3.0 + min(150.0, 250.0 * (participation ** 0.65))
            if participation > 0.03:
                status = "SIZE_DOWN"
            elif participation > 0.01:
                status = "REVIEW"
            else:
                status = "CLEAR"
        rows.append({
            "ticker": ticker,
            "weight": weight,
            "notional_model_account": notional,
            "adv_dollar": adv,
            "participation_rate": participation,
            "estimated_one_way_slippage_bps": slippage_bps,
            "estimated_round_trip_slippage_bps": slippage_bps * 2.0,
            "liquidity_label": label,
            "slippage_status": status,
            "source_file": "intraday_liquidity_proxy.csv; current book",
        })
    return pd.DataFrame(rows).sort_values("estimated_round_trip_slippage_bps", ascending=False).reset_index(drop=True)


def write_report(nav: pd.DataFrame, live_nav: pd.DataFrame, tickers: pd.DataFrame, signals: pd.DataFrame, slip: pd.DataFrame) -> None:
    sections = [
        "## Summary",
        "",
        f"- NAV rows: {len(nav)}",
        f"- Manual live NAV rows: {len(live_nav)}",
        f"- Ticker attribution rows: {len(tickers)}",
        f"- Signal attribution rows: {len(signals)}",
        f"- Slippage rows: {len(slip)}",
        "",
        "## Logic",
        "",
        "- Paper NAV is local only; it is not live brokerage equity.",
        "- If live_nav_manual.csv is filled manually, live_nav_curve.csv tracks real account NAV without connecting to a broker.",
        "- Slippage is a conservative liquidity proxy, not an executable quote.",
        "- Attribution uses realized paper trades plus open paper positions when available.",
        "",
        "## NAV tail",
        "",
        df_to_markdown(nav.tail(10)) if not nav.empty else "No NAV rows.",
        "",
        "## Manual live NAV tail",
        "",
        df_to_markdown(live_nav.tail(10)) if not live_nav.empty else "No manual live NAV rows. Fill live_nav_manual.csv from live_nav_manual_template.csv if needed.",
        "",
        "## Ticker attribution",
        "",
        df_to_markdown(tickers, max_rows=20) if not tickers.empty else "No ticker attribution rows.",
        "",
        "## Signal attribution",
        "",
        df_to_markdown(signals, max_rows=20) if not signals.empty else "No signal attribution rows.",
        "",
        "## Slippage model",
        "",
        df_to_markdown(slip, max_rows=20) if not slip.empty else "No slippage rows.",
    ]
    write_markdown_report(OUT_MD, "Canyon v9 Step 115 - Paper NAV and Attribution", sections)


def main() -> None:
    nav = build_nav_curve()
    template = build_live_nav_template()
    live_nav = build_live_nav_curve()
    tickers = build_ticker_attribution()
    signals = build_signal_attribution()
    slip = build_slippage_model()
    nav.to_csv(OUT_NAV, index=False)
    if not (ROOT / "live_nav_manual.csv").exists():
        template.to_csv(OUT_LIVE_TEMPLATE, index=False)
    live_nav.to_csv(OUT_LIVE_NAV, index=False)
    tickers.to_csv(OUT_TICKER, index=False)
    signals.to_csv(OUT_SIGNAL, index=False)
    slip.to_csv(OUT_SLIPPAGE, index=False)
    write_report(nav, live_nav, tickers, signals, slip)
    print(f"[step115] wrote {OUT_NAV.name}: {len(nav)} rows")
    print(f"[step115] wrote {OUT_LIVE_TEMPLATE.name}: {len(template)} template row")
    print(f"[step115] wrote {OUT_LIVE_NAV.name}: {len(live_nav)} rows")
    print(f"[step115] wrote {OUT_TICKER.name}: {len(tickers)} rows")
    print(f"[step115] wrote {OUT_SIGNAL.name}: {len(signals)} rows")
    print(f"[step115] wrote {OUT_SLIPPAGE.name}: {len(slip)} rows")
    print(f"[step115] wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
