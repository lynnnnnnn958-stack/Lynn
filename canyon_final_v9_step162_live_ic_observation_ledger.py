#!/usr/bin/env python3
"""
Canyon v9 Step 162 - Live IC Observation Ledger.

Research-only. No broker connection. No live orders.

This step starts the real signal-IC evidence loop without fabricating results:
it snapshots today's signal values, waits for future prices to mature, then
computes realized cross-sectional rank IC only when forward returns exist.

Outputs:
  score_history.csv
  live_ic_observation_ledger.csv
  live_ic_history.csv
  live_ic_realized_summary.csv
  live_ic_observation_state.json
  live_ic_observation_report.md
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    clean_ticker,
    df_to_markdown,
    read_csv_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_SCORE_HISTORY = ROOT / "score_history.csv"
OUT_OBSERVATIONS = ROOT / "live_ic_observation_ledger.csv"
OUT_LIVE_IC = ROOT / "live_ic_history.csv"
OUT_SUMMARY = ROOT / "live_ic_realized_summary.csv"
OUT_STATE = ROOT / "live_ic_observation_state.json"
OUT_REPORT = ROOT / "live_ic_observation_report.md"

HORIZONS = [1, 5, 20]
MIN_TICKERS = 15
MODEL_READ_TIME = datetime.now().replace(microsecond=0).isoformat()
LOCAL_QUALITY = "LOCAL_LIVE_OBSERVATION_NOT_VENDOR_PIT"


def load_prices() -> pd.DataFrame:
    for name in ["backtest_price_cache.csv", "sp500_price_cache.csv"]:
        df = read_csv_safe(ROOT / name)
        if df.empty:
            continue
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
        out.columns = [clean_ticker(c) for c in out.columns]
        out = out.apply(pd.to_numeric, errors="coerce").ffill()
        return out
    return pd.DataFrame()


def signal_columns(alpha: pd.DataFrame) -> list[str]:
    preferred = ["alpha_score", "sig_regime_ml", "sig_quality", "sig_revision", "sig_surprise", "sig_sentiment", "sig_squeeze", "sig_insider", "sig_options", "sig_ml_ensemble", "sig_momentum"]
    return [c for c in preferred if c in alpha.columns]


def append_score_history(alpha: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    cols = ["ticker", "alpha_score", "regime", "signal"]
    available = [c for c in cols if c in alpha.columns]
    snap = alpha[available].copy()
    if "ticker" not in snap.columns or "alpha_score" not in snap.columns:
        return read_csv_safe(OUT_SCORE_HISTORY)
    snap["ticker"] = snap["ticker"].map(clean_ticker)
    snap = snap[snap["ticker"] != ""].copy()
    snap["date"] = as_of_date
    snap["predicted_score"] = pd.to_numeric(snap["alpha_score"], errors="coerce")
    if "regime" not in snap.columns:
        snap["regime"] = "UNKNOWN"
    if "signal" not in snap.columns:
        snap["signal"] = "RESEARCH"
    snap["source_file"] = "alpha_scores.csv"
    snap["model_read_time"] = MODEL_READ_TIME
    snap = snap[["date", "ticker", "predicted_score", "regime", "signal", "source_file", "model_read_time"]]

    old = read_csv_safe(OUT_SCORE_HISTORY)
    combined = pd.concat([old, snap], ignore_index=True, sort=False) if not old.empty else snap
    combined["ticker"] = combined["ticker"].map(clean_ticker)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date.astype(str)
    combined = combined.dropna(subset=["date"])
    combined = combined.drop_duplicates(["date", "ticker"], keep="last").sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_csv(OUT_SCORE_HISTORY, index=False)
    return combined


def due_date_for(index: pd.DatetimeIndex, as_of: pd.Timestamp, horizon: int) -> str:
    if as_of in index:
        pos = index.get_loc(as_of)
        if isinstance(pos, slice):
            pos = pos.start
        target = int(pos) + horizon
        if target < len(index):
            return str(index[target].date())
    return str((as_of + pd.tseries.offsets.BDay(horizon)).date())


def observation_key(row: pd.Series) -> str:
    return f"{row['as_of_date']}|{row['ticker']}|{row['signal_name']}|{int(row['horizon_days'])}"


def build_new_observations(alpha: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if alpha.empty or prices.empty:
        return pd.DataFrame()
    as_of = pd.Timestamp(prices.index.max())
    as_of_date = str(as_of.date())
    cols = signal_columns(alpha)
    price_row = prices.loc[as_of]
    rows: list[dict[str, Any]] = []
    work = alpha.copy()
    work["ticker"] = work["ticker"].map(clean_ticker)
    work = work[work["ticker"].isin(prices.columns)]

    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        entry_price = float(price_row.get(ticker, np.nan))
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        for signal_name in cols:
            value = pd.to_numeric(pd.Series([row.get(signal_name)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            for horizon in HORIZONS:
                rows.append({
                    "observed_at": MODEL_READ_TIME,
                    "as_of_date": as_of_date,
                    "model_read_time": MODEL_READ_TIME,
                    "ticker": ticker,
                    "signal_name": signal_name,
                    "signal": signal_name,
                    "signal_value": round(float(value), 6),
                    "horizon_days": int(horizon),
                    "entry_price": round(entry_price, 6),
                    "evaluation_due_date": due_date_for(prices.index, as_of, horizon),
                    "evaluation_status": "PENDING_FORWARD_RETURN",
                    "exit_price": np.nan,
                    "forward_return": np.nan,
                    "source_file": "alpha_scores.csv / backtest_price_cache.csv",
                    "pit_quality_status": LOCAL_QUALITY,
                    "can_support_current_research": True,
                    "can_support_institutional_backtest": False,
                    "truth_note": "Observation is real local evidence; IC is calculated only after future prices exist.",
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["observation_key"] = out.apply(observation_key, axis=1)
    return out


def evaluate_observations(obs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if obs.empty or prices.empty:
        return obs
    out = obs.copy()
    out["ticker"] = out["ticker"].map(clean_ticker)
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce").dt.date.astype(str)
    price_dates = pd.to_datetime(prices.index).normalize()

    for idx, row in out.iterrows():
        if str(row.get("evaluation_status", "")).startswith("COMPLETE"):
            continue
        ticker = clean_ticker(row.get("ticker"))
        as_of = pd.to_datetime(row.get("as_of_date"), errors="coerce")
        horizon = int(pd.to_numeric(pd.Series([row.get("horizon_days")]), errors="coerce").fillna(0).iloc[0])
        entry_price = float(pd.to_numeric(pd.Series([row.get("entry_price")]), errors="coerce").iloc[0])
        if ticker not in prices.columns or pd.isna(as_of) or horizon <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            out.at[idx, "evaluation_status"] = "DATA_GAP"
            continue
        matches = np.where(price_dates >= as_of.normalize())[0]
        if len(matches) == 0:
            out.at[idx, "evaluation_status"] = "DATA_GAP"
            continue
        target_pos = int(matches[0]) + horizon
        if target_pos >= len(prices.index):
            out.at[idx, "evaluation_status"] = "PENDING_FORWARD_RETURN"
            continue
        exit_price = float(prices.iloc[target_pos].get(ticker, np.nan))
        if not np.isfinite(exit_price) or exit_price <= 0:
            out.at[idx, "evaluation_status"] = "DATA_GAP"
            continue
        out.at[idx, "exit_price"] = round(exit_price, 6)
        out.at[idx, "forward_return"] = round((exit_price / entry_price) - 1.0, 8)
        out.at[idx, "evaluation_due_date"] = str(pd.Timestamp(prices.index[target_pos]).date())
        out.at[idx, "evaluation_status"] = "COMPLETE_LOCAL_FORWARD_RETURN"
    return out


def rank_ic(group: pd.DataFrame) -> float:
    s = pd.to_numeric(group["signal_value"], errors="coerce")
    r = pd.to_numeric(group["forward_return"], errors="coerce")
    tmp = pd.DataFrame({"signal": s, "ret": r}).dropna()
    if len(tmp) < MIN_TICKERS:
        return np.nan
    return float(tmp["signal"].rank().corr(tmp["ret"].rank()))


def build_live_ic_summary(obs: pd.DataFrame) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame(columns=[
            "observed_at", "model_read_time", "score_date", "signal",
            "hold_days", "live_ic", "n_tickers", "evaluation_status",
            "source_file",
        ])
    rows: list[dict[str, Any]] = []
    keys = ["as_of_date", "signal_name", "horizon_days"]
    for key, grp in obs.groupby(keys, dropna=False):
        as_of_date, signal_name, horizon = key
        complete = grp[grp["evaluation_status"].astype(str).eq("COMPLETE_LOCAL_FORWARD_RETURN")]
        ic = rank_ic(complete) if len(complete) >= MIN_TICKERS else np.nan
        if np.isfinite(ic):
            status = "COMPLETE_LOCAL_IC"
        elif len(complete) > 0:
            status = "THIN_COMPLETE_SAMPLE"
        else:
            status = "PENDING_FORWARD_RETURN"
        rows.append({
            "observed_at": MODEL_READ_TIME,
            "model_read_time": MODEL_READ_TIME,
            "score_date": as_of_date,
            "signal": signal_name,
            "hold_days": int(horizon),
            "live_ic": round(ic, 4) if np.isfinite(ic) else np.nan,
            "ic": round(ic, 4) if np.isfinite(ic) else np.nan,
            "n_tickers": int(len(complete)),
            "pending_tickers": int(grp["evaluation_status"].astype(str).eq("PENDING_FORWARD_RETURN").sum()),
            "evaluation_status": status,
            "source_file": "live_ic_observation_ledger.csv",
            "pit_quality_status": LOCAL_QUALITY,
        })
    return pd.DataFrame(rows).sort_values(["score_date", "signal", "hold_days"]).reset_index(drop=True)


def build_realized_summary(live_ic: pd.DataFrame) -> pd.DataFrame:
    if live_ic.empty:
        return pd.DataFrame(columns=[
            "observed_at", "model_read_time", "signal", "horizon_days",
            "live_observation_windows", "mean_live_ic", "positive_ic_pct",
            "status", "required_next_action",
        ])
    rows: list[dict[str, Any]] = []
    for (signal_name, horizon), grp in live_ic.groupby(["signal", "hold_days"], dropna=False):
        vals = pd.to_numeric(grp.get("live_ic", pd.Series(dtype=float)), errors="coerce").dropna()
        windows = int(len(vals))
        if windows >= 10:
            mean_ic = float(vals.mean())
            pos_pct = float((vals > 0).mean() * 100.0)
            status = "LIVE_EVIDENCE_USABLE" if mean_ic > 0.02 and pos_pct >= 55 else "LIVE_EVIDENCE_WEAK"
            action = "Compare live IC to backtest IC and down-weight if decay persists."
        elif windows > 0:
            mean_ic = float(vals.mean())
            pos_pct = float((vals > 0).mean() * 100.0)
            status = "THIN_LIVE_SAMPLE"
            action = "Keep collecting; do not promote to sizing evidence yet."
        else:
            mean_ic = np.nan
            pos_pct = np.nan
            status = "PENDING_FORWARD_RETURNS"
            action = "Wait for future prices to mature; no live IC claim yet."
        rows.append({
            "observed_at": MODEL_READ_TIME,
            "model_read_time": MODEL_READ_TIME,
            "signal": signal_name,
            "horizon_days": int(horizon),
            "live_observation_windows": windows,
            "mean_live_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
            "positive_ic_pct": round(pos_pct, 1) if np.isfinite(pos_pct) else np.nan,
            "status": status,
            "required_next_action": action,
        })
    return pd.DataFrame(rows).sort_values(["status", "signal", "horizon_days"]).reset_index(drop=True)


def write_outputs() -> dict[str, Any]:
    alpha = read_csv_safe(ROOT / "alpha_scores.csv")
    prices = load_prices()
    if alpha.empty or "ticker" not in alpha.columns or prices.empty:
        obs = pd.DataFrame()
        live_ic = build_live_ic_summary(obs)
        summary = build_realized_summary(live_ic)
        status = "NO_SIGNAL_OR_PRICE_DATA"
    else:
        as_of_date = str(pd.Timestamp(prices.index.max()).date())
        append_score_history(alpha, as_of_date)
        new_obs = build_new_observations(alpha, prices)
        old_obs = read_csv_safe(OUT_OBSERVATIONS)
        obs = pd.concat([old_obs, new_obs], ignore_index=True, sort=False) if not old_obs.empty else new_obs
        if not obs.empty:
            if "observation_key" not in obs.columns:
                obs["observation_key"] = obs.apply(observation_key, axis=1)
            obs = obs.drop_duplicates("observation_key", keep="last").reset_index(drop=True)
            obs = evaluate_observations(obs, prices)
            obs = obs.sort_values(["as_of_date", "signal_name", "horizon_days", "ticker"]).reset_index(drop=True)
        live_ic = build_live_ic_summary(obs)
        summary = build_realized_summary(live_ic)
        status = "LIVE_IC_ACTIVE" if pd.to_numeric(live_ic.get("live_ic", pd.Series(dtype=float)), errors="coerce").notna().any() else "LIVE_IC_OBSERVATION_STARTED"

    obs.to_csv(OUT_OBSERVATIONS, index=False)
    live_ic.to_csv(OUT_LIVE_IC, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    complete = int(obs.get("evaluation_status", pd.Series(dtype=str)).astype(str).eq("COMPLETE_LOCAL_FORWARD_RETURN").sum()) if not obs.empty else 0
    pending = int(obs.get("evaluation_status", pd.Series(dtype=str)).astype(str).eq("PENDING_FORWARD_RETURN").sum()) if not obs.empty else 0
    state = {
        "date": today_str(),
        "generated_at": MODEL_READ_TIME,
        "overall_status": status,
        "observation_rows": int(len(obs)),
        "complete_forward_return_rows": complete,
        "pending_forward_return_rows": pending,
        "live_ic_windows": int(pd.to_numeric(live_ic.get("live_ic", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not live_ic.empty else 0,
        "signals_tracked": int(obs.get("signal_name", pd.Series(dtype=str)).nunique()) if not obs.empty else 0,
        "horizons": HORIZONS,
        "truth": "Live IC is not fabricated. Observations remain pending until future prices exist; local evidence is not vendor-grade point-in-time proof.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)

    sections = [
        "## Verdict",
        "",
        f"- Overall status: **{state['overall_status']}**",
        f"- Observation rows: **{state['observation_rows']}**",
        f"- Complete forward-return rows: **{state['complete_forward_return_rows']}**",
        f"- Pending forward-return rows: **{state['pending_forward_return_rows']}**",
        f"- Live IC windows: **{state['live_ic_windows']}**",
        "",
        state["truth"],
        "",
        "## Realized Summary",
        "",
        df_to_markdown(summary, max_rows=80),
        "",
        "## Latest Live IC Windows",
        "",
        df_to_markdown(live_ic.tail(80), max_rows=80),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 162 - Live IC Observation Ledger", sections)
    return state


def main() -> None:
    state = write_outputs()
    print("Canyon v9 Step162 live IC observation ledger complete.")
    print(f"Overall: {state.get('overall_status')}")
    print(f"Observations: {state.get('observation_rows')} | complete: {state.get('complete_forward_return_rows')} | pending: {state.get('pending_forward_return_rows')}")
    print(f"Outputs: {OUT_OBSERVATIONS.name}, {OUT_LIVE_IC.name}, {OUT_SUMMARY.name}")


if __name__ == "__main__":
    main()
