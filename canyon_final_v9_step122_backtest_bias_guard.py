#!/usr/bin/env python3
"""
Canyon v9 Step 122 - Backtest Bias Guard.

Research-only. No broker connection. No live orders.

This step makes the backtest weaknesses explicit: look-ahead risk, survivorship
risk, signal sample depth, regime fragility, transaction cost assumptions, and
failure modes. It does not replace a full institutional event-time backtester;
it creates guardrails and blocker flags around the current prototype backtests.

Outputs:
  backtest_bias_guard.csv
  backtest_walk_forward_proxy.csv
  backtest_signal_failure_modes.csv
  backtest_execution_reality_check.csv
  backtest_bias_state.json
  backtest_bias_report.md
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canyon_final_v9_risk_framework_lib import (
    ROOT,
    df_to_markdown,
    read_csv_safe,
    read_json_safe,
    today_str,
    write_json,
    write_markdown_report,
)


OUT_GUARD = ROOT / "backtest_bias_guard.csv"
OUT_WALK = ROOT / "backtest_walk_forward_proxy.csv"
OUT_FAILURE = ROOT / "backtest_signal_failure_modes.csv"
OUT_EXEC = ROOT / "backtest_execution_reality_check.csv"
OUT_STATE = ROOT / "backtest_bias_state.json"
OUT_REPORT = ROOT / "backtest_bias_report.md"


SIGNAL_FILES = [
    "regime_ml_scores.csv",
    "quality_scores.csv",
    "earnings_revision_scores.csv",
    "earnings_surprise_scores.csv",
    "finbert_sentiment_scores.csv",
    "short_squeeze_signals.csv",
    "insider_signal_scores.csv",
    "options_signals.csv",
    "ml_ensemble_scores.csv",
]


def exists_nonempty(name: str) -> bool:
    p = ROOT / name
    return p.exists() and p.stat().st_size > 10


def status_from_score(score: float) -> str:
    if score >= 80:
        return "CLEAR"
    if score >= 60:
        return "REVIEW"
    if score >= 40:
        return "WEAK"
    return "BLOCKER"


def parse_date_columns(df: pd.DataFrame) -> list[str]:
    hints = ["date", "time", "timestamp", "asof", "as_of", "published", "filing", "earnings", "rebalance", "period_end"]
    cols = []
    for c in df.columns:
        low = str(c).lower()
        if any(h in low for h in hints):
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
            if parsed.notna().sum() > 0:
                cols.append(str(c))
    return cols


def add_guard(rows: list[dict[str, Any]], control: str, score: float, evidence: str, required: str, source_file: str) -> None:
    rows.append({
        "control": control,
        "score": round(float(score), 1),
        "status": status_from_score(score),
        "evidence": evidence,
        "required_next_action": required,
        "source_file": source_file,
        "research_only": True,
    })


def build_guard() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    signal_ic = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    ledger = read_csv_safe(ROOT / "point_in_time_evidence_ledger.csv")

    signal_date_count = 0
    signal_file_count = 0
    missing_date_files = []
    for fname in SIGNAL_FILES:
        df = read_csv_safe(ROOT / fname)
        if df.empty:
            continue
        signal_file_count += 1
        date_cols = parse_date_columns(df)
        if date_cols:
            signal_date_count += 1
        else:
            missing_date_files.append(fname)
    score = 20.0 if signal_file_count == 0 else 20.0 + 55.0 * signal_date_count / max(signal_file_count, 1)
    if not ledger.empty:
        score += 10.0
    add_guard(
        rows,
        "Look-ahead bias guard",
        min(score, 75.0),
        f"{signal_date_count}/{signal_file_count} signal files have parseable date/as-of columns. Missing: {', '.join(missing_date_files[:8]) or 'none'}.",
        "Every feature must include as_of_time and every backtest join must enforce feature_time < rebalance_time.",
        "signal files / point_in_time_evidence_ledger.csv",
    )

    event_gate_state = read_json_safe(ROOT / "event_backtest_gate_state.json", {})
    event_admit = read_csv_safe(ROOT / "event_backtest_admissibility.csv")
    event_repair = read_csv_safe(ROOT / "event_time_repair_queue.csv")
    if not event_gate_state:
        add_guard(
            rows,
            "Event look-ahead admissibility",
            25,
            "No Step164 event admissibility gate found.",
            "Run Step163 and Step164 so news/event rows are excluded unless source_publish_time and first_seen_time are available.",
            "event_backtest_gate_state.json / event_backtest_admissibility.csv",
        )
    else:
        local_rows = int(event_gate_state.get("local_event_backtest_rows", 0) or 0)
        excluded = int(event_gate_state.get("excluded_from_historical_backtest", 0) or 0)
        repair_rows = int(event_gate_state.get("repair_queue_rows", len(event_repair)) or 0)
        institutional = int(event_gate_state.get("institutional_ready_rows", 0) or 0)
        gate_status = str(event_gate_state.get("overall_status", "NO_DATA"))
        event_score = 70.0 if local_rows > 0 else 35.0
        if excluded:
            event_score -= 18.0
        if repair_rows:
            event_score -= min(18.0, repair_rows * 3.0)
        if institutional == 0:
            event_score = min(event_score, 68.0)
        add_guard(
            rows,
            "Event look-ahead admissibility",
            max(25.0, event_score),
            f"Step164 status {gate_status}; local-audit rows {local_rows}; excluded rows {excluded}; repair rows {repair_rows}; institutional-ready rows {institutional}.",
            "Use pit_safe_event_signal_panel.csv for local event audits only; exclude weak-timing events and require vendor event tape before institutional claims.",
            "event_backtest_admissibility.csv / pit_safe_event_signal_panel.csv / event_time_repair_queue.csv",
        )

    event_audit_state = read_json_safe(ROOT / "event_signal_audit_state.json", {})
    event_audit = read_csv_safe(ROOT / "event_signal_local_audit_returns.csv")
    if not event_audit_state:
        add_guard(
            rows,
            "Event signal audit maturity",
            25,
            "No Step165 event signal local audit found.",
            "Run Step165 after Step164 so event-derived signals have source-publish and model-first-seen return checks.",
            "event_signal_audit_state.json / event_signal_local_audit_returns.csv",
        )
    else:
        audit_rows = int(event_audit_state.get("event_rows", len(event_audit)) or 0)
        source_1d_cov = float(event_audit_state.get("source_1d_coverage_pct", 0) or 0)
        model_1d_cov = float(event_audit_state.get("model_seen_1d_coverage_pct", 0) or 0)
        audit_status = str(event_audit_state.get("overall_status", "NO_DATA"))
        audit_score = 35.0
        if audit_rows > 100:
            audit_score += 15.0
        if source_1d_cov >= 30:
            audit_score += 10.0
        if model_1d_cov >= 30:
            audit_score += 15.0
        if model_1d_cov == 0:
            audit_score = min(audit_score, 55.0)
        if event_audit_state.get("can_support_institutional_backtest") is False:
            audit_score = min(audit_score, 62.0)
        add_guard(
            rows,
            "Event signal audit maturity",
            max(25.0, min(audit_score, 80.0)),
            f"Step165 status {audit_status}; rows {audit_rows}; source 1d coverage {source_1d_cov:.1f}%; model-seen 1d coverage {model_1d_cov:.1f}%.",
            "Do not treat source-publish reaction as tradable live validation; wait for model-seen windows and vendor-grade event tape before stronger historical claims.",
            "event_signal_local_audit_returns.csv / event_signal_local_audit_summary.csv / event_signal_failure_modes.csv",
        )

    event_rel_state = read_json_safe(ROOT / "event_signal_reliability_state.json", {})
    event_rel_bucket = read_csv_safe(ROOT / "event_signal_reliability_by_bucket.csv")
    if not event_rel_state:
        add_guard(
            rows,
            "Event reliability calibration",
            25,
            "No Step166 event reliability calibration found.",
            "Run Step166 after Step165 so event signal families are calibrated before they affect research ranking.",
            "event_signal_reliability_state.json / event_signal_reliability_by_bucket.csv",
        )
    else:
        rel_status = str(event_rel_state.get("overall_status", "NO_DATA"))
        bucket_rows = int(event_rel_state.get("bucket_rows", len(event_rel_bucket)) or 0)
        reliable = int(event_rel_state.get("reliable_bucket_count", 0) or 0)
        low_sample = int(event_rel_state.get("low_sample_bucket_count", 0) or 0)
        model_1d_cov = float(event_rel_state.get("model_seen_1d_coverage_pct", 0) or 0)
        rel_score = 45.0
        if bucket_rows > 20:
            rel_score += 10.0
        if reliable > 0:
            rel_score += min(10.0, reliable * 2.0)
        if low_sample > 0:
            rel_score -= min(8.0, low_sample * 0.5)
        if model_1d_cov == 0:
            rel_score = min(rel_score, 55.0)
        if event_rel_state.get("can_support_institutional_backtest") is False:
            rel_score = min(rel_score, 62.0)
        add_guard(
            rows,
            "Event reliability calibration",
            max(25.0, min(rel_score, 78.0)),
            f"Step166 status {rel_status}; buckets {bucket_rows}; reliable buckets {reliable}; low-sample buckets {low_sample}; model-seen 1d coverage {model_1d_cov:.1f}%.",
            "Use calibrated event scores only as research context; do not let local source-reaction calibration override PIT, risk, options, or execution gates.",
            "event_signal_reliability_by_bucket.csv / event_signal_reliability_adjusted_panel.csv / event_signal_reliability_watchlist.csv",
        )

    membership_ready = exists_nonempty("universe_membership_history.csv")
    snapshot_ready = exists_nonempty("universe_membership_snapshot.csv")
    add_guard(
        rows,
        "Survivorship bias guard",
        80 if membership_ready else (35 if snapshot_ready else 20),
        "Historical universe membership exists." if membership_ready else ("Only a current universe snapshot exists." if snapshot_ready else "No universe membership file found."),
        "Add historical constituents, additions/removals, inactive symbols, and delisted tickers by date.",
        "universe_membership_history.csv / universe_membership_snapshot.csv",
    )

    delisted_ready = exists_nonempty("delisted_tickers.csv")
    add_guard(
        rows,
        "Delisted ticker inclusion",
        75 if delisted_ready else 15,
        "Delisted ticker table found." if delisted_ready else "No delisted ticker table found.",
        "Backtests must include tickers that disappeared, were acquired, or went to zero.",
        "delisted_tickers.csv",
    )

    if monthly.empty:
        add_guard(rows, "Multi-regime coverage", 20, "No monthly backtest found.", "Run Step62 and store monthly returns.", "backtest_monthly_perf.csv")
    else:
        date_col = "rebalance_date" if "rebalance_date" in monthly.columns else monthly.columns[0]
        dates = pd.to_datetime(monthly[date_col], errors="coerce").dropna()
        years = sorted(set(dates.dt.year.astype(int))) if not dates.empty else []
        score = min(75, 25 + 10 * len(years))
        add_guard(
            rows,
            "Multi-regime coverage",
            score,
            f"{len(monthly)} monthly periods across years {years}.",
            "Validate on distinct regimes: 2020 crash, 2022 hikes, low-vol bull, bear, sideways.",
            "backtest_monthly_perf.csv",
        )

    if signal_ic.empty:
        add_guard(rows, "Signal IC sample depth", 20, "No signal IC table found.", "Track signal IC by date, horizon, sector, and regime.", "backtest_signal_ic.csv")
    else:
        n = pd.to_numeric(signal_ic.get("n_obs", pd.Series(dtype=float)), errors="coerce").fillna(0)
        min_obs = int(n.min()) if not n.empty else 0
        weak = int((n < 30).sum()) if not n.empty else len(signal_ic)
        score = 75 if min_obs >= 60 else (55 if min_obs >= 30 else 35)
        add_guard(
            rows,
            "Signal IC sample depth",
            score,
            f"{len(signal_ic)} signals; minimum observations {min_obs}; signals below 30 observations {weak}.",
            "Do not size signals aggressively until IC is stable by regime and sample size.",
            "backtest_signal_ic.csv",
        )

    if monthly.empty or "tc_cost_bps" not in monthly.columns:
        add_guard(rows, "Transaction cost realism", 25, "No explicit backtest cost column found.", "Add TCA from Step120 to backtest returns.", "backtest_monthly_perf.csv")
    else:
        tc = pd.to_numeric(monthly["tc_cost_bps"], errors="coerce").dropna()
        tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
        avg_live_tca = pd.to_numeric(tca.get("total_tca_cost_bps", pd.Series(dtype=float)), errors="coerce").mean() if not tca.empty else np.nan
        med_backtest = float(tc.median()) if not tc.empty else np.nan
        gap = avg_live_tca - med_backtest if np.isfinite(avg_live_tca) and np.isfinite(med_backtest) else np.nan
        score = 70 if np.isfinite(gap) and gap <= 15 else 45
        add_guard(
            rows,
            "Transaction cost realism",
            score,
            f"Median backtest cost {med_backtest:.1f} bps; current Step120 average TCA {avg_live_tca:.1f} bps; gap {gap:+.1f} bps.",
            "Replace static monthly cost assumptions with ticker-level spread, volume, impact, and failed-fill costs.",
            "backtest_monthly_perf.csv / institutional_tca_cost_estimates.csv",
        )

    return pd.DataFrame(rows)


def build_walk_forward_proxy() -> pd.DataFrame:
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    if monthly.empty or "strategy_ret" not in monthly.columns:
        return pd.DataFrame(columns=["window", "start", "end", "months", "annualized_return", "annualized_vol", "sharpe_proxy", "max_drawdown", "status"])
    date_col = "rebalance_date" if "rebalance_date" in monthly.columns else monthly.columns[0]
    work = monthly.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    work["strategy_ret"] = pd.to_numeric(work["strategy_ret"], errors="coerce").fillna(0.0)
    windows = []
    n = len(work)
    splits = [
        ("first_half_train_proxy", 0, max(n // 2, 1)),
        ("second_half_oos_proxy", max(n // 2, 1), n),
        ("last_12_months", max(n - 12, 0), n),
        ("last_24_months", max(n - 24, 0), n),
    ]
    for name, a, b in splits:
        sub = work.iloc[a:b].copy()
        if sub.empty:
            continue
        r = sub["strategy_ret"].astype(float)
        curve = (1.0 + r).cumprod()
        dd = curve / curve.cummax() - 1.0
        ann_ret = float((1.0 + r.mean()) ** 12 - 1.0)
        ann_vol = float(r.std(ddof=1) * math.sqrt(12)) if len(r) > 1 else np.nan
        sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else np.nan
        status = "CLEAR" if np.isfinite(sharpe) and sharpe > 0.8 else ("REVIEW" if np.isfinite(sharpe) and sharpe > 0.3 else "WEAK")
        windows.append({
            "window": name,
            "start": str(sub[date_col].iloc[0].date()),
            "end": str(sub[date_col].iloc[-1].date()),
            "months": len(sub),
            "annualized_return": ann_ret,
            "annualized_vol": ann_vol,
            "sharpe_proxy": sharpe,
            "max_drawdown": float(dd.min()),
            "status": status,
            "source_file": "backtest_monthly_perf.csv",
        })
    return pd.DataFrame(windows)


def build_failure_modes() -> pd.DataFrame:
    rows = []
    signal_ic = read_csv_safe(ROOT / "backtest_signal_ic.csv")
    if not signal_ic.empty and "signal" in signal_ic.columns:
        for _, row in signal_ic.iterrows():
            mean_ic = pd.to_numeric(pd.Series([row.get("mean_ic")]), errors="coerce").iloc[0]
            n_obs = pd.to_numeric(pd.Series([row.get("n_obs")]), errors="coerce").iloc[0]
            pos = row.get("ic_positive_pct", "")
            status = "CLEAR"
            failure = []
            if not np.isfinite(n_obs) or n_obs < 30:
                status = "WEAK"
                failure.append("small sample")
            if not np.isfinite(mean_ic) or mean_ic <= 0:
                status = "BLOCKER"
                failure.append("non-positive IC")
            elif mean_ic < 0.03:
                status = "REVIEW"
                failure.append("low IC")
            rows.append({
                "signal": row.get("signal", ""),
                "n_obs": n_obs,
                "mean_ic": mean_ic,
                "ic_positive_pct": pos,
                "failure_mode": "; ".join(failure) if failure else "none in current summary",
                "status": status,
                "required_next_action": "Track signal by regime, sector, market cap, horizon, and false-positive bucket.",
                "source_file": "backtest_signal_ic.csv",
            })
    strategy = read_csv_safe(ROOT / "strategy_backtest_results.csv")
    if not strategy.empty and {"strategy", "ret"}.issubset(strategy.columns):
        for strat, grp in strategy.groupby("strategy"):
            ret = pd.to_numeric(grp["ret"], errors="coerce").dropna()
            if ret.empty:
                continue
            hit = float((ret > 0).mean())
            worst = float(ret.min())
            avg = float(ret.mean())
            status = "CLEAR" if hit >= 0.55 and worst > -0.08 else ("REVIEW" if hit >= 0.45 else "WEAK")
            rows.append({
                "signal": f"strategy:{strat}",
                "n_obs": len(ret),
                "mean_ic": np.nan,
                "ic_positive_pct": f"{hit:.1%}",
                "failure_mode": f"worst monthly return {worst:.2%}; average {avg:.2%}",
                "status": status,
                "required_next_action": "Tag the market regime and catalyst context of every losing month.",
                "source_file": "strategy_backtest_results.csv",
            })
    return pd.DataFrame(rows)


def build_execution_check() -> pd.DataFrame:
    monthly = read_csv_safe(ROOT / "backtest_monthly_perf.csv")
    tca = read_csv_safe(ROOT / "institutional_tca_cost_estimates.csv")
    rows = []
    if monthly.empty:
        return pd.DataFrame(columns=["check", "status", "evidence", "source_file"])
    turnover = pd.to_numeric(monthly.get("turnover_pct", pd.Series(dtype=float)), errors="coerce")
    cost = pd.to_numeric(monthly.get("tc_cost_bps", pd.Series(dtype=float)), errors="coerce")
    avg_tca = pd.to_numeric(tca.get("total_tca_cost_bps", pd.Series(dtype=float)), errors="coerce").mean() if not tca.empty else np.nan
    rows.append({
        "check": "Turnover drag",
        "status": "REVIEW" if turnover.median() > 50 else "CLEAR",
        "evidence": f"Median monthly turnover {turnover.median():.1f}%; 90th percentile {turnover.quantile(0.9):.1f}%.",
        "stress_cost_bps": float(cost.median() * 2) if not cost.dropna().empty else np.nan,
        "source_file": "backtest_monthly_perf.csv",
    })
    rows.append({
        "check": "Backtest cost vs current TCA",
        "status": "WEAK" if np.isfinite(avg_tca) and not cost.dropna().empty and avg_tca > cost.median() * 2 else "REVIEW",
        "evidence": f"Median backtest cost {cost.median():.1f} bps; current average TCA {avg_tca:.1f} bps.",
        "stress_cost_bps": avg_tca,
        "source_file": "backtest_monthly_perf.csv / institutional_tca_cost_estimates.csv",
    })
    rows.append({
        "check": "Failed fill assumption",
        "status": "WEAK",
        "evidence": "Current backtests do not explicitly model missed fills, auction imbalance, partial execution, or delayed entry.",
        "stress_cost_bps": 25.0,
        "source_file": "backtest_monthly_perf.csv",
    })
    return pd.DataFrame(rows)


def write_outputs(guard: pd.DataFrame, walk: pd.DataFrame, failure: pd.DataFrame, execution: pd.DataFrame) -> None:
    guard.to_csv(OUT_GUARD, index=False)
    walk.to_csv(OUT_WALK, index=False)
    failure.to_csv(OUT_FAILURE, index=False)
    execution.to_csv(OUT_EXEC, index=False)
    avg_score = float(pd.to_numeric(guard.get("score", pd.Series(dtype=float)), errors="coerce").mean()) if not guard.empty else 0.0
    blocker_count = int(guard.get("status", pd.Series(dtype=str)).astype(str).str.upper().isin(["BLOCKER", "WEAK"]).sum()) if not guard.empty else 0
    state = {
        "date": today_str(),
        "backtest_bias_guard_score": round(avg_score, 1),
        "overall_status": status_from_score(avg_score),
        "blocker_or_weak_controls": blocker_count,
        "truth": "This is a guardrail around prototype backtests, not a full institutional event-time backtest engine.",
        "research_only": True,
        "no_broker_connection": True,
    }
    write_json(OUT_STATE, state)
    sections = [
        "## Product Truth",
        "",
        state["truth"],
        "",
        f"- Bias guard score: {state['backtest_bias_guard_score']}",
        f"- Overall status: {state['overall_status']}",
        f"- Blocker or weak controls: {state['blocker_or_weak_controls']}",
        "",
        "## Bias Guard Controls",
        "",
        df_to_markdown(guard, max_rows=80),
        "",
        "## Walk-Forward Proxy",
        "",
        df_to_markdown(walk, max_rows=20),
        "",
        "## Signal Failure Modes",
        "",
        df_to_markdown(failure, max_rows=80),
        "",
        "## Execution Reality Check",
        "",
        df_to_markdown(execution, max_rows=40),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 122 - Backtest Bias Guard", sections)


def main() -> None:
    guard = build_guard()
    walk = build_walk_forward_proxy()
    failure = build_failure_modes()
    execution = build_execution_check()
    write_outputs(guard, walk, failure, execution)
    state = read_json_safe(OUT_STATE, {})
    print(f"[step122] wrote {OUT_GUARD.name}: {len(guard)} controls")
    print(f"[step122] bias_guard_score={state.get('backtest_bias_guard_score')} status={state.get('overall_status')}")
    print(f"[step122] wrote {OUT_WALK.name}, {OUT_FAILURE.name}, {OUT_EXEC.name}, {OUT_REPORT.name}")


if __name__ == "__main__":
    main()
