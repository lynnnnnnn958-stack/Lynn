#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 58 - v8 L9 Execution Gate Bridge

Uses the v8 OperationManual philosophy to fill execution-gate gaps without
activating any live execution code.

This is a paper/research gate only:
- No broker connection.
- No live orders.
- Sector ETFs without old pre-trade rows become explicit risk-reduction /
  research-only rows instead of L9 NO_DATA.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import importlib.util

import pandas as pd

ROOT = Path.cwd()
SOURCE_CANDIDATES = [
    ROOT / "canyon_final_v8_latest_source.py",
    ROOT / "canyon_final_v8_legacy_source.py",
]
SOURCE = next((path for path in SOURCE_CANDIDATES if path.exists()), SOURCE_CANDIDATES[-1])
OUT_CSV = ROOT / "v8_l9_execution_gate.csv"
OUT_REPORT = ROOT / "v8_l9_execution_gate_report.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def load_v8_module():
    if not SOURCE.exists():
        return None
    spec = importlib.util.spec_from_file_location("canyon_v8_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        return None
    spec.loader.exec_module(module)
    return module


def risk_light(master: pd.DataFrame) -> str:
    if not master.empty and "L8_state" in master.columns and not master["L8_state"].empty:
        return str(master["L8_state"].mode().iloc[0])
    return "NO_DATA"


def existing_pretrade_tickers(pretrade: pd.DataFrame) -> set[str]:
    if pretrade.empty or "ticker" not in pretrade.columns:
        return set()
    return set(pretrade["ticker"].astype(str).str.upper().str.strip())


def candidate_rows(master: pd.DataFrame, sectors: pd.DataFrame, pretrade: pd.DataFrame) -> pd.DataFrame:
    existing = existing_pretrade_tickers(pretrade)
    rows = []

    if not master.empty and "ticker" in master.columns:
        source = master.copy()
    else:
        source = sectors.copy()
        if "rotation_label" in source.columns:
            source["master_action"] = "RISK_REDUCTION_FIRST"

    if source.empty or "ticker" not in source.columns:
        return pd.DataFrame()

    for _, row in source.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker or ticker in existing:
            continue

        action = str(row.get("master_action", "RISK_REDUCTION_FIRST")).upper()
        l3 = str(row.get("L3_state", row.get("rotation_label", "")))
        l6 = str(row.get("L6_state", ""))

        if action not in {"RISK_REDUCTION_FIRST", "RESEARCH_ONLY"}:
            continue

        rows.append({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "sleeve": "SECTOR_ROTATION" if ticker.startswith("XL") or ticker == "IYR" else "RESEARCH",
            "decision": "RISK_REDUCTION_FIRST" if action == "RISK_REDUCTION_FIRST" else "RESEARCH_ONLY",
            "risk_bucket": "SECTOR_ETF_CONTEXT",
            "effective_weight": "",
            "suggested_weight": "0.0",
            "suggested_action": "NO_NEW_RISK_RESEARCH_ONLY",
            "ledger_status": "NO_DIRECT_PRETRADE_ROW",
            "risk_light": risk_light(master),
            "risk_detail": "L8 risk override remains active; this row prevents L9 from being blank.",
            "manual_news_check": "N/A_FOR_SECTOR_CONTEXT",
            "earnings_date_check": "N/A_FOR_ETF",
            "liquidity_check": "REVIEW_IF_ACTIONABLE",
            "spread_check": "REVIEW_IF_ACTIONABLE",
            "duplicate_exposure_check": "REVIEW",
            "stress_check": "FAIL_OR_RISK_REDUCTION_ONLY",
            "paper_allowed": "RISK_REDUCTION_PAPER_ONLY",
            "live_allowed": "NO",
            "final_status": "RESEARCH_ONLY_NO_NEW_RISK",
            "reasons": (
                f"v8 OperationManual bridge; master_action={action}; "
                f"L3={l3}; L6={l6}; no old pre-trade row; no live order."
            ),
            "sizing_reason": "new exposure blocked; use only for risk reduction or research context",
        })

    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, workflow: list[dict]):
    md = [
        "# Canyon v9 Step 58 - v8 L9 Execution Gate Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Guardrails",
        "- No broker connection.",
        "- No live orders.",
        "- This fills L9 research/pre-trade gaps only.",
        "- Rows marked `RESEARCH_ONLY_NO_NEW_RISK` are not buy signals.",
        "",
        "## Summary",
        f"- Source: `{SOURCE.name}`",
        f"- Rows generated: {len(df)}",
        f"- Output: `{OUT_CSV.name}`",
        "",
    ]
    if not df.empty:
        cols = ["ticker", "decision", "risk_light", "paper_allowed", "live_allowed", "final_status"]
        md.append(df[cols].to_markdown(index=False))
        md.append("")

    if workflow:
        md.extend(["## v8 Operation Manual Workflow Reference", ""])
        wf = pd.DataFrame(workflow)
        cols = [c for c in ["time", "step", "action", "caveat"] if c in wf.columns]
        md.append(wf[cols].to_markdown(index=False))
        md.append("")

    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")


def main():
    print("=" * 88)
    print("CANYON v9 Step 58")
    print("v8 L9 Execution Gate Bridge")
    print("=" * 88)

    master = read_csv(ROOT / "master_10_layer_decision_matrix_v2.csv")
    sectors = read_csv(ROOT / "sector_rotation_scores.csv")
    pretrade = read_csv(ROOT / "pre_trade_checklist.csv")
    module = load_v8_module()
    workflow = []
    if module is not None and hasattr(module, "BookV2_OperationManual"):
        workflow = module.BookV2_OperationManual.daily_operation_workflow()

    out = candidate_rows(master, sectors, pretrade)
    out.to_csv(OUT_CSV, index=False)
    write_report(out, workflow)

    print(f"Rows generated: {len(out)}")
    print("Files generated:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_REPORT}")
    print("No broker connection. No live order.")


if __name__ == "__main__":
    main()
