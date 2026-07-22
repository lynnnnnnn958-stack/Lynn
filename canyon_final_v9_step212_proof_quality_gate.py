#!/usr/bin/env python3
"""
Canyon v9 Step 212 - Proof Quality Gate.

Research-only. No broker connection. No live orders.

Step211 tells the user what proof to collect. Step212 validates whether a
filled proof row is strong enough to move toward the proof-to-acceptance
bridge. It checks required fields, source quality, news reaction checks, and
whether the row is actually marked Verified. It does not fetch sources and does
not approve evidence automatically.

Outputs:
  quant_fund_proof_quality_gate_state.json
  quant_fund_proof_quality_gate.csv
  quant_fund_proof_missing_fields.csv
  quant_fund_proof_ready_review.csv
  quant_fund_source_quality_rules.csv
  quant_fund_proof_quality_report.md
"""
from __future__ import annotations

from typing import Any

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


OUT_STATE = ROOT / "quant_fund_proof_quality_gate_state.json"
OUT_GATE = ROOT / "quant_fund_proof_quality_gate.csv"
OUT_MISSING = ROOT / "quant_fund_proof_missing_fields.csv"
OUT_READY = ROOT / "quant_fund_proof_ready_review.csv"
OUT_RULES = ROOT / "quant_fund_source_quality_rules.csv"
OUT_REPORT = ROOT / "quant_fund_proof_quality_report.md"


GATE_COLUMNS = [
    "proof_id",
    "ticker",
    "proof_type",
    "quality_state",
    "quality_score",
    "source_quality",
    "missing_fields",
    "required_fields",
    "source_name",
    "source_url",
    "observed_value",
    "reviewer",
    "review_date",
    "price_reaction_checked",
    "volume_reaction_checked",
    "what_to_fix",
    "can_send_to_acceptance_bridge",
    "why",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

MISSING_COLUMNS = [
    "ticker",
    "proof_id",
    "proof_type",
    "missing_field",
    "why_needed",
    "how_to_fill",
    "editable_file",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

READY_COLUMNS = [
    "ticker",
    "proof_id",
    "proof_type",
    "source_name",
    "observed_value",
    "reviewer",
    "review_date",
    "next_step",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

RULE_COLUMNS = [
    "rule",
    "source_examples",
    "score_band",
    "counts_as",
    "does_not_count_as",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def short(value: Any, limit: int = 300) -> str:
    text = " ".join(as_text(value, "").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def guard_flags(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["research_only"] = True
    out["no_broker_connection"] = True
    out["no_live_orders"] = True
    return out


def truthy(value: Any) -> bool:
    text = as_text(value, "").lower()
    return text in {"yes", "true", "checked", "y", "1", "verified"}


def normalize_status(value: Any) -> str:
    return as_text(value, "").strip().lower().replace("-", " ").replace("_", " ")


def proof_type(row: pd.Series) -> str:
    group = as_text(row.get("field_group"), "")
    if group:
        return group
    field = as_text(row.get("field_name"), "").lower()
    question = as_text(row.get("required_question"), "").lower()
    text = field + " " + question
    if "news" in text or "headline" in text:
        return "News proof"
    if "spread" in text or "liquidity" in text or "trading" in text:
        return "Trading cost"
    if "event" in text or "expected" in text or "options market" in text:
        return "Event risk"
    return "General proof"


def required_fields(row: pd.Series, ptype: str) -> list[str]:
    fields = ["source_name", "observed_value", "reviewer", "review_date"]
    missing_hint = as_text(row.get("missing_proof"), "").lower()
    if "source or observation time" in missing_hint or ptype in {"News proof"}:
        fields.append("observed_time")
    if ptype == "News proof":
        fields.extend(["price_reaction_checked", "volume_reaction_checked"])
    return fields


def source_quality(source_name: str, source_url: str, source_files: str, ptype: str) -> tuple[str, int, str]:
    text = " ".join([source_name, source_url, source_files]).lower()
    if not text.strip():
        return "No source yet", 0, "Fill source name and source URL or file."

    strong = [
        "investor", "company", "sec", "10-q", "10-k", "8-k", "form 4",
        "nasdaq", "nyse", "cboe", "occ", "option", "options chain",
        "yahoo", "quote", "broker", "reuters", "bloomberg", "dow jones",
        "wall street journal", "wsj", "marketwatch", "barron",
    ]
    local_usable = [
        "execution_cost_model", "earnings_gap_down_risk", "risk_book_seed_metric_detail",
        "event_readthrough_target_ranking", "event_causal_validation_queue",
        "pm_evidence_source_proof_input", "local price", "local liquidity",
    ]
    weak = ["model", "chatgpt", "guess", "unknown", "no source", "unverified", "screenshot"]

    if any(w in text for w in weak) and not any(s in text for s in strong):
        return "Weak source", 35, "Use a direct source with timestamp or a named local evidence file."
    if any(s in text for s in strong):
        return "Strong source", 90, "Direct source quality is acceptable if fields are filled."
    if any(s in text for s in local_usable):
        score = 75 if ptype != "News proof" else 65
        return "Usable local source", score, "Local source can support research, but news proof still needs timestamp and reaction checks."
    return "Needs source review", 55, "Source may be usable, but a human should confirm it is direct and current."


def build_gate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proof_input = read_csv_safe(ROOT / "pm_evidence_source_proof_input.csv")
    if proof_input.empty:
        return (
            pd.DataFrame(columns=GATE_COLUMNS),
            pd.DataFrame(columns=MISSING_COLUMNS),
            pd.DataFrame(columns=READY_COLUMNS),
        )

    gate_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []

    for _, row in proof_input.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        pid = as_text(row.get("proof_id"), "")
        ptype = proof_type(row)
        status = normalize_status(row.get("proof_status"))
        req_fields = required_fields(row, ptype)
        missing = [field for field in req_fields if not as_text(row.get(field), "")]
        source_name = as_text(row.get("source_name"), "")
        source_url = as_text(row.get("source_url"), "")
        source_files = as_text(row.get("source_files"), "")
        source_reference_required = not source_url and not source_files
        if source_reference_required and "source_url" not in missing:
            missing.append("source_url")
        display_req_fields = list(req_fields)
        if "source_url or source_files" not in display_req_fields:
            display_req_fields.append("source_url or source_files")
        sq, source_score, source_note = source_quality(source_name, source_url, source_files, ptype)
        reaction_missing = []
        if ptype == "News proof":
            if not truthy(row.get("price_reaction_checked")):
                reaction_missing.append("price_reaction_checked")
            if not truthy(row.get("volume_reaction_checked")):
                reaction_missing.append("volume_reaction_checked")
        for field in reaction_missing:
            if field not in missing:
                missing.append(field)

        verified = status == "verified"
        score = source_score
        if missing:
            score -= min(45, 8 * len(missing))
        if not verified:
            score -= 20
        score = max(0, min(100, int(score)))

        if not verified and missing:
            q_state = "Needs proof fill"
        elif missing:
            q_state = "Missing required fields"
        elif not verified:
            q_state = "Filled but not marked Verified"
        elif source_score < 60:
            q_state = "Weak source review"
        elif ptype == "News proof" and reaction_missing:
            q_state = "Needs price and volume reaction check"
        else:
            q_state = "Ready for acceptance bridge"

        fix_parts = []
        if not verified:
            fix_parts.append("Set Proof Status to Verified only after the source is real.")
        if missing:
            fix_parts.append("Fill " + ", ".join(missing) + ".")
        if source_score < 60:
            fix_parts.append(source_note)
        if not fix_parts:
            fix_parts.append("Send to Steps 206 and 207, then check Step204 acceptance patch.")

        why = (
            f"{ptype} proof needs {', '.join(display_req_fields)}. "
            f"Source quality: {sq}. {source_note}"
        )

        gate_rows.append(guard_flags({
            "proof_id": pid,
            "ticker": ticker,
            "proof_type": ptype,
            "quality_state": q_state,
            "quality_score": score,
            "source_quality": sq,
            "missing_fields": "; ".join(missing) if missing else "None",
            "required_fields": "; ".join(display_req_fields),
            "source_name": source_name,
            "source_url": source_url,
            "observed_value": short(row.get("observed_value"), 220),
            "reviewer": as_text(row.get("reviewer"), ""),
            "review_date": as_text(row.get("review_date"), ""),
            "price_reaction_checked": "Yes" if truthy(row.get("price_reaction_checked")) else "No",
            "volume_reaction_checked": "Yes" if truthy(row.get("volume_reaction_checked")) else "No",
            "what_to_fix": short(" ".join(fix_parts), 260),
            "can_send_to_acceptance_bridge": "Yes" if q_state == "Ready for acceptance bridge" else "No",
            "why": short(why, 320),
        }))

        for field in missing:
            missing_rows.append(guard_flags({
                "ticker": ticker,
                "proof_id": pid,
                "proof_type": ptype,
                "missing_field": field,
                "why_needed": missing_field_reason(field, ptype),
                "how_to_fill": missing_field_hint(field),
                "editable_file": "pm_evidence_source_proof_input.csv",
            }))

        if q_state == "Ready for acceptance bridge":
            ready_rows.append(guard_flags({
                "ticker": ticker,
                "proof_id": pid,
                "proof_type": ptype,
                "source_name": source_name,
                "observed_value": short(row.get("observed_value"), 220),
                "reviewer": as_text(row.get("reviewer"), ""),
                "review_date": as_text(row.get("review_date"), ""),
                "next_step": "Rerun Steps 206 and 207, then inspect proof-to-acceptance bridge.",
            }))

    gate = pd.DataFrame(gate_rows, columns=GATE_COLUMNS)
    missing_df = pd.DataFrame(missing_rows, columns=MISSING_COLUMNS)
    ready = pd.DataFrame(ready_rows, columns=READY_COLUMNS)
    return gate, missing_df, ready


def missing_field_reason(field: str, ptype: str) -> str:
    if field == "source_name":
        return "A human needs to know where the proof came from."
    if field == "source_url":
        return "A link or file path makes the proof inspectable."
    if field == "observed_value":
        return "The system needs the actual value observed from the source."
    if field == "observed_time":
        return "Timing matters, especially for news and event proof."
    if field == "reviewer":
        return "A human reviewer must own the evidence."
    if field == "review_date":
        return "The proof must show when it was reviewed."
    if field == "price_reaction_checked":
        return "News proof needs market reaction, not only a headline."
    if field == "volume_reaction_checked":
        return "Volume confirms whether the market cared about the headline."
    return f"{ptype} proof requires this field."


def missing_field_hint(field: str) -> str:
    hints = {
        "source_name": "Enter the source name, such as Yahoo Finance quote, company IR, Reuters, or local execution file.",
        "source_url": "Enter a URL or local file name when available.",
        "observed_value": "Enter the exact observed value, such as spread bps, expected move %, headline timestamp, or liquidity date.",
        "observed_time": "Enter the source time or the time you checked the source.",
        "reviewer": "Enter the human reviewer name or initials.",
        "review_date": "Enter today's review date.",
        "price_reaction_checked": "Set to Yes only after checking post-event price reaction.",
        "volume_reaction_checked": "Set to Yes only after checking post-event volume reaction.",
    }
    return hints.get(field, "Fill this field from the source.")


def build_rules() -> pd.DataFrame:
    rows = [
        {
            "rule": "Strong source",
            "source_examples": "Company IR, SEC filing, exchange/option chain, current quote page, Reuters/Bloomberg/Dow Jones.",
            "score_band": "80-100",
            "counts_as": "Can pass if required fields are filled and proof is marked Verified.",
            "does_not_count_as": "Still not enough if observed value, reviewer, or review date is blank.",
        },
        {
            "rule": "Usable local source",
            "source_examples": "execution_cost_model.csv, earnings_gap_down_risk.csv, risk_book_seed_metric_detail.csv.",
            "score_band": "60-79",
            "counts_as": "Useful for research proof, especially trading cost or event-risk fallback.",
            "does_not_count_as": "Not enough for institutional historical claims or news proof without timing/reaction checks.",
        },
        {
            "rule": "Needs source review",
            "source_examples": "Named but indirect source, manually entered note, unclear timestamp.",
            "score_band": "45-59",
            "counts_as": "Needs human review before acceptance bridge.",
            "does_not_count_as": "Does not unlock evidence by itself.",
        },
        {
            "rule": "Weak source",
            "source_examples": "Model summary, guess, unknown source, screenshot without timestamp.",
            "score_band": "0-44",
            "counts_as": "Context only.",
            "does_not_count_as": "Cannot unlock Step204 acceptance.",
        },
    ]
    return pd.DataFrame([guard_flags(r) for r in rows], columns=RULE_COLUMNS)


def build_state(gate: pd.DataFrame, missing: pd.DataFrame, ready: pd.DataFrame) -> dict[str, Any]:
    counts = gate["quality_state"].value_counts().to_dict() if not gate.empty else {}
    first = gate.iloc[0].to_dict() if not gate.empty else {}
    return {
        "date": today_str(),
        "status": "Active",
        "proof_rows": int(len(gate)),
        "ready_rows": int(len(ready)),
        "missing_field_rows": int(len(missing)),
        "needs_fill_rows": int(counts.get("Needs proof fill", 0)),
        "weak_source_rows": int(counts.get("Weak source review", 0)),
        "first_ticker": as_text(first.get("ticker"), ""),
        "first_state": as_text(first.get("quality_state"), ""),
        "first_fix": as_text(first.get("what_to_fix"), ""),
        "plain_answer": (
            f"Proof quality gate is active. {len(ready)} proof rows are ready for the acceptance bridge. "
            f"{len(missing)} required fields are still missing."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    gate, missing, ready = build_gate()
    rules = build_rules()
    state = build_state(gate, missing, ready)

    gate.to_csv(OUT_GATE, index=False)
    missing.to_csv(OUT_MISSING, index=False)
    ready.to_csv(OUT_READY, index=False)
    rules.to_csv(OUT_RULES, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Proof Quality Gate\n\n" + df_to_markdown(gate.head(120)),
        "## Missing Fields\n\n" + df_to_markdown(missing.head(160)),
        "## Ready For Review\n\n" + df_to_markdown(ready.head(80)),
        "## Source Quality Rules\n\n" + df_to_markdown(rules),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Proof Quality Gate", sections)
    print(
        "Step212 complete: "
        f"{len(gate)} proof rows, {len(ready)} ready rows, {len(missing)} missing-field rows."
    )


if __name__ == "__main__":
    main()
