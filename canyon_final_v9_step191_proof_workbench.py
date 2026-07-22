#!/usr/bin/env python3
"""
Canyon v9 Step 191 - Proof Workbench.

Research-only. No broker connection. No live orders.

Step190 made the Performance page readable. Step191 makes it usable: it groups
the proof queue into work buckets and creates a fillable manual proof template.
This is not another signal. It is the desk workflow required before a ticker
can move from "research idea" to "watch-only review".

Outputs:
  sharpe4_proof_workbench_state.json
  sharpe4_proof_workbench_task_groups.csv
  sharpe4_proof_workbench_ticker_tasks.csv
  sharpe4_manual_proof_input_template.csv
  sharpe4_proof_workbench_report.md
"""
from __future__ import annotations

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


OUT_STATE = ROOT / "sharpe4_proof_workbench_state.json"
OUT_GROUPS = ROOT / "sharpe4_proof_workbench_task_groups.csv"
OUT_TASKS = ROOT / "sharpe4_proof_workbench_ticker_tasks.csv"
OUT_TEMPLATE = ROOT / "sharpe4_manual_proof_input_template.csv"
OUT_REPORT = ROOT / "sharpe4_proof_workbench_report.md"


BUCKETS = {
    "Fill earnings date and gap risk": {
        "bucket": "Earnings and gap proof",
        "purpose": "Know whether a ticker is about to face a jump-risk event.",
        "what_to_collect": "Next earnings date, report timing, expected/implied move, last surprise, and whether exposure should be flat/reduced before the event.",
        "how_to_verify": "Use a traceable source such as company IR, exchange calendar, Yahoo/market calendar, and local options/IV file when available.",
        "done_when": "The ticker has an earnings date, expected move or explicit no-data note, and a written keep/reduce/avoid rule.",
        "forbidden_until_done": "No paper size. No calls or puts.",
        "priority": 1,
    },
    "Write a tail-risk stop plan": {
        "bucket": "Tail-risk stop proof",
        "purpose": "Know the loss boundary before any watch upgrade.",
        "what_to_collect": "1-day CVaR, 5-day CVaR, max starter cap, stop/invalidation level, and what would force the idea to be removed.",
        "how_to_verify": "Use local VaR/CVaR outputs plus price chart support/resistance; write the stop as a rule, not a feeling.",
        "done_when": "The risk book has a max starter cap and a hard paper stop rule.",
        "forbidden_until_done": "No normal starter size and no option premium risk.",
        "priority": 2,
    },
    "Check crowding against peers": {
        "bucket": "Crowding and overlap proof",
        "purpose": "Avoid pretending two highly correlated names are diversification.",
        "what_to_collect": "Highest correlated peer, shared sector/theme, combined exposure cap, and whether the ticker duplicates another AI/semi/software bet.",
        "how_to_verify": "Use correlation proxy plus sector/theme map; if correlation is high, set one combined cap.",
        "done_when": "The ticker has a written exposure bucket and combined cap.",
        "forbidden_until_done": "No separate size bucket and no extra options route.",
        "priority": 3,
    },
    "Capture live spread proof": {
        "bucket": "Spread and fill proof",
        "purpose": "Stop paper Sharpe from being fake because fills are too optimistic.",
        "what_to_collect": "Bid/ask spread, expected fill quality, participation rate, and current/stress TCA.",
        "how_to_verify": "Use a manual quote snapshot or better intraday quote source, then compare with local TCA estimate.",
        "done_when": "Spread and TCA are recorded and within the desk budget.",
        "forbidden_until_done": "No paper route and no options route.",
        "priority": 4,
    },
    "Validate event-time reaction": {
        "bucket": "Event reaction proof",
        "purpose": "Prove the headline actually matters to the ticker or supply chain.",
        "what_to_collect": "Source headline, event timestamp/date, target role, 1-day/3-day price reaction, volume reaction, and whether peer/read-through names moved too.",
        "how_to_verify": "Compare event time to price/volume move and check whether the move faded; do not rely on headline tone alone.",
        "done_when": "The risk book records whether the event helped, hurt, or failed to move the ticker.",
        "forbidden_until_done": "No watch upgrade, no call/put search, and no alpha credit.",
        "priority": 5,
    },
    "Find source event proof": {
        "bucket": "Source proof",
        "purpose": "Attach a readable source before the model uses a story.",
        "what_to_collect": "Headline, publisher/source, source ticker, target ticker, and why it should help or hurt.",
        "how_to_verify": "Use only traceable local or external source records; write the read-through link plainly.",
        "done_when": "The source can be opened and the ticker link is explained.",
        "forbidden_until_done": "No event score upgrade.",
        "priority": 6,
    },
}


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return default
    return out if np.isfinite(out) else default


def shorten(text: Any, limit: int = 190) -> str:
    clean = " ".join(as_text(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def bucket_for(proof: str) -> dict[str, Any]:
    if proof in BUCKETS:
        return BUCKETS[proof]
    return {
        "bucket": "Other manual proof",
        "purpose": "A manual proof item is required before promotion.",
        "what_to_collect": proof or "Complete the missing proof item.",
        "how_to_verify": "Record the source, the fact, and the reviewer note.",
        "done_when": "The proof is written into the manual proof template.",
        "forbidden_until_done": "No paper size. No options.",
        "priority": 9,
    }


def build_tasks() -> pd.DataFrame:
    queue = read_csv_safe(ROOT / "sharpe4_simple_candidate_queue.csv")
    gate = read_csv_safe(ROOT / "sharpe4_risk_book_promotion_gate.csv")
    event = read_csv_safe(ROOT / "sharpe4_risk_book_event_route.csv")
    varliq = read_csv_safe(ROOT / "sharpe4_risk_book_var_liquidity.csv")
    corr = read_csv_safe(ROOT / "sharpe4_risk_book_correlation_proxy.csv")

    if queue.empty or "ticker" not in queue.columns:
        return pd.DataFrame()

    base = queue.copy()
    base["ticker"] = base["ticker"].apply(clean_ticker)

    for df, cols in [
        (gate, ["ticker", "promotion_status", "option_gate", "risk_level", "earnings", "correlation", "liquidity", "status_reason"]),
        (event, ["ticker", "event_score", "event_role", "event_headline", "source_news_ticker", "publisher_sample", "earnings_status", "iv_rank"]),
        (varliq, ["ticker", "annual_vol_pct", "daily_cvar_95_pct", "five_day_cvar_95_pct", "estimated_tca_bps", "price_risk"]),
        (corr, ["ticker", "highest_peer", "highest_peer_corr", "correlation_risk"]),
    ]:
        if df.empty or "ticker" not in df.columns:
            continue
        work = df.copy()
        work["ticker"] = work["ticker"].apply(clean_ticker)
        keep = [c for c in cols if c in work.columns]
        base = base.merge(work[keep].drop_duplicates("ticker"), on="ticker", how="left")

    rows = []
    for _, row in base.iterrows():
        proof = as_text(row.get("first_proof"))
        meta = bucket_for(proof)
        ticker = clean_ticker(row.get("ticker"))
        risk_line = []
        if as_text(row.get("risk_level")):
            risk_line.append(f"risk {row.get('risk_level')}")
        if as_text(row.get("earnings")):
            risk_line.append(f"earnings {row.get('earnings')}")
        if as_text(row.get("correlation")):
            risk_line.append(f"correlation {row.get('correlation')}")
        if safe_float(row.get("daily_cvar_95_pct"), np.nan) == safe_float(row.get("daily_cvar_95_pct"), np.nan):
            risk_line.append(f"1d CVaR {safe_float(row.get('daily_cvar_95_pct')):.2f}%")

        rows.append({
            "work_bucket": meta["bucket"],
            "bucket_priority": int(meta["priority"]),
            "ticker": ticker,
            "simple_status": as_text(row.get("simple_status")),
            "proof_to_collect": proof,
            "why_this_matters": as_text(row.get("why")) or meta["purpose"],
            "exact_next_step": as_text(row.get("what_to_do_next")) or meta["what_to_collect"],
            "what_to_collect": meta["what_to_collect"],
            "how_to_verify": meta["how_to_verify"],
            "done_when": meta["done_when"],
            "still_forbidden": meta["forbidden_until_done"],
            "risk_snapshot": "; ".join(risk_line),
            "option_gate": shorten(row.get("option_gate"), 180),
            "source_headline": shorten(row.get("source_headline") or row.get("event_headline"), 160),
            "source_hint": shorten(row.get("publisher_sample"), 160),
            "event_role": as_text(row.get("event_role")),
            "highest_peer": as_text(row.get("highest_peer")),
            "source_files": "sharpe4_simple_candidate_queue.csv / promotion_gate / event_route / var_liquidity / correlation_proxy",
            "research_only": True,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["bucket_priority", "ticker"]).reset_index(drop=True)
        out.insert(0, "task_rank", range(1, len(out) + 1))
    return out


def build_groups(tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame()
    rows = []
    for bucket, sub in tasks.groupby("work_bucket", sort=False):
        priority = int(sub["bucket_priority"].min())
        meta = next((v for v in BUCKETS.values() if v["bucket"] == bucket), None)
        if meta is None:
            meta = {
                "purpose": "Manual proof is required.",
                "what_to_collect": "Collect the missing proof.",
                "how_to_verify": "Record source and reviewer note.",
                "done_when": "Proof is written.",
                "forbidden_until_done": "No paper size. No options.",
            }
        rows.append({
            "bucket_priority": priority,
            "work_bucket": bucket,
            "ticker_count": int(len(sub)),
            "tickers": ", ".join(sub["ticker"].head(12).tolist()),
            "purpose": meta["purpose"],
            "what_to_collect": meta["what_to_collect"],
            "how_to_verify": meta["how_to_verify"],
            "done_when": meta["done_when"],
            "still_forbidden": meta["forbidden_until_done"],
        })
    out = pd.DataFrame(rows).sort_values("bucket_priority").reset_index(drop=True)
    out.insert(0, "group_rank", range(1, len(out) + 1))
    return out


def build_template(tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame(columns=[
            "ticker", "work_bucket", "proof_to_collect", "source_name",
            "source_url_or_file", "source_date_or_timestamp", "key_numbers",
            "evidence_summary", "pass_fail_review", "reviewer_notes",
            "next_gate_request", "date_recorded",
        ])
    rows = []
    for _, row in tasks.iterrows():
        rows.append({
            "ticker": row["ticker"],
            "work_bucket": row["work_bucket"],
            "proof_to_collect": row["proof_to_collect"],
            "source_name": "",
            "source_url_or_file": "",
            "source_date_or_timestamp": "",
            "key_numbers": "",
            "evidence_summary": "",
            "pass_fail_review": "",
            "reviewer_notes": "",
            "next_gate_request": "Keep blocked / Move to watch-only review / Remove from queue",
            "date_recorded": "",
        })
    return pd.DataFrame(rows)


def has_manual_content(row: pd.Series) -> bool:
    manual_cols = [
        "source_name",
        "source_url_or_file",
        "source_date_or_timestamp",
        "key_numbers",
        "evidence_summary",
        "pass_fail_review",
        "reviewer_notes",
        "date_recorded",
    ]
    return any(as_text(row.get(col)) for col in manual_cols)


def preserve_existing_template(new_template: pd.DataFrame) -> pd.DataFrame:
    """
    Keep user-filled proof rows when Step191 is rerun.

    The template is meant to be filled manually. Rebuilding the workbench should
    add or reorder tasks, but it must not erase evidence that Lynn already wrote.
    """
    existing = read_csv_safe(OUT_TEMPLATE)
    if existing.empty or new_template.empty:
        return new_template

    key_cols = ["ticker", "work_bucket", "proof_to_collect"]
    manual_cols = [
        "source_name",
        "source_url_or_file",
        "source_date_or_timestamp",
        "key_numbers",
        "evidence_summary",
        "pass_fail_review",
        "reviewer_notes",
        "next_gate_request",
        "date_recorded",
    ]
    for col in key_cols + manual_cols:
        if col not in existing.columns:
            existing[col] = ""
        if col not in new_template.columns:
            new_template[col] = ""

    existing = existing.copy()
    new_template = new_template.copy()
    existing["ticker"] = existing["ticker"].apply(clean_ticker)
    new_template["ticker"] = new_template["ticker"].apply(clean_ticker)

    preserved = existing[existing.apply(has_manual_content, axis=1)].copy()
    if preserved.empty:
        return new_template

    lookup = preserved.drop_duplicates(key_cols, keep="last").set_index(key_cols)
    merged_rows = []
    seen_keys = set()
    for _, row in new_template.iterrows():
        out = row.copy()
        key = tuple(out.get(col, "") for col in key_cols)
        seen_keys.add(key)
        if key in lookup.index:
            old = lookup.loc[key]
            if isinstance(old, pd.DataFrame):
                old = old.iloc[-1]
            for col in manual_cols:
                old_value = as_text(old.get(col))
                if old_value:
                    out[col] = old_value
        merged_rows.append(out)

    merged = pd.DataFrame(merged_rows)
    orphaned = []
    for _, row in preserved.iterrows():
        key = tuple(row.get(col, "") for col in key_cols)
        if key not in seen_keys:
            orphaned.append(row[[c for c in merged.columns if c in row.index]])
    if orphaned:
        merged = pd.concat([merged, pd.DataFrame(orphaned)], ignore_index=True)

    return merged


def write_report(state: dict[str, Any], groups: pd.DataFrame, tasks: pd.DataFrame) -> None:
    sections = [
        "Research-only. No broker connection. No live orders.",
        "\n".join([
            "## Current Answer",
            "",
            f"- Workbench status: **{state['status']}**",
            f"- Total tasks: **{state['total_tasks']}**",
            f"- Work buckets: **{state['work_bucket_count']}**",
            f"- First bucket: **{state['first_bucket']}**",
            f"- Paper sizing allowed now: **{state['paper_sizing_allowed_now_count']}**",
            f"- Options allowed now: **{state['options_allowed_now_count']}**",
            "",
            state["plain_english"],
        ]),
        "## Work Buckets\n\n" + df_to_markdown(groups),
        "## Ticker Tasks\n\n" + df_to_markdown(tasks.head(30)),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Step 191 - Proof Workbench", sections)


def main() -> None:
    tasks = build_tasks()
    groups = build_groups(tasks)
    template = preserve_existing_template(build_template(tasks))

    tasks.to_csv(OUT_TASKS, index=False)
    groups.to_csv(OUT_GROUPS, index=False)
    template.to_csv(OUT_TEMPLATE, index=False)

    state = {
        "date": today_str(),
        "status": "PROOF_WORKBENCH_ACTIVE" if not tasks.empty else "NO_PROOF_TASKS",
        "total_tasks": int(len(tasks)),
        "work_bucket_count": int(len(groups)),
        "first_bucket": groups.iloc[0]["work_bucket"] if not groups.empty else "No tasks",
        "first_bucket_ticker_count": int(groups.iloc[0]["ticker_count"]) if not groups.empty else 0,
        "paper_sizing_allowed_now_count": 0,
        "options_allowed_now_count": 0,
        "plain_english": "Use this as the actual desk workflow. Fill the template first; do not size or use options until the proof queue is reviewed.",
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }
    write_json(OUT_STATE, state)
    write_report(state, groups, tasks)

    print(f"[OK] Wrote {OUT_STATE.name}")
    print(f"[OK] Tasks: {state['total_tasks']}")
    print(f"[OK] Buckets: {state['work_bucket_count']}")
    print(f"[OK] First bucket: {state['first_bucket']}")


if __name__ == "__main__":
    main()
