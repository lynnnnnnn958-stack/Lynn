#!/usr/bin/env python3
"""
Canyon v9 Step 213 - Proof Fill Desk.

Research-only. No broker connection. No live orders.

Step212 says which proof rows are incomplete. Step213 turns those missing
fields into a plain work desk: which ticker to handle first, what question to
answer, where the evidence should come from, which fields to fill, and what to
rerun after the human proof is entered.

Outputs:
  quant_fund_proof_fill_desk_state.json
  quant_fund_proof_fill_cards.csv
  quant_fund_proof_fill_ticker_plan.csv
  quant_fund_proof_fill_field_recipes.csv
  quant_fund_proof_fill_copy_sheet.csv
  quant_fund_proof_fill_quality_check.csv
  quant_fund_proof_fill_desk_report.md
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


OUT_STATE = ROOT / "quant_fund_proof_fill_desk_state.json"
OUT_CARDS = ROOT / "quant_fund_proof_fill_cards.csv"
OUT_TICKER_PLAN = ROOT / "quant_fund_proof_fill_ticker_plan.csv"
OUT_FIELD_RECIPES = ROOT / "quant_fund_proof_fill_field_recipes.csv"
OUT_COPY_SHEET = ROOT / "quant_fund_proof_fill_copy_sheet.csv"
OUT_QA = ROOT / "quant_fund_proof_fill_quality_check.csv"
OUT_REPORT = ROOT / "quant_fund_proof_fill_desk_report.md"


CARD_COLUMNS = [
    "card_rank",
    "ticker",
    "proof_type",
    "priority_score",
    "plain_task",
    "question_to_answer",
    "source_to_open",
    "fields_to_fill_now",
    "fill_order",
    "good_example",
    "bad_example",
    "why_this_blocks_progress",
    "proof_id",
    "editable_file",
    "local_source_files",
    "suggested_value",
    "after_filling",
    "do_not_do",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

TICKER_COLUMNS = [
    "ticker_rank",
    "ticker",
    "open_proof_rows",
    "missing_field_rows",
    "first_proof_type",
    "first_question",
    "first_source_to_open",
    "first_fields_to_fill",
    "why_this_ticker_first",
    "estimated_minutes",
    "after_done",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

RECIPE_COLUMNS = [
    "proof_type",
    "field_to_fill",
    "plain_label",
    "what_it_means",
    "where_to_find_it",
    "what_to_type",
    "good_example",
    "bad_example",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

COPY_COLUMNS = [
    "ticker",
    "proof_type",
    "proof_id",
    "field_to_fill",
    "plain_label",
    "what_to_type",
    "where_to_find_it",
    "editable_file",
    "source_to_open",
    "question_to_answer",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

QA_COLUMNS = [
    "check",
    "status",
    "bad_rows",
    "what_it_checked",
    "fix_hint",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]


FIELD_LABELS = {
    "source_name": "Source name",
    "source_url": "Source URL or source file",
    "observed_value": "Observed value",
    "observed_time": "Source time",
    "reviewer": "Reviewer",
    "review_date": "Review date",
    "price_reaction_checked": "Price reaction checked",
    "volume_reaction_checked": "Volume reaction checked",
}


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


def short(value: Any, limit: int = 260) -> str:
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


def field_label(field: str) -> str:
    return FIELD_LABELS.get(str(field), str(field).replace("_", " ").title())


def field_labels(fields: list[str]) -> str:
    return "; ".join(field_label(f) for f in fields)


def proof_sort_bonus(ptype: str) -> int:
    if ptype == "News proof":
        return 18
    if ptype == "Event risk":
        return 14
    if ptype == "Trading cost":
        return 10
    return 0


def source_hint(proof_type: str, preferred_source: str, source_files: str) -> str:
    preferred = as_text(preferred_source, "")
    files = as_text(source_files, "")
    if preferred:
        return preferred
    if proof_type == "News proof":
        return "Open the original article, its timestamp, and the linked-stock map. Then check price and volume after the headline."
    if proof_type == "Event risk":
        return "Open the options chain, earnings event file, or documented expected-move source."
    if proof_type == "Trading cost":
        return "Open a current quote page, broker quote snapshot, execution-cost file, or liquidity file."
    if files:
        return f"Start with local evidence file: {files}."
    return "Open a direct source that can be inspected later."


def field_recipe(proof_type: str, field: str, source_to_open: str = "") -> dict[str, str]:
    common = {
        "source_name": {
            "what_it_means": "Where the proof came from.",
            "where_to_find_it": source_to_open or "Use the direct source you opened.",
            "what_to_type": "Type the source name, not a model summary.",
            "good_example": "Yahoo Finance quote page; company investor relations; SEC filing; Reuters article.",
            "bad_example": "ChatGPT said so; unknown; screenshot only.",
        },
        "source_url": {
            "what_it_means": "A link or local file that lets someone inspect the proof again.",
            "where_to_find_it": source_to_open or "Use the source page URL or local evidence file name.",
            "what_to_type": "Type the URL if available, otherwise the local file name.",
            "good_example": "https://finance.yahoo.com/quote/AAPL or execution_cost_model.csv.",
            "bad_example": "Blank; trust me; no link.",
        },
        "observed_value": {
            "what_it_means": "The actual number or fact you saw in the source.",
            "where_to_find_it": source_to_open or "Read the source and copy the relevant number or fact.",
            "what_to_type": "Type the exact value, with units when possible.",
            "good_example": "Expected move 4.8%; bid/ask spread 2.1 bps; headline time 09:31 ET.",
            "bad_example": "Looks good; bullish; bad news.",
        },
        "observed_time": {
            "what_it_means": "When the source was observed or when the news happened.",
            "where_to_find_it": "Use the article timestamp, quote snapshot time, or the time you checked the source.",
            "what_to_type": "Type date and time if available.",
            "good_example": "2026-06-05 09:45 ET.",
            "bad_example": "Today-ish; recently.",
        },
        "reviewer": {
            "what_it_means": "The human who checked the proof.",
            "where_to_find_it": "This comes from you or the reviewer.",
            "what_to_type": "Type the reviewer name or initials.",
            "good_example": "Lynn; LR; RJ.",
            "bad_example": "System; model; blank.",
        },
        "review_date": {
            "what_it_means": "When the proof was reviewed.",
            "where_to_find_it": "Use today's review date.",
            "what_to_type": "Type the review date.",
            "good_example": today_str(),
            "bad_example": "Old; someday; blank.",
        },
        "price_reaction_checked": {
            "what_it_means": "Whether price reacted after the headline.",
            "where_to_find_it": "Check intraday or daily price after the news timestamp.",
            "what_to_type": "Type Yes only if you checked it. Otherwise type No.",
            "good_example": "Yes, price rose 3.2% after headline; or No, not checked.",
            "bad_example": "Probably; maybe.",
        },
        "volume_reaction_checked": {
            "what_it_means": "Whether volume confirmed the market cared about the headline.",
            "where_to_find_it": "Check volume spike or relative volume after the news timestamp.",
            "what_to_type": "Type Yes only if you checked it. Otherwise type No.",
            "good_example": "Yes, volume was 2.1x normal after headline; or No, not checked.",
            "bad_example": "Feels active; maybe.",
        },
    }
    recipe = dict(common.get(field, {
        "what_it_means": "A required proof field.",
        "where_to_find_it": source_to_open or "Use the direct source.",
        "what_to_type": "Type the observed proof value.",
        "good_example": "A timestamped value from a direct source.",
        "bad_example": "Blank or model-only text.",
    }))
    if proof_type == "News proof" and field in {"observed_value", "observed_time"}:
        recipe["where_to_find_it"] = "Open the original news article and linked-stock map, then check price and volume after the timestamp."
        recipe["good_example"] = "Headline at 09:31 ET; price +2.4% by close; volume 1.8x normal."
    if proof_type == "Trading cost" and field == "observed_value":
        recipe["good_example"] = "Bid 182.10 / ask 182.14; spread 2.2 bps; ADV liquid."
    if proof_type == "Event risk" and field == "observed_value":
        recipe["good_example"] = "Options-implied event move 5.6%; source checked before close."
    return recipe


def normalize_missing_fields(value: Any) -> list[str]:
    text = as_text(value, "")
    if not text or text.lower() == "none":
        return []
    out = []
    for part in text.replace(",", ";").split(";"):
        field = part.strip()
        if field:
            out.append(field)
    return out


def row_by_proof_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if df.empty or "proof_id" not in df.columns:
        return out
    for _, row in df.iterrows():
        pid = as_text(row.get("proof_id"), "")
        if pid and pid not in out:
            out[pid] = row
    return out


def build_fill_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate = read_csv_safe(ROOT / "quant_fund_proof_quality_gate.csv")
    missing = read_csv_safe(ROOT / "quant_fund_proof_missing_fields.csv")
    proof_input = read_csv_safe(ROOT / "pm_evidence_source_proof_input.csv")
    tasks = read_csv_safe(ROOT / "quant_fund_proof_task_cards.csv")

    if gate.empty:
        empty_cards = pd.DataFrame(columns=CARD_COLUMNS)
        return (
            empty_cards,
            pd.DataFrame(columns=TICKER_COLUMNS),
            pd.DataFrame(columns=RECIPE_COLUMNS),
            pd.DataFrame(columns=COPY_COLUMNS),
            pd.DataFrame(columns=QA_COLUMNS),
        )

    input_by_id = row_by_proof_id(proof_input)
    task_by_id = row_by_proof_id(tasks)
    missing_by_id: dict[str, list[str]] = {}
    if not missing.empty:
        for pid, grp in missing.groupby("proof_id"):
            missing_by_id[as_text(pid)] = [as_text(x) for x in grp["missing_field"].tolist() if as_text(x)]

    card_rows: list[dict[str, Any]] = []
    recipe_rows: dict[tuple[str, str], dict[str, Any]] = {}
    copy_rows: list[dict[str, Any]] = []

    for _, gate_row in gate.iterrows():
        pid = as_text(gate_row.get("proof_id"), "")
        ticker = clean_ticker(gate_row.get("ticker"))
        ptype = as_text(gate_row.get("proof_type"), "General proof")
        missing_fields = missing_by_id.get(pid) or normalize_missing_fields(gate_row.get("missing_fields"))
        if not missing_fields:
            continue

        input_row = input_by_id.get(pid, pd.Series(dtype=object))
        task_row = task_by_id.get(pid, pd.Series(dtype=object))
        question = (
            as_text(task_row.get("question_to_answer"), "")
            or as_text(input_row.get("required_question"), "")
            or f"What proof is still missing for {ticker}?"
        )
        preferred_source = (
            as_text(task_row.get("acceptable_source"), "")
            or as_text(input_row.get("preferred_source"), "")
            or as_text(input_row.get("acceptable_proof"), "")
        )
        source_files = as_text(input_row.get("source_files"), "") or as_text(task_row.get("source_files"), "")
        source_to_open = source_hint(ptype, preferred_source, source_files)
        suggested_value = as_text(input_row.get("suggested_value"), "") or as_text(task_row.get("suggested_value"), "")
        review_score = pd.to_numeric(pd.Series([input_row.get("review_priority_score")]), errors="coerce").fillna(50).iloc[0]
        quality_score = pd.to_numeric(pd.Series([gate_row.get("quality_score")]), errors="coerce").fillna(0).iloc[0]
        priority_score = int(min(100, max(0, float(review_score) + proof_sort_bonus(ptype) + len(missing_fields) * 2 - float(quality_score) * 0.15)))
        labels = field_labels(missing_fields)

        first_recipe = field_recipe(ptype, missing_fields[0], source_to_open)
        good = first_recipe["good_example"]
        bad = "Blank fields, old screenshots, or model-only text do not count."
        why = (
            f"{ticker} cannot move forward because {labels} is still missing. "
            "Until this is filled, do not look for new size, calls, puts, or trade routes."
        )
        after = "Enter the fields in the proof input file. Set Proof Status to Verified only if the source is real, then rerun Steps 206, 207, 204, 212, and 213."
        do_not = "Do not use model text as proof. Do not mark Verified unless a human can inspect the source."

        card_rows.append(guard_flags({
            "card_rank": 0,
            "ticker": ticker,
            "proof_type": ptype,
            "priority_score": priority_score,
            "plain_task": f"Fill proof for {ticker}: {labels}.",
            "question_to_answer": short(question, 320),
            "source_to_open": short(source_to_open, 320),
            "fields_to_fill_now": labels,
            "fill_order": " -> ".join(field_label(f) for f in missing_fields),
            "good_example": short(good, 240),
            "bad_example": bad,
            "why_this_blocks_progress": short(why, 360),
            "proof_id": pid,
            "editable_file": "pm_evidence_source_proof_input.csv",
            "local_source_files": source_files,
            "suggested_value": short(suggested_value, 220),
            "after_filling": after,
            "do_not_do": do_not,
        }))

        for field in missing_fields:
            recipe = field_recipe(ptype, field, source_to_open)
            recipe_key = (ptype, field)
            if recipe_key not in recipe_rows:
                recipe_rows[recipe_key] = guard_flags({
                    "proof_type": ptype,
                    "field_to_fill": field,
                    "plain_label": field_label(field),
                    "what_it_means": recipe["what_it_means"],
                    "where_to_find_it": recipe["where_to_find_it"],
                    "what_to_type": recipe["what_to_type"],
                    "good_example": recipe["good_example"],
                    "bad_example": recipe["bad_example"],
                })
            copy_rows.append(guard_flags({
                "ticker": ticker,
                "proof_type": ptype,
                "proof_id": pid,
                "field_to_fill": field,
                "plain_label": field_label(field),
                "what_to_type": recipe["what_to_type"],
                "where_to_find_it": short(recipe["where_to_find_it"], 320),
                "editable_file": "pm_evidence_source_proof_input.csv",
                "source_to_open": short(source_to_open, 320),
                "question_to_answer": short(question, 260),
            }))

    cards = pd.DataFrame(card_rows, columns=CARD_COLUMNS)
    if not cards.empty:
        cards = cards.sort_values(["priority_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
        cards["card_rank"] = range(1, len(cards) + 1)

    ticker_rows: list[dict[str, Any]] = []
    if not cards.empty:
        for idx, (ticker, grp) in enumerate(cards.groupby("ticker", sort=False), start=1):
            first = grp.iloc[0]
            missing_count = sum(len(normalize_missing_fields(x)) for x in grp["fields_to_fill_now"])
            ticker_rows.append(guard_flags({
                "ticker_rank": idx,
                "ticker": ticker,
                "open_proof_rows": int(len(grp)),
                "missing_field_rows": int(missing_count),
                "first_proof_type": first["proof_type"],
                "first_question": first["question_to_answer"],
                "first_source_to_open": first["source_to_open"],
                "first_fields_to_fill": first["fields_to_fill_now"],
                "why_this_ticker_first": first["why_this_blocks_progress"],
                "estimated_minutes": int(max(5, min(45, 4 * missing_count + 3 * len(grp)))),
                "after_done": "Rerun Steps 206, 207, 204, 212, and 213.",
            }))

    ticker_plan = pd.DataFrame(ticker_rows, columns=TICKER_COLUMNS)
    recipes = pd.DataFrame(list(recipe_rows.values()), columns=RECIPE_COLUMNS)
    copy_sheet = pd.DataFrame(copy_rows, columns=COPY_COLUMNS)
    qa = build_qa(cards, ticker_plan, recipes, copy_sheet)
    return cards, ticker_plan, recipes, copy_sheet, qa


def build_qa(cards: pd.DataFrame, ticker_plan: pd.DataFrame, recipes: pd.DataFrame, copy_sheet: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, bad_rows: int, what: str, fix: str) -> None:
        rows.append(guard_flags({
            "check": check,
            "status": "PASS" if ok else "REVIEW",
            "bad_rows": int(bad_rows),
            "what_it_checked": what,
            "fix_hint": fix,
        }))

    add(
        "has_fill_cards",
        not cards.empty,
        0 if not cards.empty else 1,
        "There should be cards telling the user what proof to fill first.",
        "Run Steps 211 and 212 before Step213.",
    )
    raw_tokens = ["DATA_GAP", "SIZE_DOWN", "NO_GO", "NEEDS_REVIEW", "PENDING_MANUAL_CHECKS"]
    bad_text_rows = 0
    for df in [cards, ticker_plan, recipes, copy_sheet]:
        if df.empty:
            continue
        text = "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())
        bad_text_rows += sum(text.count(tok) for tok in raw_tokens)
    add(
        "no_raw_machine_tokens",
        bad_text_rows == 0,
        bad_text_rows,
        "User-facing proof fill outputs should not show raw machine states.",
        "Replace raw tokens with plain English before display.",
    )
    add(
        "has_field_recipes",
        not recipes.empty,
        0 if not recipes.empty else 1,
        "Every field type should explain where to find it and what to type.",
        "Add field recipes for missing proof fields.",
    )
    add(
        "has_copy_sheet",
        not copy_sheet.empty,
        0 if not copy_sheet.empty else 1,
        "There should be one row per field to fill for precise manual work.",
        "Rebuild from quant_fund_proof_missing_fields.csv.",
    )
    if not cards.empty:
        missing_question = int(cards["question_to_answer"].astype(str).str.strip().eq("").sum())
        missing_source = int(cards["source_to_open"].astype(str).str.strip().eq("").sum())
    else:
        missing_question = 0
        missing_source = 0
    add(
        "cards_have_question_and_source",
        missing_question + missing_source == 0,
        missing_question + missing_source,
        "Top cards need both the proof question and the source guidance.",
        "Pull required_question and preferred_source from pm_evidence_source_proof_input.csv.",
    )
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def build_state(cards: pd.DataFrame, ticker_plan: pd.DataFrame, copy_sheet: pd.DataFrame, qa: pd.DataFrame) -> dict[str, Any]:
    first = cards.iloc[0].to_dict() if not cards.empty else {}
    review_rows = int((qa["status"] != "PASS").sum()) if not qa.empty and "status" in qa.columns else 0
    return {
        "date": today_str(),
        "status": "Active",
        "fill_card_count": int(len(cards)),
        "ticker_count": int(len(ticker_plan)),
        "field_to_fill_count": int(len(copy_sheet)),
        "qa_review_count": review_rows,
        "first_ticker": as_text(first.get("ticker"), ""),
        "first_proof_type": as_text(first.get("proof_type"), ""),
        "first_task": as_text(first.get("plain_task"), ""),
        "first_source_to_open": as_text(first.get("source_to_open"), ""),
        "first_fields_to_fill": as_text(first.get("fields_to_fill_now"), ""),
        "plain_answer": (
            f"Proof Fill Desk is active. Start with {as_text(first.get('ticker'), 'no ticker')}. "
            f"{as_text(first.get('plain_task'), 'No proof task yet')} "
            f"There are {len(copy_sheet)} individual fields to fill across {len(cards)} proof rows."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    cards, ticker_plan, recipes, copy_sheet, qa = build_fill_outputs()
    state = build_state(cards, ticker_plan, copy_sheet, qa)

    cards.to_csv(OUT_CARDS, index=False)
    ticker_plan.to_csv(OUT_TICKER_PLAN, index=False)
    recipes.to_csv(OUT_FIELD_RECIPES, index=False)
    copy_sheet.to_csv(OUT_COPY_SHEET, index=False)
    qa.to_csv(OUT_QA, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## First Fill Cards\n\n" + df_to_markdown(cards.head(40)),
        "## Ticker Plan\n\n" + df_to_markdown(ticker_plan.head(40)),
        "## Field Recipes\n\n" + df_to_markdown(recipes),
        "## Copy Sheet\n\n" + df_to_markdown(copy_sheet.head(120)),
        "## QA\n\n" + df_to_markdown(qa),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Proof Fill Desk", sections)
    print(
        "Step213 complete: "
        f"{len(cards)} fill cards, {len(ticker_plan)} tickers, {len(copy_sheet)} fields to fill."
    )


if __name__ == "__main__":
    main()
