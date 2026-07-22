#!/usr/bin/env python3
"""
Canyon v9 Step 210 - Ticker Flow Cards.

Research-only. No broker connection. No live orders.

Step209 creates the state machine. Step210 turns those states into readable
ticker cards for the dashboard. The goal is plain-English PM workflow: answer,
why, what to do now, what not to do, where to click, and which route is allowed.

Outputs:
  quant_fund_ticker_flow_cards_state.json
  quant_fund_ticker_flow_cards.csv
  quant_fund_state_machine_summary.csv
  quant_fund_user_path_cards.csv
  quant_fund_flow_card_quality_check.csv
  quant_fund_ticker_flow_cards_report.md
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


OUT_STATE = ROOT / "quant_fund_ticker_flow_cards_state.json"
OUT_CARDS = ROOT / "quant_fund_ticker_flow_cards.csv"
OUT_SUMMARY = ROOT / "quant_fund_state_machine_summary.csv"
OUT_PATH = ROOT / "quant_fund_user_path_cards.csv"
OUT_QA = ROOT / "quant_fund_flow_card_quality_check.csv"
OUT_REPORT = ROOT / "quant_fund_ticker_flow_cards_report.md"


CARD_COLUMNS = [
    "card_priority",
    "ticker",
    "front_answer",
    "state",
    "what_it_means",
    "why_now",
    "do_now",
    "do_not_do",
    "where_to_click",
    "stock_or_etf_route",
    "option_route",
    "option_side",
    "short_term",
    "medium_term",
    "long_term",
    "trigger_to_watch",
    "proof_to_collect",
    "source_summary",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

SUMMARY_COLUMNS = [
    "state",
    "ticker_count",
    "plain_meaning",
    "allowed_now",
    "forbidden_now",
    "unlock_condition",
    "research_only",
    "no_broker_connection",
    "no_live_orders",
]

PATH_COLUMNS = [
    "step",
    "title",
    "plain_instruction",
    "why_this_step",
    "done_when",
    "do_not_do",
    "page",
    "panel",
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


STATE_RULES = {
    "Needs outside proof": {
        "meaning": "The idea is waiting for a real source, observed value, reviewer, and date.",
        "allowed": "Read and collect proof only.",
        "forbidden": "No new risk, no calls, no puts, no size increase.",
        "unlock": "Fill the proof item, rerun proof acceptance, then rerun the final gate.",
        "priority": 1,
    },
    "Needs PM review": {
        "meaning": "The evidence exists, but a human has not accepted or rejected it yet.",
        "allowed": "Review the evidence packet.",
        "forbidden": "Do not promote the ticker until PM review is finished.",
        "unlock": "Human review accepts the evidence or writes a rejection reason.",
        "priority": 2,
    },
    "Risk blocked": {
        "meaning": "The portfolio cannot afford this idea right now.",
        "allowed": "Risk repair, reduce exposure, or keep on watch.",
        "forbidden": "Do not add exposure just because the story sounds good.",
        "unlock": "VaR, concentration, event gap, and size checks clear after rerun.",
        "priority": 3,
    },
    "Needs execution proof": {
        "meaning": "Spread, liquidity, or fill-risk evidence is missing.",
        "allowed": "Collect quote, spread, volume, and cost proof.",
        "forbidden": "Do not size a ticker with unknown trading cost.",
        "unlock": "Execution cost and liquidity proof are filled and rerun.",
        "priority": 4,
    },
    "Watch only": {
        "meaning": "The idea may matter, but the trigger or confirmation has not happened.",
        "allowed": "Watch price, volume, event proof, and risk changes.",
        "forbidden": "Do not treat watch as buy.",
        "unlock": "Price/volume, event, and risk gates confirm together.",
        "priority": 5,
    },
    "Study only": {
        "meaning": "The ticker is useful for research but not ready for paper action.",
        "allowed": "Read, compare, and collect better evidence.",
        "forbidden": "Do not place it in the paper book.",
        "unlock": "Risk book, proof, and route gates clear.",
        "priority": 6,
    },
    "Tiny paper review allowed": {
        "meaning": "Only a very small paper review may be allowed after manual checks.",
        "allowed": "Tiny paper review only, with manual note and stop rule.",
        "forbidden": "No large paper size and no live order.",
        "unlock": "Manual paper note, risk cap, and monitor rules are filled.",
        "priority": 7,
    },
    "Research candidate": {
        "meaning": "The ticker is a candidate, but final permission still has to confirm.",
        "allowed": "Research and final-gate review.",
        "forbidden": "Do not assume it is approved.",
        "unlock": "Final gate confirms route, risk, execution, and evidence.",
        "priority": 8,
    },
}


RAW_TOKENS = [
    "SIZE_DOWN",
    "DATA_GAP",
    "NO_GO",
    "PENDING_MANUAL_CHECKS",
    "WATCH_EVENT_PROOF_FIRST",
    "WAIT_EXECUTION_OR_MONITOR_REVIEW",
    "NO_NEW_OPTION",
    "NO_DATA",
    "NEEDS_REVIEW",
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


def plain(value: Any) -> str:
    text = as_text(value, "")
    if not text:
        return ""
    replacements = {
        "SIZE_DOWN": "risk says smaller size",
        "DATA_GAP": "missing data",
        "NO_GO": "not allowed yet",
        "BLOCKED": "blocked",
        "PENDING_MANUAL_CHECKS": "needs manual checks",
        "WATCH_EVENT_PROOF_FIRST": "watch until event proof is checked",
        "WAIT_EXECUTION_OR_MONITOR_REVIEW": "wait until execution or monitor risk improves",
        "NO_NEW_OPTION": "no new option",
        "NO_OPTION_WAIT": "no options yet",
        "NO_DATA": "no data",
        "NEEDS_REVIEW": "needs human review",
        "TINY_STOCK_OR_ETF_PAPER_ONLY": "tiny stock or ETF paper review only",
    }
    for raw, friendly in replacements.items():
        text = text.replace(raw, friendly)
    words = []
    for token in text.split():
        clean = token.strip(".,;:()")
        if "_" in clean and clean.upper() == clean:
            words.append(token.replace(clean, clean.replace("_", " ").title()))
        else:
            words.append(token)
    text = " ".join(words)
    return " ".join(text.split())


def ticker_blockers(blockers: pd.DataFrame) -> dict[str, list[pd.Series]]:
    out: dict[str, list[pd.Series]] = {}
    if blockers.empty or "ticker" not in blockers.columns:
        return out
    work = blockers.copy()
    work["ticker"] = work["ticker"].apply(clean_ticker)
    for _, row in work.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        out.setdefault(ticker, []).append(row)
    return out


def source_summary(value: Any) -> str:
    text = as_text(value, "")
    if not text:
        return "Local Canyon files"
    pieces = []
    for part in text.replace("|", ";").split(";"):
        name = part.strip().split("/")[-1]
        if not name:
            continue
        name = name.replace(".csv", "").replace(".json", "").replace(".md", "").replace("_", " ")
        name = name[:1].upper() + name[1:]
        if name not in pieces:
            pieces.append(name)
    return ", ".join(pieces[:4]) + ("..." if len(pieces) > 4 else "")


def build_cards() -> pd.DataFrame:
    current = read_csv_safe(ROOT / "quant_fund_flow_current_state.csv")
    blockers = read_csv_safe(ROOT / "quant_fund_flow_blocker_queue.csv")
    blocker_map = ticker_blockers(blockers)
    rows: list[dict[str, Any]] = []

    if current.empty:
        return pd.DataFrame(columns=CARD_COLUMNS)

    for _, row in current.iterrows():
        ticker = clean_ticker(row.get("ticker"))
        if not ticker:
            continue
        state = as_text(row.get("current_state"), "Research candidate")
        rule = STATE_RULES.get(state, STATE_RULES["Research candidate"])
        top_blockers = blocker_map.get(ticker, [])[:3]

        blocker_text = "; ".join(short(b.get("blocker"), 120) for b in top_blockers if as_text(b.get("blocker"), ""))
        do_text = as_text(row.get("next_action"), "")
        if top_blockers:
            do_text = as_text(top_blockers[0].get("what_to_do"), do_text)
        if not do_text:
            do_text = rule["unlock"]

        where = as_text(row.get("next_click"), "")
        if top_blockers:
            where = as_text(top_blockers[0].get("where_to_click"), where)
        if not where:
            where = "Home"

        proof = as_text(row.get("proof_needed"), "")
        if not proof and top_blockers:
            proof = as_text(top_blockers[0].get("blocker"), "")

        option_route = plain(row.get("option_route")) or "No option route yet"
        option_side = plain(row.get("option_side")) or "None"
        stock_route = plain(row.get("stock_or_etf_route")) or "No new exposure"

        if "No new risk" in as_text(row.get("can_take_new_risk")):
            front = f"{ticker}: do not add risk yet. Start with {where}."
        elif state == "Tiny paper review allowed":
            front = f"{ticker}: tiny paper review only after manual checks."
        else:
            front = f"{ticker}: research first. Confirm the final gate before action."

        why_parts = []
        if blocker_text:
            why_parts.append(f"Main blocker: {plain(blocker_text)}.")
        why = as_text(row.get("why"), "")
        if why:
            why_parts.append(plain(why))
        news = as_text(row.get("news_read"), "")
        if news:
            why_parts.append(f"News hook: {news}.")

        rows.append(guard_flags({
            "card_priority": int(rule["priority"]),
            "ticker": ticker,
            "front_answer": short(front, 180),
            "state": state,
            "what_it_means": rule["meaning"],
            "why_now": short(" ".join(why_parts), 420),
            "do_now": short(plain(do_text), 260),
            "do_not_do": rule["forbidden"],
            "where_to_click": where,
            "stock_or_etf_route": short(stock_route, 160),
            "option_route": short(option_route, 180),
            "option_side": option_side,
            "short_term": plain(row.get("short_term_route")) or "Research only",
            "medium_term": plain(row.get("medium_term_route")) or "Research only",
            "long_term": plain(row.get("long_term_route")) or "Research only",
            "trigger_to_watch": short(row.get("trigger_to_watch"), 180),
            "proof_to_collect": short(proof, 260),
            "source_summary": source_summary(row.get("source_trail")),
        }))

    cards = pd.DataFrame(rows, columns=CARD_COLUMNS)
    if cards.empty:
        return cards
    return cards.sort_values(["card_priority", "ticker"]).reset_index(drop=True)


def build_summary(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counts = cards["state"].value_counts().to_dict() if not cards.empty else {}
    for state, rule in sorted(STATE_RULES.items(), key=lambda item: item[1]["priority"]):
        rows.append(guard_flags({
            "state": state,
            "ticker_count": int(counts.get(state, 0)),
            "plain_meaning": rule["meaning"],
            "allowed_now": rule["allowed"],
            "forbidden_now": rule["forbidden"],
            "unlock_condition": rule["unlock"],
        }))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_path_cards() -> pd.DataFrame:
    next_clicks = read_csv_safe(ROOT / "quant_fund_flow_next_clicks.csv")
    rows = []
    if next_clicks.empty:
        return pd.DataFrame(columns=PATH_COLUMNS)
    for _, row in next_clicks.iterrows():
        rows.append(guard_flags({
            "step": int(float(row.get("order", 0) or 0)),
            "title": f"{as_text(row.get('page'), 'Home')} - {as_text(row.get('panel'), 'Read first')}",
            "plain_instruction": short(row.get("what_to_read"), 240),
            "why_this_step": short(row.get("why_now"), 260),
            "done_when": short(row.get("done_when"), 240),
            "do_not_do": short(row.get("do_not_do"), 240),
            "page": as_text(row.get("page"), "Home"),
            "panel": as_text(row.get("panel"), ""),
        }))
    return pd.DataFrame(rows, columns=PATH_COLUMNS)


def build_quality(cards: pd.DataFrame, path_cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = [
        (
            "Every ticker card has an answer",
            int(cards["front_answer"].astype(str).str.strip().eq("").sum()) if not cards.empty else 0,
            "Cards should start with a clear answer.",
            "Fill front_answer from current state and next click.",
        ),
        (
            "Every ticker card has a next action",
            int(cards["do_now"].astype(str).str.strip().eq("").sum()) if not cards.empty else 0,
            "Cards need one concrete next step.",
            "Fill do_now from blocker queue or state unlock condition.",
        ),
        (
            "Every ticker card says what not to do",
            int(cards["do_not_do"].astype(str).str.strip().eq("").sum()) if not cards.empty else 0,
            "The page must prevent accidental interpretation as a trade signal.",
            "Fill forbidden action from state machine rules.",
        ),
        (
            "No obvious raw status tokens in card text",
            count_raw_tokens(cards),
            "Cards should read like English, not internal code.",
            "Add a replacement in plain().",
        ),
        (
            "User path has steps",
            0 if not path_cards.empty else 1,
            "The user needs a click order.",
            "Regenerate quant_fund_flow_next_clicks.csv first.",
        ),
    ]
    for name, bad_rows, checked, hint in checks:
        rows.append(guard_flags({
            "check": name,
            "status": "PASS" if bad_rows == 0 else "REVIEW",
            "bad_rows": int(bad_rows),
            "what_it_checked": checked,
            "fix_hint": hint,
        }))
    return pd.DataFrame(rows, columns=QA_COLUMNS)


def count_raw_tokens(cards: pd.DataFrame) -> int:
    if cards.empty:
        return 0
    cols = ["front_answer", "what_it_means", "why_now", "do_now", "do_not_do", "stock_or_etf_route", "option_route"]
    count = 0
    for _, row in cards.iterrows():
        text = " ".join(as_text(row.get(c), "") for c in cols)
        if any(tok in text for tok in RAW_TOKENS):
            count += 1
    return count


def build_state(cards: pd.DataFrame, summary: pd.DataFrame, qa: pd.DataFrame) -> dict[str, Any]:
    top = cards.iloc[0].to_dict() if not cards.empty else {}
    review_count = int((qa["status"] != "PASS").sum()) if not qa.empty and "status" in qa.columns else 0
    state_counts = cards["state"].value_counts().to_dict() if not cards.empty else {}
    first_answer = as_text(top.get("front_answer"), "Run the flow navigator first.")
    return {
        "date": today_str(),
        "status": "Active",
        "card_count": int(len(cards)),
        "state_count": int(len(summary)),
        "quality_review_count": review_count,
        "top_ticker": as_text(top.get("ticker"), ""),
        "top_state": as_text(top.get("state"), ""),
        "first_answer": first_answer,
        "needs_outside_proof_count": int(state_counts.get("Needs outside proof", 0)),
        "risk_blocked_count": int(state_counts.get("Risk blocked", 0)),
        "plain_answer": (
            f"Ticker flow cards are active. Start with {first_answer} "
            f"{len(cards)} tickers have readable cards."
        ),
        "research_only": True,
        "no_broker_connection": True,
        "no_live_orders": True,
    }


def main() -> None:
    cards = build_cards()
    summary = build_summary(cards)
    path_cards = build_path_cards()
    qa = build_quality(cards, path_cards)
    state = build_state(cards, summary, qa)

    cards.to_csv(OUT_CARDS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    path_cards.to_csv(OUT_PATH, index=False)
    qa.to_csv(OUT_QA, index=False)
    write_json(OUT_STATE, state)

    sections = [
        "Research-only. No broker connection. No live orders.",
        "## Plain Answer\n\n" + state["plain_answer"],
        "## Ticker Flow Cards\n\n" + df_to_markdown(cards.head(80)),
        "## State Machine Summary\n\n" + df_to_markdown(summary),
        "## User Path Cards\n\n" + df_to_markdown(path_cards),
        "## Quality Check\n\n" + df_to_markdown(qa),
    ]
    write_markdown_report(OUT_REPORT, "Canyon v9 Ticker Flow Cards", sections)
    print(
        "Step210 complete: "
        f"{len(cards)} cards, {len(summary)} state rows, {len(path_cards)} path cards, "
        f"{len(qa)} QA checks."
    )


if __name__ == "__main__":
    main()
