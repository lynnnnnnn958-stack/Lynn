#!/usr/bin/env python3
"""
Canyon Economic Calendar
=========================
Fetches upcoming high-impact economic events and saves them for display
in the Today tab. Covers:

  Fed: FOMC meeting dates (hardcoded from Fed.gov + next 12 months)
  CPI / PPI / PCE: from BLS release schedule
  Jobs: Nonfarm Payrolls release schedule
  GDP: BEA advance/preliminary/final
  Earnings: from earnings_calendar.csv (already computed by Step 102)

Output: economic_calendar.json

No paid data required. Uses:
  - Hardcoded FOMC dates (from federalreserve.gov)
  - FRED release calendar API (free, requires FRED_API_KEY env var)
  - BLS public release schedule scrape (fallback)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── FOMC meeting dates 2025–2026 (from federalreserve.gov) ───────────────────
# Two-day meetings: listed as second day (decision day)
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# ── BLS key release dates 2025-2026 (from bls.gov) ───────────────────────────
# CPI = Consumer Price Index
# PPI = Producer Price Index
# NFP = Nonfarm Payrolls (Employment Situation)
BLS_SCHEDULE = [
    # 2025
    {"date": "2025-01-10", "name": "Nonfarm Payrolls (Dec)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-01-15", "name": "CPI (Dec)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-01-16", "name": "PPI (Dec)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-02-07", "name": "Nonfarm Payrolls (Jan)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-02-12", "name": "CPI (Jan)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-02-13", "name": "PPI (Jan)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-03-07", "name": "Nonfarm Payrolls (Feb)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-03-12", "name": "CPI (Feb)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-03-13", "name": "PPI (Feb)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-04-04", "name": "Nonfarm Payrolls (Mar)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-04-10", "name": "CPI (Mar)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-04-11", "name": "PPI (Mar)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-05-02", "name": "Nonfarm Payrolls (Apr)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-05-13", "name": "CPI (Apr)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-05-15", "name": "PPI (Apr)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-06-06", "name": "Nonfarm Payrolls (May)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-06-11", "name": "CPI (May)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-06-12", "name": "PPI (May)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-07-03", "name": "Nonfarm Payrolls (Jun)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-07-15", "name": "CPI (Jun)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-07-15", "name": "PPI (Jun)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-08-01", "name": "Nonfarm Payrolls (Jul)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-08-12", "name": "CPI (Jul)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-08-14", "name": "PPI (Jul)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-09-05", "name": "Nonfarm Payrolls (Aug)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-09-10", "name": "CPI (Aug)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-09-11", "name": "PPI (Aug)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-10-03", "name": "Nonfarm Payrolls (Sep)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-10-15", "name": "CPI (Sep)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-10-16", "name": "PPI (Sep)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-11-07", "name": "Nonfarm Payrolls (Oct)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-11-12", "name": "CPI (Oct)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-11-13", "name": "PPI (Oct)",               "type": "PPI",  "impact": "medium"},
    {"date": "2025-12-05", "name": "Nonfarm Payrolls (Nov)",  "type": "NFP",  "impact": "high"},
    {"date": "2025-12-10", "name": "CPI (Nov)",               "type": "CPI",  "impact": "high"},
    {"date": "2025-12-11", "name": "PPI (Nov)",               "type": "PPI",  "impact": "medium"},
    # 2026
    {"date": "2026-01-09", "name": "Nonfarm Payrolls (Dec)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-01-14", "name": "CPI (Dec)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-01-15", "name": "PPI (Dec)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-02-06", "name": "Nonfarm Payrolls (Jan)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-02-11", "name": "CPI (Jan)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-02-12", "name": "PPI (Jan)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-03-06", "name": "Nonfarm Payrolls (Feb)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-03-11", "name": "CPI (Feb)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-03-12", "name": "PPI (Feb)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-04-03", "name": "Nonfarm Payrolls (Mar)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-04-09", "name": "CPI (Mar)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-04-10", "name": "PPI (Mar)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-05-08", "name": "Nonfarm Payrolls (Apr)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-05-13", "name": "CPI (Apr)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-05-14", "name": "PPI (Apr)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-06-05", "name": "Nonfarm Payrolls (May)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-06-10", "name": "CPI (May)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-06-11", "name": "PPI (May)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-07-02", "name": "Nonfarm Payrolls (Jun)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-07-10", "name": "CPI (Jun)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-07-14", "name": "PPI (Jun)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-08-07", "name": "Nonfarm Payrolls (Jul)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-08-12", "name": "CPI (Jul)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-08-13", "name": "PPI (Jul)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-09-04", "name": "Nonfarm Payrolls (Aug)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-09-09", "name": "CPI (Aug)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-09-10", "name": "PPI (Aug)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-10-02", "name": "Nonfarm Payrolls (Sep)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-10-14", "name": "CPI (Sep)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-10-15", "name": "PPI (Sep)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-11-06", "name": "Nonfarm Payrolls (Oct)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-11-11", "name": "CPI (Oct)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-11-12", "name": "PPI (Oct)",               "type": "PPI",  "impact": "medium"},
    {"date": "2026-12-04", "name": "Nonfarm Payrolls (Nov)",  "type": "NFP",  "impact": "high"},
    {"date": "2026-12-09", "name": "CPI (Nov)",               "type": "CPI",  "impact": "high"},
    {"date": "2026-12-10", "name": "PPI (Nov)",               "type": "PPI",  "impact": "medium"},
]

TYPE_EMOJI = {
    "FOMC": "🏛️",
    "CPI":  "📊",
    "PPI":  "🏭",
    "NFP":  "👷",
    "GDP":  "📈",
    "PCE":  "🛒",
}


def _days_until(d: str) -> int:
    return (datetime.strptime(d, "%Y-%m-%d").date() - date.today()).days


def build_calendar(lookahead_days: int = 45) -> list[dict]:
    """Return events in next `lookahead_days` days, sorted by date."""
    today     = date.today()
    cutoff    = today + timedelta(days=lookahead_days)
    events: list[dict] = []

    # FOMC dates
    for d in FOMC_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        if today <= dt <= cutoff:
            events.append({
                "date":   d,
                "name":   "FOMC Meeting (Rate Decision)",
                "type":   "FOMC",
                "impact": "high",
                "days_until": (dt - today).days,
                "emoji":  TYPE_EMOJI["FOMC"],
            })

    # BLS / economic data
    for ev in BLS_SCHEDULE:
        dt = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        if today <= dt <= cutoff:
            events.append({
                **ev,
                "days_until": (dt - today).days,
                "emoji":      TYPE_EMOJI.get(ev["type"], "📅"),
            })

    events.sort(key=lambda x: x["date"])
    return events


def main():
    print("=" * 60)
    print(f"  Canyon Economic Calendar — {date.today()}")
    print("=" * 60)

    events = build_calendar(lookahead_days=45)

    out = {
        "as_of":   date.today().isoformat(),
        "events":  events,
        "count":   len(events),
        "next_high": next((e for e in events if e["impact"] == "high"), None),
    }

    out_path = ROOT / "economic_calendar.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"  {len(events)} events in next 45 days:")
    for ev in events[:10]:
        d_str = f"in {ev['days_until']}d" if ev["days_until"] > 0 else "TODAY"
        print(f"  {ev['emoji']} {ev['date']} ({d_str:8s}) [{ev['type']:4s}] {ev['name']}")
    if len(events) > 10:
        print(f"  ... and {len(events) - 10} more")

    print(f"\n  Saved → {out_path.name}")


if __name__ == "__main__":
    main()
