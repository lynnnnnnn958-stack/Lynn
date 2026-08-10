#!/usr/bin/env python3
"""
step_sp500_pit_membership.py — reconstruct point-in-time S&P 500 membership
===========================================================================
Removes survivorship bias from the backtest. Using the CURRENT constituent list
historically inflates returns (it silently excludes companies that WERE in the
index but got removed/delisted — often the losers). This reconstructs approximate
month-end membership back to 2010 by starting from today's list and walking the
Wikipedia add/remove change log backwards.

Output: sp500_pit_membership.csv  (long: date, ticker)  — month-end snapshots
        used by step_rigorous_backtest.py to restrict each rebalance to names
        that were actually in the index at that time.

Caveat: Wikipedia's change log is good but not a vendor-grade PIT feed; treat as
a strong mitigation, not perfect. Coverage thins pre-2010.
"""
from __future__ import annotations
import ssl, urllib.request, io
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START = pd.Timestamp("2010-01-01")


def _fetch():
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, context=ctx, timeout=30).read().decode("utf-8")
    tabs = pd.read_html(io.StringIO(html))
    current = tabs[0]
    changes = tabs[1]
    changes.columns = ['_'.join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                       for c in changes.columns]
    return current, changes


def reconstruct() -> pd.DataFrame:
    current, changes = _fetch()
    sym_col = "Symbol" if "Symbol" in current.columns else current.columns[0]
    members_now = set(current[sym_col].astype(str).str.replace(".", "-", regex=False))

    ch = changes.rename(columns={
        "Effective Date_Effective Date": "date",
        "Added_Ticker": "added", "Removed_Ticker": "removed"})
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date"]).sort_values("date")

    # month-end snapshots from today back to START
    month_ends = pd.date_range(START, pd.Timestamp.today(), freq="ME")
    snapshots = {}
    members = set(members_now)
    # walk from newest month back to oldest, reversing changes as we cross them
    prev = pd.Timestamp.today()
    for me in reversed(month_ends):
        # reverse all changes effective in (me, prev]
        window = ch[(ch["date"] > me) & (ch["date"] <= prev)]
        for _, r in window.iterrows():
            add = str(r.get("added", "")).replace(".", "-")
            rem = str(r.get("removed", "")).replace(".", "-")
            if add and add != "nan":
                members.discard(add)     # it was added after `me` → not a member at `me`
            if rem and rem != "nan":
                members.add(rem)         # it was removed after `me` → still a member at `me`
        snapshots[me] = set(members)
        prev = me

    rows = []
    for me, mem in snapshots.items():
        for t in mem:
            if t and t != "nan":
                rows.append({"date": me.strftime("%Y-%m-%d"), "ticker": t})
    return pd.DataFrame(rows)


def main():
    print("Reconstructing point-in-time S&P 500 membership …")
    try:
        df = reconstruct()
    except Exception as e:
        print(f"  fetch/reconstruct failed ({e}) — skipping (backtest falls back to current universe)")
        return
    df.to_csv(ROOT / "sp500_pit_membership.csv", index=False)
    n_months = df["date"].nunique()
    avg = df.groupby("date").size().mean()
    print(f"✓ sp500_pit_membership.csv: {n_months} month-ends, avg {avg:.0f} members/month "
          f"({df['date'].min()} → {df['date'].max()})")


if __name__ == "__main__":
    main()
