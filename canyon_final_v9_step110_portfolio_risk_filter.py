#!/usr/bin/env python3
"""
Canyon v9 — Step 110: Portfolio Risk Filter
============================================
Three-stage portfolio risk management engine running AFTER Step 87 (alpha
aggregator) and BEFORE position sizing.  No live orders.

Stage 1 — Correlation Deduplication
-------------------------------------
For each pair of picks with 60-day rolling correlation > 0.75, keep the
higher-alpha name and replace the lower with the next eligible stock from
alpha_scores.csv (must be outside the existing picks set and below the
correlation threshold with all retained picks).

  Rationale: two highly correlated positions provide almost no diversification
  but consume full capital.  A 0.75 threshold allows some overlap (sector
  peers) while eliminating near-duplicate exposures (e.g. MSFT + GOOGL when
  both are in the same bull-tech regime).

Stage 2 — Sector Concentration Limits
---------------------------------------
Hard cap: no single GICS sector may represent > 35% of the total portfolio
weight.  If exceeded, trim the lowest-alpha picks in the over-weight sector
until the limit is satisfied.  Trimmed tickers are added to a "sector_trim"
list in the output.

  Rationale: sector concentration is the #1 source of non-systematic drawdown
  in single-stock portfolios.  A 35% cap still allows meaningful overweights
  while capping catastrophic sector blow-ups.

Stage 3 — Portfolio Drawdown Circuit Breaker
---------------------------------------------
Reads portfolio_nav.csv (date, nav columns).  Computes drawdown from the
rolling 252-day high-water mark (HWM).

  Drawdown > 15%  →  exposure_multiplier = 0.20  (emergency de-risk)
  Drawdown > 10%  →  exposure_multiplier = 0.50  (defensive mode)
  Otherwise       →  exposure_multiplier = 1.00  (normal)

Writes exposure_override.json.  Step 87 reads this on each run and scales
POSITION_CAP accordingly.  Circuit breaker resets automatically once NAV
recovers above the HWM × 0.95 threshold.

Inputs
------
  daily_picks.csv          — Step 87 output (ticker, weight_pct, alpha_score …)
  alpha_scores.csv         — full-universe scored list (for replacements)
  sp500_price_cache.csv    — 60-day prices (for correlation matrix)
  sector_map.csv           — ticker → GICS sector
  portfolio_nav.csv        — date, nav (written by this script + paper ledger)

Outputs
-------
  daily_picks_filtered.csv — filtered / adjusted picks (same schema as input)
  exposure_override.json   — {exposure_multiplier, drawdown_pct, hwm, reason, date}
  portfolio_risk_report.md — human-readable audit trail

Usage
-----
  python3 canyon_final_v9_step110_portfolio_risk_filter.py
  python3 canyon_final_v9_step110_portfolio_risk_filter.py --sector-cap 0.30
  python3 canyon_final_v9_step110_portfolio_risk_filter.py --corr-threshold 0.80
  python3 canyon_final_v9_step110_portfolio_risk_filter.py --no-circuit-breaker
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Defaults ──────────────────────────────────────────────────────────────────
SECTOR_CAP          = 0.35   # max fraction of portfolio in one GICS sector
CORR_THRESHOLD      = 0.75   # pairwise correlation threshold for deduplication
CORR_LOOKBACK_DAYS  = 60     # trading days for correlation window
DD_SOFT_THRESHOLD   = 0.10   # 10% drawdown → 50% exposure
DD_HARD_THRESHOLD   = 0.15   # 15% drawdown → 20% exposure
DD_RESET_BUFFER     = 0.05   # allow exposure back once DD < hard − 5%

# File paths
PICKS_CSV     = ROOT / "daily_picks.csv"
ALPHA_CSV     = ROOT / "alpha_scores.csv"
PRICE_CSV     = ROOT / "sp500_price_cache.csv"
SECTOR_CSV    = ROOT / "sector_map.csv"
NAV_CSV       = ROOT / "portfolio_nav.csv"
OUT_PICKS     = ROOT / "daily_picks_filtered.csv"
OUT_OVERRIDE  = ROOT / "exposure_override.json"
OUT_REPORT    = ROOT / "portfolio_risk_report.md"


# =============================================================================
# 1.  Data loaders
# =============================================================================

def load_picks() -> pd.DataFrame:
    """Load current daily_picks.csv.  Returns empty DataFrame on failure."""
    if not PICKS_CSV.exists():
        print("  [WARN] daily_picks.csv not found — run step87 first")
        return pd.DataFrame()
    df = pd.read_csv(PICKS_CSV)
    if "ticker" not in df.columns:
        print("  [WARN] daily_picks.csv missing 'ticker' column")
        return pd.DataFrame()
    return df


def load_alpha_scores() -> pd.DataFrame:
    """Load full-universe alpha_scores.csv for replacement candidates."""
    if not ALPHA_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALPHA_CSV)
    if {"ticker", "alpha_score"}.issubset(df.columns):
        return df.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def load_prices(tickers: list[str], lookback: int = CORR_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Return log-return matrix (dates × tickers) over `lookback` trading days.
    Falls back to yfinance if cache is missing.
    """
    for price_path in (ROOT / "sp500_price_cache.csv",
                       ROOT / "backtest_price_cache.csv"):
        if not price_path.exists():
            continue
        try:
            prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
            prices = prices.sort_index()
            available = [t for t in tickers if t in prices.columns]
            if not available:
                continue
            px = prices[available].tail(lookback + 5).ffill().bfill()
            rets = np.log(px / px.shift(1)).dropna()
            if len(rets) >= 20:
                return rets
        except Exception as e:
            print(f"  [price] {price_path.name}: {e}")
            continue

    # yfinance fallback
    try:
        import yfinance as yf
        end   = datetime.now()
        start = end - timedelta(days=int(lookback * 1.6))
        raw   = yf.download(
            tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True
        )
        px = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(px, pd.Series):
            px = px.to_frame(tickers[0])
        rets = np.log(px / px.shift(1)).dropna()
        print(f"  [price] yfinance fallback: {rets.shape[1]} tickers × {len(rets)} days")
        return rets
    except Exception as e:
        print(f"  [price] yfinance fallback failed: {e}")
        return pd.DataFrame()


def load_sector_map() -> dict[str, str]:
    """Return {ticker: sector} from sector_map.csv."""
    if not SECTOR_CSV.exists():
        return {}
    try:
        df = pd.read_csv(SECTOR_CSV)
        if {"ticker", "sector"}.issubset(df.columns):
            return df.set_index("ticker")["sector"].to_dict()
    except Exception:
        pass
    return {}


def load_portfolio_nav() -> pd.DataFrame:
    """
    Load portfolio_nav.csv (columns: date, nav).
    Creates an empty DataFrame if file doesn't exist.
    """
    if not NAV_CSV.exists():
        return pd.DataFrame(columns=["date", "nav"])
    try:
        df = pd.read_csv(NAV_CSV, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "nav"])


# =============================================================================
# 2.  Correlation filter
# =============================================================================

def build_corr_matrix(picks_tickers: list[str]) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation matrix from 60-day log-returns.
    Returns empty DataFrame if prices unavailable.
    """
    rets = load_prices(picks_tickers)
    if rets.empty:
        return pd.DataFrame()
    common = [t for t in picks_tickers if t in rets.columns]
    if len(common) < 2:
        return pd.DataFrame()
    return rets[common].corr()


def correlation_filter(
    picks: pd.DataFrame,
    alpha_all: pd.DataFrame,
    corr_thresh: float = CORR_THRESHOLD,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Stage 1: Remove correlated duplicate picks and replace with uncorrelated
    alternatives from the full alpha universe.

    Returns
    -------
    filtered_picks : DataFrame  (same columns as input picks)
    audit_log      : list of dicts describing each swap
    """
    audit: list[dict] = []

    if picks.empty or len(picks) < 2:
        return picks.copy(), audit

    tickers = picks["ticker"].tolist()
    corr_df = build_corr_matrix(tickers)

    if corr_df.empty:
        print("  [corr] price data unavailable — correlation filter skipped")
        return picks.copy(), audit

    # Build set of correlated pairs (upper triangle only)
    high_corr_pairs: list[tuple[str, str, float]] = []
    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            ti = corr_df.columns[i]
            tj = corr_df.columns[j]
            if ti not in corr_df.index or tj not in corr_df.columns:
                continue
            c = corr_df.loc[ti, tj]
            if not np.isnan(c) and abs(c) > corr_thresh:
                high_corr_pairs.append((ti, tj, c))

    if not high_corr_pairs:
        print(f"  [corr] no pairs above threshold {corr_thresh:.2f} — all picks retained")
        return picks.copy(), audit

    # Score map for fast lookup
    score_map: dict[str, float] = dict(
        zip(picks["ticker"], picks["alpha_score"])
    )

    # Tickers already confirmed for removal
    to_remove: set[str] = set()
    retained:  set[str] = set(tickers)

    for ti, tj, c in sorted(high_corr_pairs, key=lambda x: -x[2]):
        if ti in to_remove or tj in to_remove:
            continue
        # Remove the lower-alpha one
        if score_map.get(ti, 0) >= score_map.get(tj, 0):
            loser = tj
        else:
            loser = ti
        to_remove.add(loser)
        retained.discard(loser)
        audit.append({
            "action":     "CORR_REMOVE",
            "removed":    loser,
            "reason":     f"corr={c:.3f} with {ti if loser == tj else tj}",
            "alpha_removed": score_map.get(loser, 0.0),
        })

    if not to_remove:
        return picks.copy(), audit

    # Find replacements from full alpha universe
    existing_set = set(picks["ticker"])
    candidates   = (
        alpha_all[~alpha_all["ticker"].isin(existing_set)]
        .reset_index(drop=True)
    ) if not alpha_all.empty else pd.DataFrame()

    replacements: list[dict] = []
    for removed_ticker in list(to_remove):
        added = False
        if not candidates.empty:
            # Try each candidate in alpha rank order
            for _, cand_row in candidates.iterrows():
                cand = cand_row["ticker"]
                # Check corr against all retained tickers
                test_set = list(retained) + [cand]
                test_rets = load_prices(test_set)
                ok = True
                if not test_rets.empty and cand in test_rets.columns:
                    for r in retained:
                        if r in test_rets.columns:
                            pair_c = test_rets[[r, cand]].corr().iloc[0, 1]
                            if not np.isnan(pair_c) and abs(pair_c) > corr_thresh:
                                ok = False
                                break
                if ok:
                    retained.add(cand)
                    # Remove from candidates so it isn't reused
                    candidates = candidates[candidates["ticker"] != cand]
                    replacements.append({
                        "action":      "CORR_ADD",
                        "added":       cand,
                        "replaced":    removed_ticker,
                        "alpha_added": float(cand_row["alpha_score"]),
                    })
                    audit.append(replacements[-1])
                    added = True
                    break
        if not added:
            audit.append({
                "action":    "CORR_NO_REPLACEMENT",
                "removed":   removed_ticker,
                "reason":    "no uncorrelated candidate found — position count reduced",
            })

    # Rebuild picks DataFrame
    keep_tickers = list(retained)
    base = picks[picks["ticker"].isin(retained)].copy()

    # Add replacement rows from alpha_all
    replacement_tickers = [r["added"] for r in replacements if "added" in r]
    if replacement_tickers and not alpha_all.empty:
        new_rows = alpha_all[alpha_all["ticker"].isin(replacement_tickers)].copy()
        # Carry over columns that exist in picks
        for col in picks.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        new_rows = new_rows[picks.columns]
        base = pd.concat([base, new_rows], ignore_index=True)

    # Recalculate weights proportional to alpha_score with POSITION_CAP
    if "alpha_score" in base.columns and base["alpha_score"].sum() > 0:
        scores = base["alpha_score"].clip(0, 100).values
        raw_w  = scores / scores.sum()
        from_picks = picks.get("weight_pct", pd.Series(dtype=float))
        pos_cap = float(from_picks.max() / 100.0) if not from_picks.empty else 0.15
        raw_w  = np.clip(raw_w, 0, pos_cap)
        raw_w /= raw_w.sum()
        base["weight_pct"] = (raw_w * 100).round(2)

    base = base.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    return base, audit


# =============================================================================
# 3.  Sector concentration filter
# =============================================================================

def sector_concentration_filter(
    picks: pd.DataFrame,
    sector_map: dict[str, str],
    sector_cap: float = SECTOR_CAP,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Stage 2: Enforce sector concentration limit.
    Trims lowest-alpha tickers in over-weighted sectors until all sectors
    are within the cap.

    Returns
    -------
    filtered : DataFrame
    audit    : list of dicts
    """
    audit: list[dict] = []
    if picks.empty:
        return picks.copy(), audit

    df = picks.copy()

    # Assign sector to each pick
    df["_sector"] = df["ticker"].map(sector_map).fillna("Unknown")

    # Normalize weights to sum to 1.0
    total_w = df["weight_pct"].sum()
    if total_w <= 0:
        return picks.copy(), audit
    df["_w"] = df["weight_pct"] / total_w

    max_iter = 20  # safety limit
    iteration = 0
    while iteration < max_iter:
        sector_totals = df.groupby("_sector")["_w"].sum()
        over_sectors  = sector_totals[sector_totals > sector_cap]
        if over_sectors.empty:
            break
        # Process most-over-concentrated sector first
        worst_sector = over_sectors.idxmax()
        over_excess  = float(over_sectors[worst_sector]) - sector_cap

        # Find the lowest-alpha ticker in that sector
        in_sector = df[df["_sector"] == worst_sector].sort_values(
            "alpha_score", ascending=True
        )
        if in_sector.empty:
            break
        trim_ticker = in_sector.iloc[0]["ticker"]
        trim_w      = float(in_sector.iloc[0]["_w"])
        trim_alpha  = float(in_sector.iloc[0].get("alpha_score", 0.0))

        # Remove the ticker
        df = df[df["ticker"] != trim_ticker].copy()

        # Re-normalize
        new_total = df["_w"].sum()
        if new_total > 0:
            df["_w"] = df["_w"] / new_total
        else:
            break

        audit.append({
            "action":       "SECTOR_TRIM",
            "removed":      trim_ticker,
            "sector":       worst_sector,
            "sector_w_was": round(float(over_sectors[worst_sector]) * 100, 1),
            "alpha_removed": round(trim_alpha, 2),
            "reason":       f"{worst_sector} was {over_sectors[worst_sector]*100:.1f}% > {sector_cap*100:.0f}% cap",
        })
        iteration += 1

    # Update weight_pct from final normalized weights
    df["weight_pct"] = (df["_w"] * 100).round(2)
    df = df.drop(columns=["_w", "_sector"], errors="ignore")
    df = df.sort_values("alpha_score", ascending=False).reset_index(drop=True)

    if audit:
        print(f"  [sector] trimmed {len(audit)} tickers to enforce {sector_cap*100:.0f}% cap")
    else:
        sector_max = picks["ticker"].map(sector_map).fillna("Unknown")
        # Recalculate for reporting
        sector_weights = {}
        total = picks["weight_pct"].sum() or 1.0
        for _, row in picks.iterrows():
            sec = sector_map.get(row["ticker"], "Unknown")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + row["weight_pct"] / total
        max_sec = max(sector_weights, key=lambda s: sector_weights[s]) if sector_weights else "?"
        max_pct = sector_weights.get(max_sec, 0.0) * 100
        print(f"  [sector] all sectors within {sector_cap*100:.0f}% cap  "
              f"(max: {max_sec} {max_pct:.1f}%)")

    return df, audit


# =============================================================================
# 4.  Portfolio drawdown circuit breaker
# =============================================================================

def estimate_nav_from_picks(picks: pd.DataFrame) -> Optional[float]:
    """
    Estimate today's portfolio NAV from price changes.
    Uses equal weight if weight_pct not available.
    Returns None if prices not available.
    """
    if picks.empty:
        return None
    tickers = picks["ticker"].tolist()
    try:
        rets = load_prices(tickers, lookback=5)
        if rets.empty:
            return None
        # Last day's return, portfolio-weighted
        today_rets = rets.iloc[-1] if len(rets) > 0 else pd.Series(dtype=float)
        if today_rets.empty:
            return None
        weights = (
            picks.set_index("ticker")["weight_pct"]
            if "weight_pct" in picks.columns else pd.Series(1.0, index=tickers)
        )
        weights = weights / (weights.sum() or 1.0)
        common  = [t for t in tickers if t in today_rets.index]
        if not common:
            return None
        port_ret = sum(today_rets[t] * weights.get(t, 0.0) for t in common)
        return float(np.exp(port_ret))  # as a multiplier vs previous day
    except Exception:
        return None


def load_or_create_nav(picks: pd.DataFrame) -> pd.DataFrame:
    """
    Load portfolio_nav.csv.  If it doesn't exist or is empty, create a
    bootstrap row using today = 100.0 (normalised NAV).
    """
    nav_df = load_portfolio_nav()
    today  = datetime.now().strftime("%Y-%m-%d")

    if nav_df.empty:
        # Seed with 100.0
        nav_df = pd.DataFrame([{"date": pd.Timestamp(today), "nav": 100.0}])
        nav_df.to_csv(NAV_CSV, index=False)
        print(f"  [nav] portfolio_nav.csv created — seeded with NAV=100 on {today}")
        return nav_df

    # Check if today is already logged
    today_ts = pd.Timestamp(today)
    if today_ts in nav_df["date"].values:
        return nav_df

    # Estimate today's NAV using price change multiplier
    prev_nav = float(nav_df.iloc[-1]["nav"])
    multiplier = estimate_nav_from_picks(picks)
    if multiplier is not None and 0.80 <= multiplier <= 1.20:
        today_nav = round(prev_nav * multiplier, 4)
    else:
        today_nav = prev_nav  # carry-forward if prices unavailable

    new_row = pd.DataFrame([{"date": today_ts, "nav": today_nav}])
    nav_df  = pd.concat([nav_df, new_row], ignore_index=True)
    nav_df.to_csv(NAV_CSV, index=False)
    multiplier_txt = f"{multiplier:.4f}" if multiplier is not None else "N/A"
    print(f"  [nav] {today}: NAV={today_nav:.4f}  (prev={prev_nav:.4f}  "
          f"mult={multiplier_txt})")
    return nav_df


def compute_drawdown(nav_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Compute current drawdown from 252-day rolling high-water mark.

    Returns
    -------
    drawdown_pct : float  (positive = below HWM; 0.12 = 12% drawdown)
    hwm          : float  (high-water mark)
    current_nav  : float
    """
    if nav_df.empty or "nav" not in nav_df.columns:
        return 0.0, 100.0, 100.0

    nav_series  = nav_df.sort_values("date")["nav"].values
    window      = min(252, len(nav_series))
    recent_navs = nav_series[-window:]
    hwm         = float(recent_navs.max())
    current_nav = float(nav_series[-1])
    drawdown    = max(0.0, (hwm - current_nav) / hwm)
    return drawdown, hwm, current_nav


def apply_circuit_breaker(
    drawdown_pct: float,
    hwm: float,
    current_nav: float,
    force_circuit: bool = False,
) -> dict:
    """
    Compute exposure_multiplier based on drawdown level.
    Also reads existing override file to implement hysteresis (prevent
    flip-flopping between normal and reduced exposure).

    Returns the override dict to be written to exposure_override.json.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Hysteresis: read existing override
    existing_mult = 1.0
    if OUT_OVERRIDE.exists():
        try:
            existing = json.loads(OUT_OVERRIDE.read_text())
            existing_mult = float(existing.get("exposure_multiplier", 1.0))
        except Exception:
            pass

    # Determine new multiplier
    if drawdown_pct >= DD_HARD_THRESHOLD or force_circuit:
        mult   = 0.20
        reason = f"HARD CIRCUIT BREAKER: drawdown {drawdown_pct*100:.1f}% ≥ {DD_HARD_THRESHOLD*100:.0f}%"
        level  = "HARD"
    elif drawdown_pct >= DD_SOFT_THRESHOLD:
        mult   = 0.50
        reason = f"SOFT CIRCUIT BREAKER: drawdown {drawdown_pct*100:.1f}% ≥ {DD_SOFT_THRESHOLD*100:.0f}%"
        level  = "SOFT"
    elif existing_mult < 1.0:
        # Check if we should release the circuit breaker
        # Release criterion: drawdown < hard_threshold - reset_buffer
        release_threshold = DD_HARD_THRESHOLD - DD_RESET_BUFFER
        if drawdown_pct < release_threshold:
            mult   = 1.0
            reason = f"CIRCUIT RESET: drawdown {drawdown_pct*100:.1f}% < reset level {release_threshold*100:.1f}%"
            level  = "NONE"
        else:
            # Maintain previous override (hysteresis)
            mult   = existing_mult
            reason = f"CIRCUIT MAINTAINED: drawdown {drawdown_pct*100:.1f}% in recovery zone"
            level  = "MAINTAINED"
    else:
        mult   = 1.0
        reason = f"No circuit breaker: drawdown {drawdown_pct*100:.1f}% < {DD_SOFT_THRESHOLD*100:.0f}%"
        level  = "NONE"

    override = {
        "date":               date_str,
        "exposure_multiplier": mult,
        "circuit_level":      level,
        "reason":             reason,
        "drawdown_pct":       round(drawdown_pct * 100, 2),
        "hwm":                round(hwm, 4),
        "current_nav":        round(current_nav, 4),
        "dd_soft_threshold":  DD_SOFT_THRESHOLD * 100,
        "dd_hard_threshold":  DD_HARD_THRESHOLD * 100,
    }
    return override


# =============================================================================
# 5.  Report writer
# =============================================================================

def write_report(
    orig_picks:    pd.DataFrame,
    filt_picks:    pd.DataFrame,
    corr_audit:    list[dict],
    sector_audit:  list[dict],
    override:      dict,
    sector_map:    dict[str, str],
    corr_thresh:   float,
    sector_cap:    float,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Canyon v9 — Portfolio Risk Filter Report (Step 110)",
        f"Generated: {ts}",
        "",
        "---",
        "",
    ]

    # ── Stage 1: Correlation ─────────────────────────────────────────────────
    lines += [
        f"## Stage 1 — Correlation Deduplication  (threshold: {corr_thresh:.2f})",
        "",
    ]
    removes = [a for a in corr_audit if a["action"] == "CORR_REMOVE"]
    adds    = [a for a in corr_audit if a["action"] == "CORR_ADD"]
    norep   = [a for a in corr_audit if a["action"] == "CORR_NO_REPLACEMENT"]

    if not removes:
        lines.append(f"✅  No correlated pairs found above {corr_thresh:.2f} threshold.\n")
    else:
        lines.append(f"Found {len(removes)} correlated pair(s) — {len(removes)} removed, "
                     f"{len(adds)} replaced, {len(norep)} position(s) dropped.\n")
        lines.append("| Removed | Reason | Alpha | Replaced By | New Alpha |")
        lines.append("|---------|--------|-------|-------------|-----------|")
        for rm in removes:
            repl_row = next((a for a in adds if a.get("replaced") == rm["removed"]), None)
            repl_by  = repl_row["added"] if repl_row else "— (dropped)"
            new_alp  = f"{repl_row['alpha_added']:.1f}" if repl_row else "—"
            lines.append(
                f"| {rm['removed']} | {rm['reason']} | "
                f"{rm.get('alpha_removed', 0):.1f} | {repl_by} | {new_alp} |"
            )
        lines.append("")

    # ── Stage 2: Sector ──────────────────────────────────────────────────────
    lines += [
        f"## Stage 2 — Sector Concentration  (cap: {sector_cap*100:.0f}%)",
        "",
    ]
    # Show sector breakdown AFTER filtering
    if not filt_picks.empty:
        total_w = filt_picks["weight_pct"].sum() or 1.0
        filt_picks["_sec"] = filt_picks["ticker"].map(sector_map).fillna("Unknown")
        sec_table = (
            filt_picks.groupby("_sec")["weight_pct"].sum() / total_w * 100
        ).sort_values(ascending=False)
        lines.append("| Sector | Weight | Status |")
        lines.append("|--------|--------|--------|")
        for sec, pct in sec_table.items():
            flag = "⚠️ OVER" if pct > sector_cap * 100 else ("✅" if pct < sector_cap * 100 * 0.8 else "✓")
            lines.append(f"| {sec} | {pct:.1f}% | {flag} |")
        filt_picks.drop(columns=["_sec"], inplace=True, errors="ignore")
        lines.append("")

    if sector_audit:
        lines.append("### Sector Trims")
        lines.append("")
        lines.append("| Removed | Sector | Was% | Alpha | Reason |")
        lines.append("|---------|--------|------|-------|--------|")
        for s in sector_audit:
            lines.append(
                f"| {s['removed']} | {s['sector']} | "
                f"{s['sector_w_was']}% | {s['alpha_removed']} | {s['reason']} |"
            )
        lines.append("")
    else:
        lines.append(f"✅  All sectors within {sector_cap*100:.0f}% cap.\n")

    # ── Stage 3: Drawdown ────────────────────────────────────────────────────
    mult  = override["exposure_multiplier"]
    level = override["circuit_level"]
    dd    = override["drawdown_pct"]
    hwm   = override["hwm"]
    nav   = override["current_nav"]

    status_icon = {
        "NONE":       "✅",
        "SOFT":       "⚠️",
        "HARD":       "🚨",
        "MAINTAINED": "⚠️",
    }.get(level, "?")

    lines += [
        "## Stage 3 — Drawdown Circuit Breaker",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Portfolio NAV | {nav:.4f} |",
        f"| High-Water Mark (252d) | {hwm:.4f} |",
        f"| Drawdown | {dd:.2f}% |",
        f"| Circuit Level | {status_icon} {level} |",
        f"| Exposure Multiplier | **{mult:.2f}×** |",
        f"| Reason | {override['reason']} |",
        "",
    ]

    if mult < 1.0:
        lines += [
            f"> ⚠️  **Exposure reduced to {mult*100:.0f}%**.  ",
            f"> All position weights in `daily_picks_filtered.csv` scaled by {mult:.2f}.  ",
            f"> Step 87 will also respect this override on next run.",
            "",
        ]

    # ── Final picks summary ──────────────────────────────────────────────────
    lines += [
        "## Final Filtered Picks",
        "",
        f"Original picks: {len(orig_picks)}  →  After filters: {len(filt_picks)}",
        "",
    ]
    if not filt_picks.empty:
        show_cols = [c for c in ["ticker", "action", "weight_pct", "alpha_score", "sector"]
                     if c in filt_picks.columns]
        lines.append("| " + " | ".join(show_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(show_cols)) + "|")
        for _, row in filt_picks.head(30).iterrows():
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in show_cols) + " |")
        lines.append("")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"  [written] {OUT_REPORT}")


# =============================================================================
# 6.  Main
# =============================================================================

def main(
    sector_cap:        float = SECTOR_CAP,
    corr_threshold:    float = CORR_THRESHOLD,
    no_circuit_breaker: bool = False,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}")
    print(f"Canyon v9 — Step 110: Portfolio Risk Filter  [{ts}]")
    print(f"{'='*65}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading inputs …")
    picks      = load_picks()
    alpha_all  = load_alpha_scores()
    sector_map = load_sector_map()

    if picks.empty:
        print("  No picks to filter.  Exiting.")
        return

    # Ensure alpha_score column exists (estimate from weight if missing)
    if "alpha_score" not in picks.columns:
        if "weight_pct" in picks.columns:
            picks["alpha_score"] = picks["weight_pct"]
        else:
            picks["alpha_score"] = 50.0

    orig_picks = picks.copy()
    print(f"  Loaded {len(picks)} picks | alpha_all={len(alpha_all)} | sectors={len(sector_map)}")

    # ── Stage 1: Correlation filter ───────────────────────────────────────────
    print(f"\n[2/4] Stage 1 — Correlation filter  (threshold={corr_threshold:.2f}) …")
    picks, corr_audit = correlation_filter(picks, alpha_all, corr_thresh=corr_threshold)
    n_removed = len([a for a in corr_audit if a["action"] == "CORR_REMOVE"])
    n_added   = len([a for a in corr_audit if a["action"] == "CORR_ADD"])
    if n_removed:
        print(f"  Removed {n_removed} correlated tickers, added {n_added} replacements")
    for entry in corr_audit:
        if entry["action"] == "CORR_REMOVE":
            added = next((a for a in corr_audit
                          if a["action"] == "CORR_ADD" and a.get("replaced") == entry["removed"]), None)
            repl_str = f"  → replaced by {added['added']}" if added else "  (no replacement)"
            print(f"    {entry['removed']:8s} removed ({entry['reason']}){repl_str}")

    # ── Stage 2: Sector concentration ────────────────────────────────────────
    print(f"\n[3/4] Stage 2 — Sector concentration filter  (cap={sector_cap*100:.0f}%) …")
    picks, sector_audit = sector_concentration_filter(picks, sector_map, sector_cap)

    # ── Stage 3: Drawdown circuit breaker ────────────────────────────────────
    print("\n[4/4] Stage 3 — Drawdown circuit breaker …")
    override: dict

    if no_circuit_breaker:
        override = {
            "date":               datetime.now().strftime("%Y-%m-%d"),
            "exposure_multiplier": 1.0,
            "circuit_level":      "DISABLED",
            "reason":             "--no-circuit-breaker flag set",
            "drawdown_pct":       0.0,
            "hwm":                100.0,
            "current_nav":        100.0,
            "dd_soft_threshold":  DD_SOFT_THRESHOLD * 100,
            "dd_hard_threshold":  DD_HARD_THRESHOLD * 100,
        }
        print("  Circuit breaker disabled via --no-circuit-breaker flag")
    else:
        nav_df     = load_or_create_nav(picks)
        dd_pct, hwm, cur_nav = compute_drawdown(nav_df)
        override   = apply_circuit_breaker(dd_pct, hwm, cur_nav)
        mult       = override["exposure_multiplier"]
        print(f"  Drawdown={dd_pct*100:.2f}%  HWM={hwm:.4f}  NAV={cur_nav:.4f}  "
              f"mult={mult:.2f}  [{override['circuit_level']}]")

        # Apply multiplier to weights
        if mult < 1.0:
            picks["weight_pct"] = (picks["weight_pct"] * mult).round(2)
            print(f"  ⚠️  Position weights scaled to {mult*100:.0f}% of normal")

    # Write exposure_override.json (Step 87 reads this)
    OUT_OVERRIDE.write_text(json.dumps(override, indent=2))
    print(f"  [written] {OUT_OVERRIDE}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    picks.to_csv(OUT_PICKS, index=False)
    print(f"  [written] {OUT_PICKS}  ({len(picks)} rows)")

    # Report
    write_report(
        orig_picks, picks, corr_audit, sector_audit, override,
        sector_map, corr_threshold, sector_cap,
    )

    # Console summary
    print(f"\n{'─'*65}")
    print(f"Summary:")
    print(f"  Original picks : {len(orig_picks)}")
    print(f"  After corr     : {len(orig_picks) - n_removed + n_added} (−{n_removed} +{n_added})")
    print(f"  After sector   : {len(picks)}")
    print(f"  Exposure mult  : {override['exposure_multiplier']:.2f}×  "
          f"[{override['circuit_level']}]")
    print(f"  Output         : {OUT_PICKS.name}")
    print(f"{'─'*65}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Canyon v9 Step 110 — Portfolio Risk Filter"
    )
    parser.add_argument(
        "--sector-cap", type=float, default=SECTOR_CAP,
        help=f"Max sector weight fraction (default {SECTOR_CAP:.2f})"
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=CORR_THRESHOLD,
        help=f"Pairwise correlation threshold (default {CORR_THRESHOLD:.2f})"
    )
    parser.add_argument(
        "--no-circuit-breaker", action="store_true",
        help="Disable portfolio drawdown circuit breaker"
    )
    args = parser.parse_args()

    main(
        sector_cap=args.sector_cap,
        corr_threshold=args.corr_threshold,
        no_circuit_breaker=args.no_circuit_breaker,
    )
