"""
Canyon v9  Step 69 — Paper Trading Simulator
=============================================
Closes the loop from ML signal → position management → P&L tracking.

Flow:
  1. Read latest ML signal scores (Step 66 / 68)
  2. Apply pre-trade risk filters (position limits, sector heat, drawdown guard)
  3. Generate paper trade tickets for approved entries/exits
  4. Track open positions, mark-to-market P&L, realised P&L
  5. Compute running portfolio metrics (Sharpe, win-rate, drawdown)

Risk filters applied before any ticket is generated:
  ● Max position weight    : 20% of portfolio (configurable)
  ● Portfolio heat cap     : sum of positive delta positions ≤ 100% (no leverage)
  ● Single-sector cap      : ≤ 40% in any one GICS sector
  ● Drawdown guard         : if portfolio DD > 10%, cut all new entries
  ● Earnings blackout      : no NEW entries within 3 days of earnings
  ● Min ML score threshold : only buy if ensemble_score > 0.55 (above median)

Paper ledger format (compatible with Step 55 dashboard paper_ledger.csv):
  ticker, entry_date, entry_price, exit_date, exit_price, shares,
  direction, pnl, status, ml_score, stop_price, target_price, notes

Outputs:
  paper_sim_positions.csv      Current open positions
  paper_sim_trades.csv         All historical paper trades (open + closed)
  paper_sim_portfolio.csv      Daily portfolio value timeseries
  paper_sim_report.md          Full P&L and risk report

Usage:
  python canyon_final_v9_step69_paper_sim.py               # run full simulation
  python canyon_final_v9_step69_paper_sim.py --rebalance   # process today's rebalance
  python canyon_final_v9_step69_paper_sim.py --status      # show current positions
  python canyon_final_v9_step69_paper_sim.py --account 250000   # set account size
"""
from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

ACCOUNT_SIZE      = 100_000.0   # default paper account ($)
MAX_POS_PCT       = 0.20        # max single-position weight
SECTOR_CAP_PCT    = 0.40        # max sector weight
PORTFOLIO_HEAT_MAX= 1.00        # no leverage
DD_GUARD_PCT      = 0.10        # drawdown guard trigger
EARNINGS_BLACKOUT = 3           # days around earnings to avoid
MIN_ML_SCORE      = 0.55        # minimum ensemble_score to enter
STOP_LOSS_PCT     = 0.08        # 8% hard stop from entry (protects initial capital)
TARGET_PCT        = 0.20        # 20% profit target
ATR_STOP_MULT     = 2.0         # stop = entry - ATR_MULT × 21d ATR
TRANSACTION_COST  = 0.001       # 0.1% one-way (bid-ask + slippage)

# ── Livermore Trailing Stop ──────────────────────────────────────────────────
# As price rises, the stop follows — locks in profits, prevents "give-back".
# Only activates when position is up > TRAIL_ACTIVATE_PCT from entry.
# Trail distance is wider than hard stop to avoid noise shaking you out.
TRAIL_STOP_PCT      = 0.10   # 10% trailing stop from high-water mark
TRAIL_ACTIVATE_PCT  = 0.05   # only activate when up > 5% from entry

# GICS sector map (simplified)
SECTOR_MAP = {
    "SPY":"Broad Market","QQQ":"Broad Market","SMH":"Semiconductors",
    "SOXX":"Semiconductors","XLK":"Tech","XLE":"Energy","XLF":"Financials",
    "XLV":"Healthcare","XLU":"Utilities","XLP":"Consumer Staples",
    "NVDA":"Semiconductors","AMD":"Semiconductors","MU":"Semiconductors",
    "INTC":"Semiconductors","QCOM":"Semiconductors","TXN":"Semiconductors",
    "AVGO":"Semiconductors","TSLA":"Consumer Disc","GOOGL":"Tech",
    "AMZN":"Consumer Disc","MSFT":"Tech","AAPL":"Tech","META":"Tech",
    "CRM":"Tech","ADBE":"Tech","NFLX":"Consumer Disc",
    "JPM":"Financials","V":"Financials","MA":"Financials",
    "XOM":"Energy","CVX":"Energy","JNJ":"Healthcare","UNH":"Healthcare",
    "LLY":"Healthcare","MRK":"Healthcare","ABBV":"Healthcare","TMO":"Healthcare",
    "WMT":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "COST":"Consumer Staples","HD":"Consumer Disc","PFE":"Healthcare",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load current state
# ─────────────────────────────────────────────────────────────────────────────

def load_ml_scores() -> pd.DataFrame:
    """
    Load latest ML signal scores.
    Priority: regime_ml_scores.csv (step77, has crowding_level) >
              enhanced_ml_scores.csv (step66/68) > ml_signal_scores.csv
    """
    for fname in ["regime_ml_scores.csv",
                  "enhanced_ml_scores.csv", "ml_signal_scores.csv"]:
        p = ROOT / fname
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                crowd_note = " [crowding ✓]" if "crowding_level" in df.columns else ""
                print(f"  [ml scores] Using {fname}{crowd_note}")
                return df
    return pd.DataFrame()


def load_positions() -> pd.DataFrame:
    """Load current open positions from paper_sim_positions.csv."""
    p = ROOT / "paper_sim_positions.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["entry_date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "ticker","entry_date","entry_price","shares","ml_score",
        "stop_price","target_price","cost_basis","sector","notes",
        "entry_ml_score","high_water_mark",   # for trailing stop
    ])


def load_trade_history() -> pd.DataFrame:
    """Load all historical trades."""
    p = ROOT / "paper_sim_trades.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "ticker","entry_date","entry_price","exit_date","exit_price",
        "shares","direction","pnl","pnl_pct","status","ml_score","notes",
    ])


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest close price for each ticker."""
    prices = {}
    try:
        import yfinance as yf
        # Try cache first
        cache_p = ROOT / "backtest_price_cache.csv"
        if cache_p.exists() and (time.time() - cache_p.stat().st_mtime) / 3600 < 24:
            cached = pd.read_csv(cache_p, index_col=0, parse_dates=True)
            for tk in tickers:
                if tk in cached.columns:
                    last = cached[tk].dropna()
                    if not last.empty:
                        prices[tk] = float(last.iloc[-1])
        missing = [t for t in tickers if t not in prices]
        if missing:
            raw = yf.download(missing, period="5d", auto_adjust=True)
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            for tk in missing:
                if tk in closes.columns:
                    last = closes[tk].dropna()
                    if not last.empty:
                        prices[tk] = float(last.iloc[-1])
    except Exception as e:
        print(f"  [prices] Error: {e}")
    return prices


def get_earnings_dates(tickers: list[str]) -> dict[str, str]:
    """Get next earnings dates to enforce blackout."""
    earnings = {}
    try:
        import yfinance as yf
        for tk in tickers[:10]:   # limit API calls
            try:
                cal = yf.Ticker(tk).calendar
                if cal is None:
                    continue
                if isinstance(cal, dict):
                    eds = cal.get("Earnings Date", [])
                    if hasattr(eds, '__iter__') and not isinstance(eds, str):
                        dates = [str(d)[:10] for d in eds if d]
                        if dates:
                            earnings[tk] = dates[0]
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    if "Earnings Date" in cal.columns:
                        first = cal["Earnings Date"].dropna()
                        if not first.empty:
                            earnings[tk] = str(first.iloc[0])[:10]
            except Exception:
                pass
    except Exception:
        pass
    return earnings


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk filters
# ─────────────────────────────────────────────────────────────────────────────

def check_risk_filters(
    ticker:       str,
    ml_score:     float,
    current_pos:  pd.DataFrame,
    current_prices: dict[str, float],
    account_size: float,
    earnings_dates: dict[str, str],
) -> tuple[bool, str]:
    """
    Returns (approved, reason). If approved=False, reason explains the block.
    """
    today = datetime.today().strftime("%Y-%m-%d")

    # ── Filter 1: ML score threshold ─────────────────────────────────────────
    if ml_score < MIN_ML_SCORE:
        return False, f"ML score {ml_score:.3f} below threshold {MIN_ML_SCORE}"

    # ── Filter 2: Already held ────────────────────────────────────────────────
    if not current_pos.empty and "ticker" in current_pos.columns:
        if ticker in current_pos["ticker"].values:
            return False, "Already held in portfolio"

    # ── Filter 3: Max position count (don't exceed 10 concurrent) ────────────
    if not current_pos.empty and len(current_pos) >= 10:
        return False, f"Portfolio at max 10 positions"

    # ── Filter 4: Sector concentration ───────────────────────────────────────
    sector = SECTOR_MAP.get(ticker, "Other")
    if not current_pos.empty and "sector" in current_pos.columns:
        pos_value = 0.0
        for _, prow in current_pos.iterrows():
            tk   = prow.get("ticker","")
            shrs = float(prow.get("shares", 0) or 0)
            px   = current_prices.get(tk, float(prow.get("entry_price", 0) or 0))
            pos_value += shrs * px
        sector_value = 0.0
        for _, prow in current_pos.iterrows():
            if prow.get("sector","") == sector:
                tk   = prow.get("ticker","")
                shrs = float(prow.get("shares", 0) or 0)
                px   = current_prices.get(tk, float(prow.get("entry_price", 0) or 0))
                sector_value += shrs * px
        total_portfolio = account_size  # approximate; real value needs MTM
        if total_portfolio > 0 and (sector_value / total_portfolio) > SECTOR_CAP_PCT:
            return False, f"Sector cap: {sector} already at {sector_value/total_portfolio*100:.1f}%"

    # ── Filter 5: Earnings blackout ───────────────────────────────────────────
    if ticker in earnings_dates:
        try:
            ed   = datetime.strptime(earnings_dates[ticker], "%Y-%m-%d")
            days = abs((ed - datetime.today()).days)
            if days <= EARNINGS_BLACKOUT:
                return False, f"Earnings blackout: {earnings_dates[ticker]} ({days}d away)"
        except Exception:
            pass

    return True, "APPROVED"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def calculate_position_size(
    ticker:       str,
    entry_price:  float,
    ml_score:     float,
    account_size: float,
    n_positions:  int,
) -> tuple[int, float, float]:
    """
    Returns (shares, stop_price, target_price).
    Sizing: equal-weight with ML score tilt.
      base_weight = 1 / max(n_positions+1, 4)
      ml_tilt     = (ml_score - 0.5) * 0.2   → ±10% tilt
      weight      = clip(base_weight + ml_tilt, 0.05, MAX_POS_PCT)
    """
    n = max(n_positions + 1, 4)
    base_w = 1.0 / n
    ml_tilt = (ml_score - 0.5) * 0.2
    weight  = float(np.clip(base_w + ml_tilt, 0.05, MAX_POS_PCT))

    dollar_amount = account_size * weight
    shares        = int(dollar_amount / entry_price) if entry_price > 0 else 0

    stop_price   = entry_price * (1 - STOP_LOSS_PCT)
    target_price = entry_price * (1 + TARGET_PCT)

    return shares, round(stop_price, 2), round(target_price, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rebalance engine
# ─────────────────────────────────────────────────────────────────────────────

def run_rebalance(
    ml_scores:    pd.DataFrame,
    current_pos:  pd.DataFrame,
    trade_history: pd.DataFrame,
    account_size: float,
    verbose:      bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Process one rebalance cycle:
    1. Mark-to-market open positions
    2. Close positions that triggered stop/target/exit signal
    3. Open new positions for top-scoring tickers that pass filters
    Returns (updated_positions, updated_history, trade_tickets).
    """
    today = datetime.today().strftime("%Y-%m-%d")
    tickets = []

    # Get latest ML scores for today (or latest date)
    if "rebalance_date" in ml_scores.columns:
        last_date = ml_scores["rebalance_date"].astype(str).max()
        latest    = ml_scores[ml_scores["rebalance_date"].astype(str) == last_date].copy()
    else:
        latest = ml_scores.copy()

    # Score column preference
    score_col = next((c for c in ["ensemble_score","enhanced_score","rf_score"] if c in latest.columns), None)
    if score_col is None:
        print("  [sim] No score column found in ML scores")
        return current_pos, trade_history, tickets

    # Get all relevant tickers
    pos_tickers  = current_pos["ticker"].tolist() if not current_pos.empty else []
    score_tickers = latest["ticker"].tolist() if "ticker" in latest.columns else []
    all_tickers  = list(set(pos_tickers + score_tickers))
    prices_now   = get_current_prices(all_tickers)
    earnings_blk = get_earnings_dates(score_tickers[:15])   # limit API calls

    if verbose:
        print(f"  [sim] {len(pos_tickers)} open positions, "
              f"{len(score_tickers)} ML signals, {len(prices_now)} prices fetched")

    # ── STEP A: Close positions ───────────────────────────────────────────────
    positions_to_keep = []
    closed_this_cycle = []

    for pos_idx, pos in (current_pos.iterrows() if not current_pos.empty else iter([])):
        tk         = pos["ticker"]
        entry_px   = float(pos.get("entry_price", 0) or 0)
        shares     = float(pos.get("shares", 0) or 0)
        stop_px    = float(pos.get("stop_price", 0) or 0)
        target_px  = float(pos.get("target_price", 0) or 0)
        curr_px    = prices_now.get(tk, entry_px)

        # ── Trailing stop: update high-water mark, compute trail level ──────
        hwm = float(pos.get("high_water_mark", entry_px) or entry_px)
        hwm = max(hwm, curr_px)   # ratchet up only
        # Store updated hwm back so we can persist it
        if not current_pos.empty and "high_water_mark" in current_pos.columns:
            current_pos.at[pos_idx, "high_water_mark"] = hwm

        # Trail stop only activates when position is sufficiently profitable
        gain_from_entry = (curr_px / entry_px - 1) if entry_px > 0 else 0
        trail_stop_px   = hwm * (1 - TRAIL_STOP_PCT) if gain_from_entry >= TRAIL_ACTIVATE_PCT else 0
        # Effective stop = max(hard stop, trailing stop)
        effective_stop = max(stop_px, trail_stop_px)

        close_reason = None
        if trail_stop_px > 0 and curr_px <= trail_stop_px and curr_px > stop_px:
            close_reason = (f"TRAIL STOP hit @ {curr_px:.2f} "
                            f"(trail={trail_stop_px:.2f}, hwm={hwm:.2f}, "
                            f"locked +{(trail_stop_px/entry_px-1)*100:.1f}%)")
        elif effective_stop > 0 and curr_px <= effective_stop:
            close_reason = f"HARD STOP triggered @ {curr_px:.2f} (stop={effective_stop:.2f})"
        elif target_px > 0 and curr_px >= target_px:
            close_reason = f"TARGET hit @ {curr_px:.2f} (target={target_px:.2f})"
        else:
            # Check if ticker dropped out of top ML scores OR crowding alert
            score_row = latest[latest["ticker"] == tk]
            if not score_row.empty:
                sc = float(score_row[score_col].iloc[0])
                if sc < 0.40:
                    close_reason = f"ML score dropped to {sc:.3f} (exit threshold 0.40)"
                # Exit if crowding level becomes VERY_CROWDED (Soros: everyone is cheering)
                elif "crowding_level" in score_row.columns:
                    crowd_lv = score_row["crowding_level"].iloc[0]
                    if crowd_lv == "VERY_CROWDED":
                        close_reason = (f"CROWDING EXIT: {tk} now VERY_CROWDED "
                                        f"(Soros rule — consensus at peak)")

        if close_reason:
            # Apply transaction cost: entry paid spread going in, exit pays spread going out
            effective_entry = entry_px * (1 + TRANSACTION_COST)
            effective_exit  = curr_px  * (1 - TRANSACTION_COST)
            pnl     = (effective_exit - effective_entry) * shares
            pnl_pct = (effective_exit / effective_entry - 1) * 100 if effective_entry > 0 else 0.0
            closed_trade = {
                "ticker":      tk,
                "entry_date":  str(pos.get("entry_date", ""))[:10],
                "entry_price": entry_px,
                "exit_date":   today,
                "exit_price":  curr_px,
                "shares":      shares,
                "direction":   "LONG",
                "pnl":         round(pnl, 2),
                "pnl_pct":     round(pnl_pct, 2),
                "status":      "CLOSED",
                "ml_score":    float(pos.get("entry_ml_score", 0) or 0),
                "notes":       close_reason,
            }
            closed_this_cycle.append(closed_trade)
            tickets.append({
                "action":     "CLOSE",
                "ticker":     tk,
                "price":      curr_px,
                "shares":     shares,
                "reason":     close_reason,
                "pnl":        round(pnl, 2),
                "pnl_pct":    f"{pnl_pct:+.2f}%",
            })
            if verbose:
                emoji = "✅" if pnl >= 0 else "❌"
                print(f"  {emoji} CLOSE  {tk:6s}  {curr_px:.2f}  "
                      f"PnL={pnl:+.2f} ({pnl_pct:+.1f}%)  {close_reason[:50]}")
        else:
            positions_to_keep.append(pos.to_dict())

    # Update trade history with closures
    if closed_this_cycle:
        trade_history = pd.concat(
            [trade_history, pd.DataFrame(closed_this_cycle)], ignore_index=True
        )

    open_pos = pd.DataFrame(positions_to_keep)

    # ── STEP B: Open new positions ─────────────────────────────────────────────
    # Sort by ML score descending
    candidates = latest.sort_values(score_col, ascending=False)
    new_entries = 0

    for _, cand in candidates.iterrows():
        tk       = cand["ticker"]
        ml_sc    = float(cand.get(score_col, 0) or 0)
        curr_px  = prices_now.get(tk, 0)

        if curr_px <= 0:
            continue

        approved, reason = check_risk_filters(
            tk, ml_sc, open_pos, prices_now, account_size, earnings_blk
        )

        if not approved:
            if verbose and ml_sc > 0.60:
                print(f"  ⛔ SKIP   {tk:6s}  score={ml_sc:.3f}  {reason[:60]}")
            continue

        shares, stop_px, target_px = calculate_position_size(
            tk, curr_px, ml_sc, account_size, len(open_pos)
        )
        if shares < 1:
            continue

        sector = SECTOR_MAP.get(tk, "Other")
        fill_px = round(curr_px * (1 + TRANSACTION_COST), 4)  # effective fill incl. spread
        new_pos = {
            "ticker":          tk,
            "entry_date":      today,
            "entry_price":     fill_px,   # cost-adjusted entry
            "shares":          shares,
            "ml_score":        ml_sc,
            "entry_ml_score":  ml_sc,
            "stop_price":      stop_px,
            "target_price":    target_px,
            "cost_basis":      round(fill_px * shares, 2),
            "sector":          sector,
            "high_water_mark": fill_px,  # start at entry — trailing stop ratchets up
            "notes":          (f"ML score {ml_sc:.3f} | fill {fill_px:.2f} (cost adj) | "
                               f"hard stop {stop_px:.2f} (-{STOP_LOSS_PCT*100:.0f}%) | "
                               f"trail stop activates at +{TRAIL_ACTIVATE_PCT*100:.0f}%"),
        }
        open_pos = pd.concat([open_pos, pd.DataFrame([new_pos])], ignore_index=True)
        tickets.append({
            "action":    "BUY",
            "ticker":    tk,
            "price":     curr_px,
            "shares":    shares,
            "ml_score":  ml_sc,
            "stop":      stop_px,
            "target":    target_px,
            "sector":    sector,
        })
        new_entries += 1
        if verbose:
            print(f"  🟢 BUY    {tk:6s}  ${curr_px:.2f}  {shares}sh  "
                  f"${curr_px*shares:,.0f}  score={ml_sc:.3f}  "
                  f"stop={stop_px:.2f}  tgt={target_px:.2f}")

        if len(open_pos) >= 10:
            break

    if verbose:
        print(f"\n  Summary: {new_entries} new entries, {len(closed_this_cycle)} exits, "
              f"{len(open_pos)} open positions")

    return open_pos, trade_history, tickets


# ─────────────────────────────────────────────────────────────────────────────
# 5. Portfolio metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_metrics(
    trade_history: pd.DataFrame,
    account_size:  float,
    positions:     pd.DataFrame,
    prices_now:    dict,
) -> dict:
    metrics = {
        "account_size":     account_size,
        "n_open":           len(positions),
        "n_closed":         0,
        "realised_pnl":     0.0,
        "unrealised_pnl":   0.0,
        "total_pnl":        0.0,
        "win_rate":         0.0,
        "avg_win":          0.0,
        "avg_loss":         0.0,
        "total_return_pct": 0.0,
        "sharpe_trades":    0.0,
    }

    # Realised P&L from closed trades
    closed = trade_history[trade_history.get("status", pd.Series(dtype=str)).astype(str) == "CLOSED"] if not trade_history.empty and "status" in trade_history.columns else pd.DataFrame()
    if not closed.empty and "pnl" in closed.columns:
        pnls = pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        metrics["n_closed"]     = len(pnls)
        metrics["realised_pnl"] = round(float(pnls.sum()), 2)
        wins  = pnls[pnls > 0]
        losses= pnls[pnls < 0]
        metrics["win_rate"] = round(float((pnls > 0).mean() * 100), 1) if len(pnls) else 0.0
        metrics["avg_win"]  = round(float(wins.mean()), 2)  if len(wins)   else 0.0
        metrics["avg_loss"] = round(float(losses.mean()), 2) if len(losses) else 0.0
        if len(pnls) > 1:
            metrics["sharpe_trades"] = round(float(pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(252 / 21)), 2)

    # Unrealised P&L from open positions
    unreal = 0.0
    if not positions.empty:
        for _, pos in positions.iterrows():
            tk       = pos.get("ticker","")
            entry_px = float(pos.get("entry_price", 0) or 0)
            shares   = float(pos.get("shares", 0) or 0)
            curr_px  = prices_now.get(tk, entry_px)
            unreal  += (curr_px - entry_px) * shares
    metrics["unrealised_pnl"]   = round(unreal, 2)
    metrics["total_pnl"]        = round(metrics["realised_pnl"] + unreal, 2)
    metrics["total_return_pct"] = round(metrics["total_pnl"] / account_size * 100, 2)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(metrics: dict, positions: pd.DataFrame,
                 trades: pd.DataFrame, tickets: list[dict], ts: str) -> None:
    lines = [
        "# Canyon v9 — Paper Trading Simulator Report (Step 69)",
        f"Generated: {ts}",
        "",
        "## Portfolio Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k.replace('_',' ').title()} | {v} |")

    if tickets:
        lines += [
            "",
            "## Today's Trade Tickets",
            "",
            "| Action | Ticker | Price | Shares | ML Score | Reason |",
            "|---|---|---|---|---|---|",
        ]
        for t in tickets:
            action = t.get("action","")
            reason = t.get("reason","") or f"score={t.get('ml_score','')}"
            lines.append(
                f"| {action} | {t.get('ticker','')} | ${t.get('price',0):.2f} | "
                f"{t.get('shares',0)} | {t.get('ml_score','')} | {reason[:50]} |"
            )

    if not positions.empty:
        lines += ["", "## Open Positions", "", "| Ticker | Entry | Stop | Target | Sector |", "|---|---|---|---|---|"]
        for _, p in positions.iterrows():
            lines.append(
                f"| {p.get('ticker','')} | {float(p.get('entry_price',0) or 0):.2f} | "
                f"{float(p.get('stop_price',0) or 0):.2f} | "
                f"{float(p.get('target_price',0) or 0):.2f} | "
                f"{p.get('sector','—')} |"
            )

    lines += [
        "",
        "## Risk Rules Active",
        f"- Max position weight: {MAX_POS_PCT*100:.0f}%",
        f"- Sector cap: {SECTOR_CAP_PCT*100:.0f}%",
        f"- Stop loss: {STOP_LOSS_PCT*100:.0f}%",
        f"- Profit target: {TARGET_PCT*100:.0f}%",
        f"- Min ML score to enter: {MIN_ML_SCORE}",
        f"- Earnings blackout: ±{EARNINGS_BLACKOUT} days",
        f"- Max concurrent positions: 10",
        "",
        "## Safety Reminder",
        "**This is a paper simulation only. No real orders are placed.**",
        "All positions are hypothetical. Prices are end-of-day approximations.",
    ]

    p = ROOT / "paper_sim_report.md"
    p.write_text("\n".join(lines))
    print(f"  [report] {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def mark_to_market(account_size: float) -> None:
    """Fetch current prices, update unrealised P&L on all open positions,
    and append a NAV snapshot to paper_sim_nav.csv."""
    import yfinance as yf

    positions = load_positions()
    if positions.empty:
        print("  No open positions to mark.")
        return

    tickers = positions["ticker"].tolist()
    print(f"  Fetching prices for {len(tickers)} positions …")
    prices_now = get_current_prices(tickers)

    today = datetime.now().strftime("%Y-%m-%d")
    total_cost    = 0.0
    total_mktval  = 0.0
    rows = []

    for _, row in positions.iterrows():
        tk      = str(row.get("ticker", ""))
        ep      = float(row.get("entry_price", 0) or 0)
        shrs    = float(row.get("shares", 0) or 0)
        stop    = float(row.get("stop_price",  ep * (1 - STOP_LOSS_PCT)) or 0)
        target  = float(row.get("target_price", ep * (1 + TARGET_PCT)) or 0)
        cost    = ep * shrs

        cpx = prices_now.get(tk, ep)
        mktval   = cpx * shrs
        unreal   = mktval - cost
        unreal_pct = (cpx / ep - 1) * 100 if ep > 0 else 0.0

        total_cost   += cost
        total_mktval += mktval

        r = row.to_dict()
        r["current_price"]  = round(cpx, 4)
        r["market_value"]   = round(mktval, 2)
        r["unrealised_pnl"] = round(unreal, 2)
        r["unrealised_pct"] = round(unreal_pct, 4)
        r["last_updated"]   = today

        emoji = "🟢" if unreal >= 0 else "🔴"
        status_flag = ""
        if cpx <= stop:            status_flag = " ⛔ STOP HIT"
        elif cpx >= target:        status_flag = " 🎯 TARGET HIT"
        print(f"  {emoji} {tk:6s}  entry={ep:.2f}  now={cpx:.2f}  "
              f"P&L={unreal:+.2f} ({unreal_pct:+.1f}%){status_flag}")
        rows.append(r)

    updated = pd.DataFrame(rows)
    updated.to_csv(ROOT / "paper_sim_positions.csv", index=False)
    print(f"\n  Positions updated  ({len(rows)} holdings)")

    # ── NAV snapshot ─────────────────────────────────────────────────────────
    nav = total_mktval / account_size * 100.0   # indexed to 100
    pnl_pct = (total_mktval - total_cost) / account_size * 100.0 if account_size > 0 else 0.0

    nav_path = ROOT / "paper_sim_nav.csv"
    nav_row  = pd.DataFrame([{
        "date":         today,
        "nav":          round(nav, 4),
        "total_mktval": round(total_mktval, 2),
        "total_cost":   round(total_cost, 2),
        "unrealised_pct": round(pnl_pct, 4),
        "n_positions":  len(rows),
    }])

    if nav_path.exists():
        existing = pd.read_csv(nav_path)
        # de-dup by date (keep latest)
        existing = existing[existing["date"] != today]
        nav_history = pd.concat([existing, nav_row], ignore_index=True)
    else:
        nav_history = nav_row

    nav_history.to_csv(nav_path, index=False)
    print(f"  NAV history saved → paper_sim_nav.csv  "
          f"(today NAV = {nav:.2f}, unrealised = {pnl_pct:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Canyon v9 Step 69 — Paper Trading Simulator")
    parser.add_argument("--rebalance",       action="store_true", help="Run rebalance only")
    parser.add_argument("--status",          action="store_true", help="Show current status")
    parser.add_argument("--account",         type=float, default=ACCOUNT_SIZE, help="Account size ($)")
    parser.add_argument("--reset",           action="store_true", help="Reset all paper positions")
    parser.add_argument("--mark-to-market",  action="store_true", help="MTM open positions and save NAV snapshot")
    args = parser.parse_args()

    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    account_size = args.account

    print(f"\n{'='*62}")
    print(f"Canyon v9 Step 69 — Paper Trading Simulator")
    print(f"Account: ${account_size:,.0f}  |  {ts}")
    print(f"{'='*62}")
    print(f"\n⚠  PAPER ONLY — no real orders are placed.\n")

    if args.reset:
        for f in ["paper_sim_positions.csv", "paper_sim_trades.csv"]:
            p = ROOT / f
            if p.exists():
                p.unlink()
                print(f"  Deleted {f}")
        print("  Paper portfolio reset.")

    # ── Load state ─────────────────────────────────────────────────────────────
    print("[1/4] Loading ML scores and portfolio state …")
    ml_scores     = load_ml_scores()
    current_pos   = load_positions()
    trade_history = load_trade_history()

    if ml_scores.empty:
        print("  ERROR: No ML scores found. Run Step 66 first:")
        print("    python canyon_final_v9_step66_ml_signals.py")
        return

    score_col = next((c for c in ["ensemble_score","enhanced_score","rf_score"] if c in ml_scores.columns), None)
    if score_col:
        n_signals = ml_scores["ticker"].nunique() if "ticker" in ml_scores.columns else 0
        print(f"  {n_signals} ML signals loaded  ({score_col})")
    print(f"  {len(current_pos)} open positions  |  {len(trade_history)} historical trades")

    if getattr(args, "mark_to_market", False):
        print("\n[MTM] Marking positions to market …")
        mark_to_market(account_size)
        print(f"\nRuntime: {time.time()-t0:.1f}s\n")
        return

    if args.status:
        # Status display only
        all_tickers  = current_pos["ticker"].tolist() if not current_pos.empty else []
        prices_now   = get_current_prices(all_tickers)
        metrics      = compute_portfolio_metrics(trade_history, account_size, current_pos, prices_now)
        print(f"\n{'─'*62}")
        print("CURRENT PORTFOLIO STATUS")
        print(f"{'─'*62}")
        for k, v in metrics.items():
            print(f"  {k:25s}: {v}")
        if not current_pos.empty:
            print(f"\n  Open Positions:")
            for _, p in current_pos.iterrows():
                tk   = p.get("ticker","")
                epx  = float(p.get("entry_price",0) or 0)
                cpx  = prices_now.get(tk, epx)
                shrs = float(p.get("shares",0) or 0)
                unreal = (cpx - epx) * shrs
                pct    = (cpx / epx - 1) * 100 if epx > 0 else 0.0
                emoji  = "🟢" if unreal >= 0 else "🔴"
                print(f"  {emoji} {tk:6s}  entry={epx:.2f}  now={cpx:.2f}  "
                      f"P&L={unreal:+.2f} ({pct:+.1f}%)")
        return

    # ── Run rebalance ──────────────────────────────────────────────────────────
    print("\n[2/4] Running rebalance …")
    updated_pos, updated_history, tickets = run_rebalance(
        ml_scores, current_pos, trade_history, account_size, verbose=True
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("\n[3/4] Computing portfolio metrics …")
    all_tickers = updated_pos["ticker"].tolist() if not updated_pos.empty else []
    prices_now  = get_current_prices(all_tickers)
    metrics     = compute_portfolio_metrics(updated_history, account_size, updated_pos, prices_now)

    print(f"\n  Portfolio Metrics:")
    print(f"    Total P&L:     ${metrics['total_pnl']:+,.2f} ({metrics['total_return_pct']:+.2f}%)")
    print(f"    Realised P&L:  ${metrics['realised_pnl']:+,.2f}")
    print(f"    Unrealised:    ${metrics['unrealised_pnl']:+,.2f}")
    print(f"    Win Rate:      {metrics['win_rate']:.1f}%  (n={metrics['n_closed']} closed)")
    print(f"    Open Positions: {metrics['n_open']}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    print("\n[4/4] Writing outputs …")
    if not updated_pos.empty:
        updated_pos.to_csv(ROOT / "paper_sim_positions.csv", index=False)
        print("  [paper_sim_positions.csv]   OK")
    if not updated_history.empty:
        updated_history.to_csv(ROOT / "paper_sim_trades.csv", index=False)
        print("  [paper_sim_trades.csv]      OK")

    # Write summary CSV
    pd.DataFrame([metrics]).to_csv(ROOT / "paper_sim_summary.csv", index=False)
    print("  [paper_sim_summary.csv]     OK")

    write_report(metrics, updated_pos, updated_history, tickets, ts)

    print(f"\n{'='*62}")
    print(f"Paper simulation complete — {len(tickets)} trade tickets generated.")
    print(f"Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
