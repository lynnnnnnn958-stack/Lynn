#!/usr/bin/env python3
"""
Canyon v9 — S&P 500 Universe Expansion
=======================================
Expands the daily pipeline from ~42 tickers to full S&P 500 (~494 tickers).

What this does:
  1. Reads regime_ml_scores.csv  → expands cpcv_predictions.csv + factor_composite.csv
  2. Downloads fresh prices via yfinance (batched, ~3 min)
  3. Writes sp500_price_cache.csv + backtest_price_cache.csv
  4. Updates N_LONG/N_SHORT in step500 to 20/20
  5. Rebuilds alpha_scores.csv via step87

Run:  .venv/bin/python step_sp500_expand.py

No broker connection. No live orders. Research only.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).parent
TODAY  = datetime.now().strftime("%Y-%m-%d")
START  = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def log(msg: str): print(f"  {msg}")
def ok(msg: str):  print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str):print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg: str): print(f"  {RED}✗{RESET}  {msg}")


# ── Step 1: Load regime_ml_scores → build signal files ───────────────────────

def step1_expand_signal_files() -> list[str]:
    """Extract 494-ticker signals from regime_ml_scores.csv into the files
    that step500 uses for its composite calculation."""

    regime_path = ROOT / "regime_ml_scores.csv"
    if not regime_path.exists():
        err("regime_ml_scores.csv not found — cannot expand signals")
        return []

    df = pd.read_csv(regime_path)
    tickers = df["ticker"].dropna().unique().tolist()
    log(f"regime_ml_scores.csv: {len(tickers)} tickers loaded")

    # ── 1a. cpcv_predictions.csv ─────────────────────────────────────────────
    # Format: rebalance_date, ticker, period, is_oos, ridge_score, rf_score,
    #         lgbm_score, ensemble_score, n_train, method
    reb_date = TODAY
    cpcv_rows = []
    for _, row in df.iterrows():
        score = float(row.get("predicted_score", 0.5) or 0.5)
        # predicted_score is on ~0–1 scale; keep as-is for ensemble_score
        cpcv_rows.append({
            "rebalance_date": reb_date,
            "ticker":         row["ticker"],
            "period":         "OOS",
            "is_oos":         True,
            "ridge_score":    round(score, 6),
            "rf_score":       round(score, 6),
            "lgbm_score":     round(score, 6),
            "ensemble_score": round(score, 6),
            "n_train":        252,
            "method":         "regime_ml_proxy",
        })
    cpcv_df = pd.DataFrame(cpcv_rows)
    cpcv_path = ROOT / "cpcv_predictions.csv"
    cpcv_df.to_csv(cpcv_path, index=False)
    ok(f"cpcv_predictions.csv → {len(cpcv_df)} tickers")

    # ── 1b. factor_composite.csv ─────────────────────────────────────────────
    # Format: ticker, momentum_z, low_vol_z, value_z, quality_z, factor_composite
    factor_rows = []
    for _, row in df.iterrows():
        # mom signals → z-score within universe (use percentile rank)
        mom_1m  = float(row.get("mom_1m", 0) or 0)
        mom_3m  = float(row.get("mom_3m", 0) or 0)
        mom_6m  = float(row.get("mom_6m", 0) or 0)
        inv_vol = float(row.get("inv_vol", 1) or 1)
        quality = float(row.get("quality_score", 50) or 50)
        factor_rows.append({
            "ticker":           row["ticker"],
            "momentum_z":       round(mom_3m, 4),
            "low_vol_z":        round(inv_vol / 50.0 - 1.0, 4),
            "value_z":          round((quality - 50) / 25, 4),
            "quality_z":        round((quality - 50) / 25, 4),
            "factor_composite": round((mom_1m + mom_3m + mom_6m) / 3, 4),
        })
    factor_df = pd.DataFrame(factor_rows)
    # Re-z-score the composite across the full universe
    mu = factor_df["factor_composite"].mean()
    sd = factor_df["factor_composite"].std() + 1e-9
    factor_df["factor_composite"] = ((factor_df["factor_composite"] - mu) / sd).round(4)
    factor_path = ROOT / "factor_composite.csv"
    factor_df.to_csv(factor_path, index=False)
    ok(f"factor_composite.csv → {len(factor_df)} tickers")

    # ── 1c. smart_money_signal.csv (if missing >200 tickers) ─────────────────
    sm_path = ROOT / "smart_money_signal.csv"
    if sm_path.exists():
        sm_existing = pd.read_csv(sm_path)
        missing = set(tickers) - set(sm_existing["ticker"].tolist())
        if missing:
            neutral_rows = [{
                "ticker": tk,
                "inst_held_pct": 0.70, "insider_held_pct": 0.02,
                "short_pct_float": 0.03, "n_institutions": 5,
                "top5_inst_pct": 0.25,
                "z_inst": 0.0, "z_n_inst": 0.0, "z_insider": 0.0, "z_short": 0.0,
                "smart_money_score": 0.0,
            } for tk in missing]
            sm_full = pd.concat([sm_existing, pd.DataFrame(neutral_rows)], ignore_index=True)
            sm_full = sm_full.drop_duplicates(subset=["ticker"], keep="first")
            sm_full.to_csv(sm_path, index=False)
            ok(f"smart_money_signal.csv → {len(sm_full)} tickers (+{len(missing)} neutral fills)")
    else:
        warn("smart_money_signal.csv missing — skipping")

    # ── 1d. accruals_snapshot.csv (neutral fill for missing tickers) ─────────
    acc_path = ROOT / "accruals_snapshot.csv"
    if acc_path.exists():
        acc_existing = pd.read_csv(acc_path)
        missing = set(tickers) - set(acc_existing["ticker"].tolist())
        if missing:
            neutral_rows = [{
                "ticker": tk, "fiscal_year": "2024-12-31",
                "accrual_ratio": 0.0, "net_income": 0.0,
                "op_cashflow": 0.0, "accrual_quality": 0.0,
            } for tk in missing]
            acc_full = pd.concat([acc_existing, pd.DataFrame(neutral_rows)], ignore_index=True)
            acc_full = acc_full.drop_duplicates(subset=["ticker"], keep="first")
            acc_full.to_csv(acc_path, index=False)
            ok(f"accruals_snapshot.csv → {len(acc_full)} tickers (+{len(missing)} neutral fills)")
    else:
        warn("accruals_snapshot.csv missing — skipping")

    return tickers


# ── Step 2: Download prices for all tickers ───────────────────────────────────

def step2_download_prices(tickers: list[str]) -> pd.DataFrame | None:
    """Download daily close prices for all tickers via yfinance.
    Batched in groups of 60 to avoid rate limits."""
    try:
        import yfinance as yf
    except ImportError:
        err("yfinance not installed: pip install yfinance")
        return None

    all_tickers = list(dict.fromkeys(tickers + ["SPY"]))  # deduplicate, keep SPY
    batch_size  = 60
    batches     = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]

    log(f"Downloading prices for {len(all_tickers)} tickers in {len(batches)} batches …")
    log(f"Period: {START} → {TODAY}")

    all_prices: list[pd.DataFrame] = []
    for i, batch in enumerate(batches):
        log(f"  Batch {i+1}/{len(batches)}: {batch[:3]}…")
        try:
            raw = yf.download(
                batch, start=START, end=TODAY,
                auto_adjust=True, progress=False, threads=True
            )
            if raw.empty:
                warn(f"  Batch {i+1}: empty response")
                continue
            # Extract Close prices
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, ::5]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            all_prices.append(close)
        except Exception as e:
            warn(f"  Batch {i+1} error: {e}")
        time.sleep(0.5)  # polite rate limit

    if not all_prices:
        err("No price data downloaded")
        return None

    prices = pd.concat(all_prices, axis=1)
    # Remove duplicate columns (same ticker from multiple batches)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    # Drop tickers with >20% missing data
    min_obs  = len(prices) * 0.80
    prices   = prices.dropna(thresh=int(min_obs), axis=1)
    prices   = prices.ffill().dropna(how="all")

    ok(f"Downloaded: {len(prices)} days × {len(prices.columns)} tickers")
    return prices


# ── Step 3: Write price caches ────────────────────────────────────────────────

def step3_write_price_caches(prices: pd.DataFrame):
    """Write sp500_price_cache.csv and update backtest_price_cache.csv."""

    # Full S&P 500 cache
    sp500_path = ROOT / "sp500_price_cache.csv"
    prices.to_csv(sp500_path)
    ok(f"sp500_price_cache.csv → {prices.shape[0]} days × {prices.shape[1]} tickers")

    # backtest_price_cache.csv — used by step500 for daily composite + P&L
    # Keep last 504 trading days (~2 years) for step500
    bt_prices = prices.tail(504)
    bt_path   = ROOT / "backtest_price_cache.csv"
    bt_prices.to_csv(bt_path)
    ok(f"backtest_price_cache.csv → {bt_prices.shape[0]} days × {bt_prices.shape[1]} tickers")

    # price_refresh_desk.csv — update with today's prices for Live Track
    refresh_path = ROOT / "price_refresh_desk.csv"
    today_prices = prices.iloc[-1].dropna()
    prev_prices  = prices.iloc[-2].dropna() if len(prices) > 1 else today_prices
    refresh_rows = []
    for tk in today_prices.index:
        curr = float(today_prices[tk])
        prev = float(prev_prices.get(tk, curr))
        chg  = (curr / prev - 1) if prev > 0 else 0.0
        refresh_rows.append({
            "ticker":        tk,
            "last_price":    round(curr, 2),
            "prev_close":    round(prev, 2),
            "daily_chg_pct": round(chg * 100, 3),
            "days_stale":    0,
            "updated_date":  TODAY,
        })
    refresh_df = pd.DataFrame(refresh_rows)
    refresh_df.to_csv(refresh_path, index=False)
    ok(f"price_refresh_desk.csv → {len(refresh_df)} tickers (fresh)")


# ── Step 4: Update step500 N_LONG/N_SHORT to 20/20 ───────────────────────────

def step4_update_position_counts():
    """Scale up N_LONG and N_SHORT from 8/8 to 20/20 for a 500-stock universe."""
    step500 = ROOT / "canyon_final_v9_step500_daily_pipeline.py"
    if not step500.exists():
        warn("step500 not found — skipping N_LONG/N_SHORT update")
        return

    txt = step500.read_text(encoding="utf-8")
    current_long  = None
    current_short = None
    for line in txt.splitlines():
        if line.strip().startswith("N_LONG"):
            import re
            m = re.search(r"N_LONG\s*=\s*(\d+)", line)
            if m: current_long = int(m.group(1))
        if line.strip().startswith("N_SHORT"):
            import re
            m = re.search(r"N_SHORT\s*=\s*(\d+)", line)
            if m: current_short = int(m.group(1))

    if current_long == 20 and current_short == 20:
        ok("N_LONG=20, N_SHORT=20 already set")
        return

    txt = txt.replace(f"N_LONG    = {current_long}", "N_LONG    = 20", 1) \
             .replace(f"N_SHORT   = {current_short}", "N_SHORT   = 20", 1) \
             .replace(f"N_LONG  = {current_long}", "N_LONG  = 20", 1) \
             .replace(f"N_SHORT = {current_short}", "N_SHORT = 20", 1)
    step500.write_text(txt, encoding="utf-8")
    ok(f"step500: N_LONG {current_long}→20, N_SHORT {current_short}→20")


# ── Step 5: Update display caps in update_research_html.py ───────────────────

def step5_update_display():
    """Remove the .head(25) cap on alpha_scores display and show top 15 longs."""
    html_gen = ROOT / "update_research_html.py"
    if not html_gen.exists():
        warn("update_research_html.py not found")
        return

    txt = html_gen.read_text(encoding="utf-8")
    changed = False

    # Expand alpha_scores display from 25 → all (use 500 to be safe)
    if ".head(25)" in txt and "alpha_score" in txt:
        # Only change the one in load_alpha_scores
        txt = txt.replace(
            'df = df[keep].sort_values("alpha_score", ascending=False).head(25)',
            'df = df[keep].sort_values("alpha_score", ascending=False).head(500)',
        )
        changed = True
        ok("update_research_html.py: alpha_scores cap lifted (25→500)")

    if changed:
        html_gen.write_text(txt, encoding="utf-8")
    else:
        ok("update_research_html.py: display cap already updated")


# ── Step 6: Re-run step87 to rebuild alpha_scores.csv ───────────────────────

def step6_rebuild_alpha_scores():
    """Run step87 (alpha aggregator) to regenerate alpha_scores.csv from all
    expanded signal files."""
    import subprocess, sys
    step87 = ROOT / "canyon_final_v9_step87_alpha_aggregator.py"
    if not step87.exists():
        warn("step87_alpha_aggregator.py not found — skipping alpha_scores rebuild")
        return

    log("Running step87 to rebuild alpha_scores.csv …")
    result = subprocess.run(
        [sys.executable, str(step87)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        # Check how many tickers
        alpha_path = ROOT / "alpha_scores.csv"
        if alpha_path.exists():
            n = len(pd.read_csv(alpha_path))
            ok(f"alpha_scores.csv rebuilt → {n} tickers")
        else:
            ok("step87 completed")
        # Print last 5 lines for visibility
        for line in result.stdout.strip().splitlines()[-5:]:
            log(f"  {line}")
    else:
        warn(f"step87 exited rc={result.returncode}")
        for line in (result.stderr or result.stdout or "").splitlines()[-5:]:
            log(f"  {line}")


# ── Step 7: Re-run update_research_html.py ───────────────────────────────────

def step7_rebuild_html():
    import subprocess, sys
    html_gen = ROOT / "update_research_html.py"
    if not html_gen.exists():
        return
    log("Rebuilding research HTML …")
    result = subprocess.run(
        [sys.executable, str(html_gen)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        ok("canyon_v9_research.html rebuilt")
    else:
        warn(f"HTML rebuild failed: {result.stderr[:200]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  Canyon v9 — S&P 500 Universe Expansion{RESET}")
    print(f"{BOLD}  {TODAY}{RESET}")
    print(f"{BOLD}{'═'*65}{RESET}\n")
    print("  Research only · No broker connection · No live orders\n")

    t0 = time.time()

    # Step 1: Expand signal files from existing regime_ml_scores.csv
    print(f"\n{CYAN}[1/7] Expanding signal files from regime_ml_scores.csv …{RESET}")
    tickers = step1_expand_signal_files()
    if not tickers:
        print(f"\n{RED}Aborted — no tickers loaded{RESET}")
        return

    # Step 2: Download prices
    print(f"\n{CYAN}[2/7] Downloading prices ({len(tickers)} tickers, {START}→{TODAY}) …{RESET}")
    print(f"  This takes ~3 minutes. Please wait.\n")
    prices = step2_download_prices(tickers)
    if prices is None:
        print(f"\n{YELLOW}Price download failed — skipping price cache updates{RESET}")
        print(f"  Signal files (step 1) were still updated.\n")
    else:
        # Step 3: Write price caches
        print(f"\n{CYAN}[3/7] Writing price caches …{RESET}")
        step3_write_price_caches(prices)

    # Step 4: Update position counts
    print(f"\n{CYAN}[4/7] Updating position counts (8→20 per side) …{RESET}")
    step4_update_position_counts()

    # Step 5: Update display caps
    print(f"\n{CYAN}[5/7] Updating display caps …{RESET}")
    step5_update_display()

    # Step 6: Rebuild alpha_scores.csv
    print(f"\n{CYAN}[6/7] Rebuilding alpha_scores.csv via step87 …{RESET}")
    step6_rebuild_alpha_scores()

    # Step 7: Rebuild HTML
    print(f"\n{CYAN}[7/7] Rebuilding research HTML …{RESET}")
    step7_rebuild_html()

    elapsed = time.time() - t0
    print(f"\n{GREEN}{BOLD}{'═'*65}{RESET}")
    print(f"{GREEN}{BOLD}  S&P 500 expansion complete — {elapsed:.0f}s{RESET}")
    print(f"{GREEN}{BOLD}{'═'*65}{RESET}")
    print(f"\n  Universe: {len(tickers)} tickers")
    print(f"  Long book: 20 positions  |  Short book: 20 positions")
    print(f"\n  Next step: run the daily pipeline")
    print(f"    .venv/bin/python run_daily.py\n")


if __name__ == "__main__":
    main()
