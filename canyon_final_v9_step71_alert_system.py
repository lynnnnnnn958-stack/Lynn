"""
Canyon v9 — Step 71: Alert Monitor System
Scans output CSVs and fires alerts when conditions are met.
Usage:
    python3 step71_alert_system.py            # run all checks, print alerts
    python3 step71_alert_system.py --summary  # show count of active alerts by type
    python3 step71_alert_system.py --clear    # archive old alerts, keep last 24h
"""

import argparse
import csv
import os
import subprocess
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

ML_ENHANCED     = os.path.join(BASE, "enhanced_ml_scores.csv")
ML_SIGNAL       = os.path.join(BASE, "ml_signal_scores.csv")
POSITIONS_CSV   = os.path.join(BASE, "paper_sim_positions.csv")
EARNINGS_CSV    = os.path.join(BASE, "earnings_calendar.csv")
ALERTS_CSV      = os.path.join(BASE, "alerts.csv")
ALERTS_ARCHIVE  = os.path.join(BASE, "alerts_archive.csv")
ALERTS_REPORT   = os.path.join(BASE, "alerts_report.md")

ALERTS_COLS = ["timestamp", "alert_type", "severity", "ticker", "message", "value", "threshold"]

SEVERITY_MAP = {
    "STOP_LOSS_NEAR":   "CRITICAL",
    "TARGET_NEAR":      "INFO",
    "DRAWDOWN_WARN":    "CRITICAL",
    "ML_SCORE_DROPPED": "WARNING",
    "EARNINGS_BLACKOUT":"WARNING",
    "ML_HIGH_SIGNAL":   "INFO",
    "SECTOR_OVER":      "WARNING",
    "REGIME_CHANGE":    "WARNING",
}

DEDUP_HOURS = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str, label: str) -> pd.DataFrame | None:
    """Load CSV safely; return None and print a warning on failure."""
    if not os.path.exists(path):
        print(f"[WARN] {label}: file not found ({path}) — skipping check.")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] {label}: file is empty — skipping check.")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] {label}: could not read ({exc}) — skipping check.")
        return None


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch last close prices for a list of tickers via yfinance."""
    prices: dict[str, float] = {}
    if not tickers:
        return prices
    try:
        data = yf.download(tickers, period="2d", progress=False, auto_adjust=True)
        close = data["Close"] if "Close" in data.columns else data.get("Adj Close", pd.DataFrame())
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        for t in tickers:
            if t in close.columns:
                val = close[t].dropna()
                if not val.empty:
                    prices[t] = float(val.iloc[-1])
    except Exception as exc:
        print(f"[WARN] yfinance price fetch failed: {exc}")
    return prices


def _make_alert(alert_type: str, ticker: str, message: str, value: float, threshold: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alert_type": alert_type,
        "severity": SEVERITY_MAP.get(alert_type, "INFO"),
        "ticker": ticker,
        "message": message,
        "value": round(value, 6),
        "threshold": threshold,
    }


# ── AlertEngine ───────────────────────────────────────────────────────────────

class AlertEngine:

    def __init__(self):
        self._existing: pd.DataFrame = self._load_existing_alerts()

    # ── dedup ──────────────────────────────────────────────────────────────────

    def _load_existing_alerts(self) -> pd.DataFrame:
        if not os.path.exists(ALERTS_CSV):
            return pd.DataFrame(columns=ALERTS_COLS)
        try:
            df = pd.read_csv(ALERTS_CSV)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=ALERTS_COLS)

    def is_duplicate(self, alert_type: str, ticker: str) -> bool:
        if self._existing.empty:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(hours=DEDUP_HOURS)
        mask = (
            (self._existing["alert_type"] == alert_type)
            & (self._existing["ticker"] == ticker)
            & (self._existing["timestamp"] >= cutoff)
        )
        return bool(mask.any())

    # ── check_ml_signals ──────────────────────────────────────────────────────

    def check_ml_signals(self) -> list[dict]:
        alerts: list[dict] = []

        # Prefer enhanced_ml_scores; fall back to ml_signal_scores
        df = _load_csv(ML_ENHANCED, "enhanced_ml_scores")
        score_col = "enhanced_score"

        if df is None:
            df = _load_csv(ML_SIGNAL, "ml_signal_scores")
            score_col = "ensemble_score"

        if df is None:
            return alerts

        # Keep only the most recent run per ticker
        if "rebalance_date" in df.columns:
            df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], errors="coerce")
            df = df.sort_values("rebalance_date").groupby("ticker").last().reset_index()

        if score_col not in df.columns:
            print(f"[WARN] ML score column '{score_col}' missing — skipping ML_HIGH_SIGNAL.")
            return alerts

        threshold = 0.70
        high = df[df[score_col] > threshold]
        for _, row in high.iterrows():
            ticker = str(row.get("ticker", "UNKNOWN"))
            score  = float(row[score_col])
            if self.is_duplicate("ML_HIGH_SIGNAL", ticker):
                continue
            alerts.append(_make_alert(
                "ML_HIGH_SIGNAL", ticker,
                f"{ticker} ML score {score:.3f} exceeds {threshold}",
                score, threshold
            ))

        # ML_SCORE_DROPPED: compare current scores for position tickers vs previous alerts
        pos_df = _load_csv(POSITIONS_CSV, "paper_sim_positions (ML drop check)")
        if pos_df is not None and "ticker" in pos_df.columns and score_col in df.columns:
            pos_tickers = pos_df["ticker"].astype(str).unique().tolist()
            drop_threshold = 0.45
            for t in pos_tickers:
                row = df[df["ticker"] == t]
                if row.empty:
                    continue
                current_score = float(row[score_col].iloc[-1])
                if current_score < drop_threshold:
                    if self.is_duplicate("ML_SCORE_DROPPED", t):
                        continue
                    alerts.append(_make_alert(
                        "ML_SCORE_DROPPED", t,
                        f"{t} ML score dropped to {current_score:.3f} (below {drop_threshold})",
                        current_score, drop_threshold
                    ))

        return alerts

    # ── check_positions ───────────────────────────────────────────────────────

    def check_positions(self) -> list[dict]:
        alerts: list[dict] = []
        df = _load_csv(POSITIONS_CSV, "paper_sim_positions")
        if df is None:
            return alerts

        required = {"ticker", "stop_price", "target_price", "cost_basis", "shares"}
        if not required.issubset(df.columns):
            print(f"[WARN] paper_sim_positions missing columns: {required - set(df.columns)}")
            return alerts

        tickers = df["ticker"].astype(str).tolist()
        prices  = _fetch_prices(tickers)

        stop_pct_thresh   = 0.02   # within 2% of stop
        target_pct_thresh = 0.05   # within 5% of target

        for _, row in df.iterrows():
            ticker      = str(row["ticker"])
            stop_price  = float(row.get("stop_price", 0) or 0)
            target_price = float(row.get("target_price", 0) or 0)
            current     = prices.get(ticker)

            if current is None or current <= 0:
                continue

            # STOP_LOSS_NEAR
            if stop_price > 0:
                dist_pct = (current - stop_price) / current
                if 0 <= dist_pct <= stop_pct_thresh:
                    if not self.is_duplicate("STOP_LOSS_NEAR", ticker):
                        alerts.append(_make_alert(
                            "STOP_LOSS_NEAR", ticker,
                            f"{ticker} price {current:.2f} within {dist_pct*100:.1f}% of stop {stop_price:.2f}",
                            dist_pct, stop_pct_thresh
                        ))

            # TARGET_NEAR
            if target_price > 0:
                dist_pct = (target_price - current) / current
                if 0 <= dist_pct <= target_pct_thresh:
                    if not self.is_duplicate("TARGET_NEAR", ticker):
                        alerts.append(_make_alert(
                            "TARGET_NEAR", ticker,
                            f"{ticker} price {current:.2f} within {dist_pct*100:.1f}% of target {target_price:.2f}",
                            dist_pct, target_pct_thresh
                        ))

        # DRAWDOWN_WARN
        total_cost  = df["cost_basis"].astype(float).sum()
        total_shares = df["shares"].astype(float)
        market_vals = pd.Series([
            (prices.get(str(row["ticker"]), float(row.get("entry_price", 0))) * float(row["shares"]))
            for _, row in df.iterrows()
        ])
        total_market = market_vals.sum()
        if total_cost > 0:
            unreal_pnl_pct = (total_market - total_cost) / total_cost
            threshold = -0.05
            if unreal_pnl_pct < threshold:
                if not self.is_duplicate("DRAWDOWN_WARN", "PORTFOLIO"):
                    alerts.append(_make_alert(
                        "DRAWDOWN_WARN", "PORTFOLIO",
                        f"Portfolio unrealised P&L {unreal_pnl_pct*100:.1f}% below {threshold*100:.0f}% threshold",
                        unreal_pnl_pct, threshold
                    ))

        # EARNINGS_BLACKOUT
        earn_df = _load_csv(EARNINGS_CSV, "earnings_calendar")
        if earn_df is not None and "ticker" in earn_df.columns and "next_earnings" in earn_df.columns:
            pos_tickers = set(df["ticker"].astype(str).tolist())
            today = pd.Timestamp.now(tz="UTC").normalize()
            earn_df["next_earnings"] = pd.to_datetime(earn_df["next_earnings"], errors="coerce", utc=True)
            for _, erow in earn_df.iterrows():
                t = str(erow["ticker"])
                if t not in pos_tickers:
                    continue
                edate = erow["next_earnings"]
                if pd.isna(edate):
                    continue
                bdays = np.busday_count(today.date(), edate.date())
                if 1 <= bdays <= 3:
                    if not self.is_duplicate("EARNINGS_BLACKOUT", t):
                        alerts.append(_make_alert(
                            "EARNINGS_BLACKOUT", t,
                            f"{t} earnings in {bdays} business day(s) — blackout window active",
                            float(bdays), 3.0
                        ))

        return alerts

    # ── check_sector ──────────────────────────────────────────────────────────

    def check_sector(self) -> list[dict]:
        alerts: list[dict] = []
        df = _load_csv(POSITIONS_CSV, "paper_sim_positions (sector check)")
        if df is None:
            return alerts

        if "sector" not in df.columns or "cost_basis" not in df.columns:
            return alerts

        df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce").fillna(0)
        total = df["cost_basis"].sum()
        if total <= 0:
            return alerts

        sector_weights = df.groupby("sector")["cost_basis"].sum() / total
        threshold = 0.35
        for sector, weight in sector_weights.items():
            if weight > threshold:
                label = str(sector)
                if not self.is_duplicate("SECTOR_OVER", label):
                    alerts.append(_make_alert(
                        "SECTOR_OVER", label,
                        f"Sector '{label}' weight {weight*100:.1f}% exceeds {threshold*100:.0f}% limit",
                        weight, threshold
                    ))

        return alerts

    # ── check_regime ──────────────────────────────────────────────────────────

    def check_regime(self) -> list[dict]:
        alerts: list[dict] = []
        try:
            spy = yf.download("SPY", period="300d", progress=False, auto_adjust=True)
            if spy.empty:
                print("[WARN] Could not fetch SPY data — skipping REGIME_CHANGE check.")
                return alerts

            close = spy["Close"].dropna()
            if len(close) < 200:
                print("[WARN] Insufficient SPY history for MA cross — skipping.")
                return alerts

            ma50  = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()

            prev_diff = float(ma50.iloc[-2] - ma200.iloc[-2])
            curr_diff = float(ma50.iloc[-1] - ma200.iloc[-1])

            if prev_diff < 0 and curr_diff >= 0:
                cross_type = "GOLDEN CROSS (50MA crossed above 200MA)"
            elif prev_diff > 0 and curr_diff <= 0:
                cross_type = "DEATH CROSS (50MA crossed below 200MA)"
            else:
                return alerts

            if not self.is_duplicate("REGIME_CHANGE", "SPY"):
                alerts.append(_make_alert(
                    "REGIME_CHANGE", "SPY",
                    f"SPY regime change detected: {cross_type}",
                    round(curr_diff, 4), 0.0
                ))
        except Exception as exc:
            print(f"[WARN] REGIME_CHANGE check failed: {exc}")

        return alerts

    # ── run_all_checks ────────────────────────────────────────────────────────

    def run_all_checks(self) -> list[dict]:
        all_alerts: list[dict] = []
        print("[INFO] Running ML signal checks…")
        all_alerts.extend(self.check_ml_signals())
        print("[INFO] Running position checks…")
        all_alerts.extend(self.check_positions())
        print("[INFO] Running sector checks…")
        all_alerts.extend(self.check_sector())
        print("[INFO] Running regime checks…")
        all_alerts.extend(self.check_regime())
        return all_alerts

    # ── fire_alerts ───────────────────────────────────────────────────────────

    def fire_alerts(self, alerts: list[dict]) -> None:
        if not alerts:
            print("[INFO] No new alerts to fire.")
            return

        # Append to CSV
        existing_rows: list[dict] = []
        if os.path.exists(ALERTS_CSV):
            try:
                with open(ALERTS_CSV, newline="") as f:
                    existing_rows = list(csv.DictReader(f))
            except Exception:
                pass

        all_rows = existing_rows + alerts
        with open(ALERTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALERTS_COLS)
            writer.writeheader()
            writer.writerows(all_rows)

        # Write markdown report
        self._write_report(all_rows)

        # Print and notify for CRITICAL
        for a in alerts:
            sev   = a["severity"]
            atype = a["alert_type"]
            msg   = a["message"]
            print(f"  [{sev}] {atype} | {msg}")
            if sev == "CRITICAL":
                self._notify(atype, msg)

        print(f"[INFO] {len(alerts)} alert(s) written to {ALERTS_CSV}")

    def _notify(self, alert_type: str, msg: str) -> None:
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{msg}" with title "Canyon Alert: {alert_type}"'],
                timeout=5, check=False
            )
        except Exception as exc:
            print(f"[WARN] macOS notification failed: {exc}")

    def _write_report(self, rows: list[dict]) -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Canyon v9 — Alert Report",
            f"_Generated: {now_str}_",
            "",
            f"**Total alerts on file:** {len(rows)}",
            "",
            "| Timestamp | Type | Severity | Ticker | Message | Value | Threshold |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows[-50:]:  # last 50 rows for readability
            lines.append(
                f"| {r.get('timestamp','')} | {r.get('alert_type','')} "
                f"| {r.get('severity','')} | {r.get('ticker','')} "
                f"| {r.get('message','')} | {r.get('value','')} | {r.get('threshold','')} |"
            )
        with open(ALERTS_REPORT, "w") as f:
            f.write("\n".join(lines) + "\n")


# ── CLI helpers ───────────────────────────────────────────────────────────────

def cmd_summary() -> None:
    """Print count of active alerts by type."""
    if not os.path.exists(ALERTS_CSV):
        print("No alerts file found.")
        return
    try:
        df = pd.read_csv(ALERTS_CSV)
        if df.empty:
            print("alerts.csv is empty.")
            return
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        recent = df[df["timestamp"] >= cutoff]
        if recent.empty:
            print("No alerts in the last 24 hours.")
            return
        counts = recent["alert_type"].value_counts()
        print("\n=== Active Alerts (last 24h) ===")
        for atype, cnt in counts.items():
            print(f"  {atype}: {cnt}")
        print(f"  TOTAL: {len(recent)}")
    except Exception as exc:
        print(f"[ERROR] Could not read alerts: {exc}")


def cmd_clear() -> None:
    """Archive alerts older than 24h; keep recent ones in alerts.csv."""
    if not os.path.exists(ALERTS_CSV):
        print("No alerts file to clear.")
        return
    try:
        df = pd.read_csv(ALERTS_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        old    = df[df["timestamp"] < cutoff]
        recent = df[df["timestamp"] >= cutoff]

        if not old.empty:
            archive_exists = os.path.exists(ALERTS_ARCHIVE)
            old.to_csv(ALERTS_ARCHIVE, mode="a", header=not archive_exists, index=False)
            print(f"[INFO] Archived {len(old)} old alert(s) to {ALERTS_ARCHIVE}")

        recent.to_csv(ALERTS_CSV, index=False)
        print(f"[INFO] {len(recent)} recent alert(s) kept in {ALERTS_CSV}")
    except Exception as exc:
        print(f"[ERROR] Clear failed: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Canyon v9 Alert Monitor")
    parser.add_argument("--summary", action="store_true", help="Show alert counts by type")
    parser.add_argument("--clear",   action="store_true", help="Archive old alerts, keep last 24h")
    args = parser.parse_args()

    if args.summary:
        cmd_summary()
        return

    if args.clear:
        cmd_clear()
        return

    # Default: run all checks
    engine = AlertEngine()
    alerts = engine.run_all_checks()
    engine.fire_alerts(alerts)

    if not alerts:
        print("[INFO] All clear — no new alerts.")
    else:
        criticals = [a for a in alerts if a["severity"] == "CRITICAL"]
        if criticals:
            print(f"\n[!] {len(criticals)} CRITICAL alert(s) require immediate attention.")


if __name__ == "__main__":
    main()
