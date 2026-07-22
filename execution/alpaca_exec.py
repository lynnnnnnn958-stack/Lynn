"""
W36-W37: Alpaca Paper Trading Executor (TWAP)
=============================================
Connects to Alpaca paper trading API for order simulation.
No real money is ever traded — this is research paper-tracking only.

Configuration:
  Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment variables or .env file.
  Paper trading endpoint: https://paper-api.alpaca.markets

W36: Connection test + account config
W37: TWAP executor — splits large orders into time-weighted slices

TWAP algorithm:
  Total order is split into N slices (default: N=6 = 30 min slices in 3hr window)
  Each slice: order_size / N shares
  Execution window: first 3 hours of market session (9:30-12:30 ET)
  Rationale: TWAP minimises market timing risk at cost of slightly higher market impact

No broker connection is made if credentials are absent — graceful degradation.

Usage:
    from execution.alpaca_exec import AlpacaExecutor, test_connection
    executor = AlpacaExecutor()
    executor.test_connection()
    executor.submit_twap("AAPL", quantity=100, side="buy", n_slices=6)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent.parent

PAPER_BASE_URL  = "https://paper-api.alpaca.markets"
DATA_BASE_URL   = "https://data.alpaca.markets"
TWAP_N_SLICES   = 6    # number of order slices
TWAP_WINDOW_MIN = 180  # TWAP window in minutes (3 hours)
SLICE_INTERVAL  = TWAP_WINDOW_MIN / TWAP_N_SLICES  # minutes between slices


class AlpacaExecutor:
    """
    Paper trading executor via Alpaca API.

    Only submits orders to paper account — never live trading.
    All orders are logged to execution_log.csv.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key    = api_key    or os.environ.get("ALPACA_API_KEY",    "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._api       = None
        self._connected = False

        # Try to load .env if keys not found
        if not self.api_key:
            env_path = ROOT / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"')
                        if k == "ALPACA_API_KEY":    self.api_key    = v
                        if k == "ALPACA_SECRET_KEY": self.secret_key = v

    def _connect(self) -> bool:
        """Attempt to connect to Alpaca paper API."""
        if self._connected:
            return True
        if not self.api_key or not self.secret_key:
            print("  [Alpaca] No API credentials — paper trading unavailable")
            print("  Set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env or environment")
            return False
        try:
            import alpaca_trade_api as tradeapi
            self._api       = tradeapi.REST(self.api_key, self.secret_key,
                                            PAPER_BASE_URL, api_version="v2")
            acct            = self._api.get_account()
            self._connected = True
            print(f"  [Alpaca] Connected to paper account: {acct.account_number}")
            print(f"    Portfolio value: ${float(acct.portfolio_value):,.2f}")
            print(f"    Buying power:   ${float(acct.buying_power):,.2f}")
            print(f"    Status:         {acct.status}")
            return True
        except ImportError:
            print("  [Alpaca] alpaca-trade-api not installed:")
            print("    pip install alpaca-trade-api")
            return False
        except Exception as e:
            print(f"  [Alpaca] Connection failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test API connection and print account details."""
        return self._connect()

    def get_positions(self) -> pd.DataFrame:
        """Get current paper positions."""
        if not self._connect():
            return pd.DataFrame()
        try:
            positions = self._api.list_positions()
            rows = [{
                "ticker":    p.symbol,
                "qty":       int(p.qty),
                "avg_cost":  float(p.avg_entry_price),
                "market_val": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "side":      p.side,
            } for p in positions]
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"  [Alpaca] get_positions error: {e}")
            return pd.DataFrame()

    def submit_market_order(
        self,
        ticker: str,
        quantity: int,
        side: str,  # "buy" or "sell"
        log: bool = True,
    ) -> Optional[dict]:
        """Submit a single market order."""
        if not self._connect():
            return None
        try:
            order = self._api.submit_order(
                symbol     = ticker,
                qty        = abs(quantity),
                side       = side,
                type       = "market",
                time_in_force = "day",
            )
            result = {
                "order_id":   order.id,
                "ticker":     ticker,
                "qty":        quantity,
                "side":       side,
                "status":     order.status,
                "submitted":  datetime.now().isoformat(),
            }
            if log:
                self._log_order(result)
            print(f"  [Alpaca] Order submitted: {side} {quantity} {ticker} (id={order.id})")
            return result
        except Exception as e:
            print(f"  [Alpaca] Order error for {ticker}: {e}")
            return None

    def submit_twap(
        self,
        ticker: str,
        quantity: int,
        side: str,
        n_slices: int = TWAP_N_SLICES,
        dry_run: bool = True,
    ) -> list[dict]:
        """
        Submit TWAP order: split quantity into n_slices equal market orders.

        Args:
            ticker:   Stock ticker.
            quantity: Total shares to trade.
            side:     "buy" or "sell".
            n_slices: Number of time slices (default: 6 = every 30 min in 3hr window).
            dry_run:  If True, log but do not actually submit orders.

        Returns: list of order results.
        """
        slice_qty    = max(1, quantity // n_slices)
        remainder    = quantity - slice_qty * n_slices
        interval_min = TWAP_WINDOW_MIN / n_slices

        results = []
        print(f"  [TWAP] {side} {quantity} {ticker}: "
              f"{n_slices} slices × {slice_qty} shares, "
              f"every {interval_min:.0f} min")

        for i in range(n_slices):
            qty_i = slice_qty + (remainder if i == n_slices - 1 else 0)

            if dry_run:
                result = {
                    "order_id":  f"DRY_{ticker}_{i}",
                    "ticker":    ticker,
                    "qty":       qty_i,
                    "side":      side,
                    "slice":     i + 1,
                    "status":    "dry_run",
                    "submitted": datetime.now().isoformat(),
                }
                print(f"    [DRY_RUN] Slice {i+1}/{n_slices}: {side} {qty_i} {ticker}")
            else:
                result = self.submit_market_order(ticker, qty_i, side, log=False)
                if result:
                    result["slice"] = i + 1
                else:
                    break

            results.append(result)
            self._log_order(result)

            # Wait for next slice (except on last slice)
            if i < n_slices - 1 and not dry_run:
                time.sleep(interval_min * 60)

        return results

    def _log_order(self, order: dict) -> None:
        """Append order to execution_log.csv."""
        log_path = ROOT / "execution_log.csv"
        row_df   = pd.DataFrame([order])
        if log_path.exists():
            row_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(log_path, index=False)


def test_connection() -> bool:
    """Quick connection test (no credentials needed for basic test)."""
    executor = AlpacaExecutor()
    return executor.test_connection()


if __name__ == "__main__":
    print("W36: Alpaca Paper Trading Connection Test")
    print("=" * 50)
    executor = AlpacaExecutor()
    connected = executor.test_connection()
    if not connected:
        print("\n  To enable paper trading:")
        print("  1. Create free account at alpaca.markets")
        print("  2. Generate paper trading API keys")
        print("  3. Add to .env file:")
        print("     ALPACA_API_KEY=your_key")
        print("     ALPACA_SECRET_KEY=your_secret")
    else:
        positions = executor.get_positions()
        print(f"\n  Current positions: {len(positions)}")

    print("\nW37: TWAP Order Test (dry run)")
    executor.submit_twap("AAPL", quantity=100, side="buy", n_slices=6, dry_run=True)
