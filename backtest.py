"""
backtest.py — Simulation engine and reporting.

The engine iterates over each trading day, asks the strategy for signals,
simulates fills with slippage, tracks stops/targets bar-by-bar, and
produces a full statistical report.

Design note
-----------
BacktestEngine only calls public methods on ORBStrategy and operates on plain
DataFrames.  Swapping the strategy is as easy as passing a different object
that exposes identify_opening_range() and scan_for_signals().
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy import ORBStrategy, Trade, TradeSignal

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtester for intraday strategies.

    Parameters
    ----------
    strategy       : An ORBStrategy instance (or any object with the same interface).
    config         : Dict from BACKTEST_CONFIG in config.py.
    """

    def __init__(self, strategy: ORBStrategy, config: dict):
        self.strategy        = strategy
        self.lot_size        = config.get("lot_size", 50)
        self.initial_capital = config.get("initial_capital", 100_000)
        self.brokerage       = config.get("brokerage_per_trade", 40)
        self.slippage        = config.get("slippage_points", 2)
        self.trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Single-day simulation
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        day_data: pd.DataFrame,
        trade_date: date,
        allowed_directions: Optional[set] = None,
    ) -> List[Trade]:
        day_trades: List[Trade] = []

        or_result = self.strategy.identify_opening_range(day_data)
        if or_result is None:
            logger.debug("Skipping %s — no opening range data", trade_date)
            return day_trades

        oh, ol = or_result
        logger.debug("OR for %s:  High=%.2f  Low=%.2f", trade_date, oh, ol)

        signals = self.strategy.scan_for_signals(day_data, oh, ol)
        if not signals:
            logger.debug("No signals on %s", trade_date)
            return day_trades

        # VIX direction filter: drop signals outside the allowed set
        if allowed_directions is not None:
            signals = [s for s in signals if s.direction in allowed_directions]
            if not signals:
                logger.debug(
                    "No signals on %s after VIX direction filter (allowed: %s)",
                    trade_date, allowed_directions,
                )
                return day_trades

        # Simulation data: everything from trading_start onward
        window = day_data[
            day_data["datetime"].dt.time >= self.strategy.trading_start
        ].copy().reset_index(drop=True)

        for signal in signals:
            trade = self._simulate_trade(signal, window)
            if trade:
                day_trades.append(trade)
                logger.debug(
                    "%s %s  entry=%.2f  exit=%.2f  pnl_pts=%.2f  reason=%s",
                    trade_date, signal.direction,
                    trade.entry_price, trade.exit_price or 0,
                    trade.pnl_points, trade.exit_reason,
                )

        return day_trades

    def _simulate_trade(
        self, signal: TradeSignal, window: pd.DataFrame
    ) -> Optional[Trade]:
        """Simulate one trade from signal to exit, bar by bar."""

        # Apply entry slippage
        if signal.direction == "LONG":
            entry_px = signal.entry_price + self.slippage
        else:
            entry_px = signal.entry_price - self.slippage

        trade = Trade(
            signal      = signal,
            entry_time  = signal.signal_time,
            entry_price = entry_px,
            lot_size    = self.lot_size,
        )

        # Find the bar immediately after the signal candle
        after_signal = window[window["datetime"] > signal.signal_time]
        if after_signal.empty:
            return None

        start_idx = after_signal.index[0]

        for i in range(start_idx, len(window)):
            bar      = window.iloc[i]
            bar_time = bar["datetime"].time()

            # Force-close at end of session
            if bar_time >= self.strategy.trading_end:
                trade.exit_time   = bar["datetime"]
                trade.exit_price  = bar["open"]   # exit at next bar's open
                trade.exit_reason = "eod"
                break

            if signal.direction == "LONG":
                # Stop hit first (conservative — use bar low)
                if bar["low"] <= signal.stop_price:
                    trade.exit_time   = bar["datetime"]
                    trade.exit_price  = signal.stop_price - self.slippage
                    trade.exit_reason = "stop"
                    break
                # Target hit
                if bar["high"] >= signal.target_price:
                    trade.exit_time   = bar["datetime"]
                    trade.exit_price  = signal.target_price - self.slippage
                    trade.exit_reason = "target"
                    break
            else:  # SHORT
                if bar["high"] >= signal.stop_price:
                    trade.exit_time   = bar["datetime"]
                    trade.exit_price  = signal.stop_price + self.slippage
                    trade.exit_reason = "stop"
                    break
                if bar["low"] <= signal.target_price:
                    trade.exit_time   = bar["datetime"]
                    trade.exit_price  = signal.target_price + self.slippage
                    trade.exit_reason = "target"
                    break

        # Fallback EOD: if the loop exhausted all bars with no exit
        # (e.g. signal fired in the last minute of the session)
        if trade.exit_price is None and not window.empty:
            last_bar = window.iloc[-1]
            trade.exit_time   = last_bar["datetime"]
            trade.exit_price  = last_bar["close"]
            trade.exit_reason = "eod"

        # P&L
        if trade.exit_price is not None:
            if signal.direction == "LONG":
                trade.pnl_points = trade.exit_price - trade.entry_price
            else:
                trade.pnl_points = trade.entry_price - trade.exit_price

            trade.pnl_amount = trade.pnl_points * self.lot_size - self.brokerage

        return trade

    # ------------------------------------------------------------------
    # Full backtest run
    # ------------------------------------------------------------------

    def run(
        self,
        all_data: pd.DataFrame,
        start_date: date,
        end_date: date,
        vix_data: Optional["pd.DataFrame"] = None,
        vix_config: Optional[dict] = None,
    ) -> Dict:
        """
        Iterate over every trading day in [start_date, end_date] and collect
        all trades. Returns a comprehensive results dictionary.
        """
        logger.info("Backtest: %s → %s", start_date, end_date)
        self.trades = []

        all_data = all_data.copy()
        all_data["_date"] = all_data["datetime"].dt.date

        trading_days = sorted(
            d for d in all_data["_date"].unique() if start_date <= d <= end_date
        )
        logger.info("Trading days in range: %d", len(trading_days))

        use_vix = (
            vix_data is not None
            and not vix_data.empty
            and vix_config
            and vix_config.get("enabled")
        )
        min_vix = float(vix_config.get("min_vix", 0))     if vix_config else 0
        max_vix = float(vix_config.get("max_vix", 9999))  if vix_config else 9999
        dir_filter = bool(vix_config.get("direction_filter")) if vix_config else False

        for day in trading_days:
            allowed_directions: Optional[set] = None

            if use_vix:
                day_row = vix_data[vix_data["date"] == day]
                if not day_row.empty:
                    vix_level = float(day_row.iloc[0]["vix"])

                    if not (min_vix <= vix_level <= max_vix):
                        logger.info(
                            "Skipping %s — India VIX %.2f outside [%.1f, %.1f]",
                            day, vix_level, min_vix, max_vix,
                        )
                        continue

                    if dir_filter:
                        prev_rows = vix_data[vix_data["date"] < day]
                        if not prev_rows.empty:
                            prev_vix = float(prev_rows.iloc[-1]["vix"])
                            if vix_level > prev_vix:
                                allowed_directions = {"SHORT"}
                            elif vix_level < prev_vix:
                                allowed_directions = {"LONG"}
                            logger.debug(
                                "%s  VIX=%.2f (prev=%.2f) → %s",
                                day, vix_level, prev_vix,
                                next(iter(allowed_directions)) if allowed_directions else "BOTH",
                            )
                else:
                    logger.debug("No VIX data for %s — no filter applied", day)

            day_df = all_data[all_data["_date"] == day].drop(columns="_date")
            self.trades.extend(
                self._simulate_day(day_df, day, allowed_directions=allowed_directions)
            )

        return self._build_report()

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _build_report(self) -> Dict:
        if not self.trades:
            logger.warning("No trades generated — report empty")
            return {"error": "No trades generated in the specified date range."}

        trades      = self.trades
        n           = len(trades)
        winners     = [t for t in trades if t.pnl_points > 0]
        losers      = [t for t in trades if t.pnl_points <= 0]
        win_rate    = len(winners) / n * 100

        pnl_pts   = sum(t.pnl_points for t in trades)
        pnl_inr   = sum(t.pnl_amount for t in trades)

        gross_win  = sum(t.pnl_amount for t in winners if t.pnl_amount > 0)
        gross_loss = abs(sum(t.pnl_amount for t in losers if t.pnl_amount < 0))
        pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

        cum_pnl    = np.cumsum([t.pnl_amount for t in trades])
        running_hi = np.maximum.accumulate(cum_pnl)
        drawdown   = cum_pnl - running_hi
        max_dd     = abs(float(drawdown.min())) if len(drawdown) else 0.0

        avg_win  = float(np.mean([t.pnl_points for t in winners])) if winners else 0.0
        avg_loss = float(np.mean([t.pnl_points for t in losers]))  if losers  else 0.0
        exp_val  = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

        exits = {
            "target": sum(1 for t in trades if t.exit_reason == "target"),
            "stop":   sum(1 for t in trades if t.exit_reason == "stop"),
            "eod":    sum(1 for t in trades if t.exit_reason == "eod"),
        }

        # Monthly breakdown
        monthly: Dict[str, dict] = {}
        for t in trades:
            key = t.entry_time.strftime("%Y-%m")
            if key not in monthly:
                monthly[key] = {"trades": 0, "wins": 0, "pnl_points": 0.0, "pnl_amount": 0.0}
            monthly[key]["trades"]     += 1
            monthly[key]["wins"]       += 1 if t.pnl_points > 0 else 0
            monthly[key]["pnl_points"] += t.pnl_points
            monthly[key]["pnl_amount"] += t.pnl_amount

        return {
            # Summary stats
            "total_trades":    n,
            "winning_trades":  len(winners),
            "losing_trades":   len(losers),
            "win_rate":        round(win_rate, 2),
            "long_trades":     sum(1 for t in trades if t.signal.direction == "LONG"),
            "short_trades":    sum(1 for t in trades if t.signal.direction == "SHORT"),

            # P&L
            "total_pnl_points": round(pnl_pts, 2),
            "total_pnl_amount": round(pnl_inr, 2),
            "avg_win_points":   round(avg_win, 2),
            "avg_loss_points":  round(avg_loss, 2),
            "expectancy":       round(exp_val, 2),
            "profit_factor":    round(pf, 2),

            # Capital
            "initial_capital":  self.initial_capital,
            "final_capital":    round(self.initial_capital + pnl_inr, 2),
            "return_pct":       round(pnl_inr / self.initial_capital * 100, 2),
            "max_drawdown":     round(max_dd, 2),
            "max_drawdown_pct": round(max_dd / self.initial_capital * 100, 2),

            # Breakdowns
            "exit_reasons": exits,
            "monthly":      monthly,

            # Raw trade objects (for CSV export / custom analysis)
            "trades": trades,
        }


# ---------------------------------------------------------------------------
# Convenience: export results to CSV
# ---------------------------------------------------------------------------

def export_trades_csv(report: Dict, filename: str) -> None:
    """
    Flatten the trade list in `report` into a CSV file.

    Columns include all signal metadata so the file is self-contained for
    post-analysis in Excel / Jupyter.
    """
    if "error" in report or not report.get("trades"):
        logger.warning("Nothing to export")
        return

    rows = []
    for t in report["trades"]:
        s = t.signal
        rows.append({
            "date":             t.entry_time.date(),
            "direction":        s.direction,
            "entry_time":       t.entry_time,
            "entry_price":      t.entry_price,
            "exit_time":        t.exit_time,
            "exit_price":       t.exit_price,
            "stop_price":       s.stop_price,
            "target_price":     s.target_price,
            "risk_points":      round(s.risk_points, 2),
            "pnl_points":       round(t.pnl_points, 2),
            "pnl_amount":       round(t.pnl_amount, 2),
            "exit_reason":      t.exit_reason,
            "lot_size":         t.lot_size,
            "fvg_direction":    s.fvg.direction,
            "fvg_gap_high":     s.fvg.gap_high,
            "fvg_gap_low":      s.fvg.gap_low,
            "engulfing_volume": s.engulfing_volume,
            "avg_volume":       round(s.avg_volume, 2),
            "volume_ratio":     round(s.engulfing_volume / s.avg_volume, 2)
                                if s.avg_volume else 0,
        })

    pd.DataFrame(rows).to_csv(filename, index=False)
    logger.info("Trade log saved → %s", filename)
