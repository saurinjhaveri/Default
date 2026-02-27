"""
strategy.py — Core trading logic for the NSE Opening Range Breakout strategy.

Full Rule Set
-------------
Opening range:
  • The 15-minute candle from 9:15 AM to 9:30 AM defines Opening High (OH)
    and Opening Low (OL).

Trade signal (all 5 conditions must be met):
  1. Price Break  — a 1-min close breaks above OH (bullish) or below OL (bearish).
  2. FVG formed   — the 3-candle move that caused the break contains a Fair Value Gap
                    (gap between candle-1's high/low and candle+1's low/high).
  3. FVG Retest   — price pulls back into the FVG zone.
  4. Engulfing    — a candle inside the FVG zone fully engulfs the prior candle's body
                    in the direction of the original breakout.
  5. Volume       — the engulfing candle's volume ≥ 1.25× its rolling average.

Trade management:
  • Entry  : close of the confirmed engulfing candle.
  • Stop   : low of FVG candle-1 (longs) / high of FVG candle-1 (shorts).
  • Target : entry ± risk × RR (default 3:1).

Only one trade per direction per day.
"""

import logging
from dataclasses import dataclass, field
from datetime import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FairValueGap:
    """Three-candle impulse pattern that leaves an unfilled price gap."""
    direction:    str    # "bullish" | "bearish"
    gap_high:     float  # Upper edge of the FVG zone
    gap_low:      float  # Lower edge of the FVG zone
    # Indices inside the *trading window* DataFrame
    c1_idx:       int
    c2_idx:       int
    c3_idx:       int
    # Key levels of the first candle (used for stop placement)
    c1_high:      float
    c1_low:       float
    formed_at:    pd.Timestamp


@dataclass
class TradeSignal:
    """Fully-qualified trade signal ready for execution."""
    direction:        str            # "LONG" | "SHORT"
    entry_price:      float
    stop_price:       float
    target_price:     float
    risk_points:      float
    reward_points:    float
    signal_time:      pd.Timestamp
    fvg:              FairValueGap
    engulfing_idx:    int
    engulfing_volume: float
    avg_volume:       float


@dataclass
class Trade:
    """Executed trade with outcome recorded by the backtest engine."""
    signal:       TradeSignal
    entry_time:   pd.Timestamp
    entry_price:  float
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    exit_reason:  Optional[str]          = None  # "target" | "stop" | "eod"
    pnl_points:   float                  = 0.0
    pnl_amount:   float                  = 0.0
    lot_size:     int                    = 1


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class ORBStrategy:
    """
    Opening Range Breakout strategy with FVG + volume confirmation.

    Parameters (passed via config dict, keys mirror TRADING_CONFIG in config.py):
      opening_range_start  : str  "HH:MM"
      opening_range_end    : str  "HH:MM"
      trading_start        : str  "HH:MM"
      trading_end          : str  "HH:MM"
      risk_reward_ratio    : float
      volume_multiplier    : float
      volume_lookback      : int
    """

    def __init__(self, config: dict):
        def _t(key: str) -> time:
            h, m = map(int, config[key].split(":"))
            return time(h, m)

        self.opening_start    = _t("opening_range_start")   # 09:15
        self.opening_end      = _t("opening_range_end")     # 09:30
        self.trading_start    = _t("trading_start")         # 09:30
        self.trading_end      = _t("trading_end")           # 15:00
        self.rr_ratio         = float(config.get("risk_reward_ratio", 3.0))
        self.vol_multiplier   = float(config.get("volume_multiplier", 1.25))
        self.vol_lookback     = int(config.get("volume_lookback", 20))

    # ------------------------------------------------------------------
    # Step 1: Opening Range
    # ------------------------------------------------------------------

    def identify_opening_range(
        self, day_data: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """
        Extract the 15-minute opening candle from 9:15 to 9:30 AM.

        Returns (opening_high, opening_low) or None when data is absent.
        """
        mask = (
            (day_data["datetime"].dt.time >= self.opening_start)
            & (day_data["datetime"].dt.time < self.opening_end)
        )
        opening = day_data[mask]

        if opening.empty:
            logger.debug("Opening range data missing for this day")
            return None

        return opening["high"].max(), opening["low"].min()

    # ------------------------------------------------------------------
    # Step 2: FVG detection
    # ------------------------------------------------------------------

    def _detect_fvg(
        self, bars: pd.DataFrame, idx: int, direction: str
    ) -> Optional[FairValueGap]:
        """
        Test whether the three bars ending at `idx` form an FVG.

        Bullish FVG : bars[idx-2].high  <  bars[idx].low
        Bearish FVG : bars[idx-2].low   >  bars[idx].high
        """
        if idx < 2:
            return None

        c1 = bars.iloc[idx - 2]
        c3 = bars.iloc[idx]

        if direction == "bullish" and c1["high"] < c3["low"]:
            return FairValueGap(
                direction="bullish",
                gap_high=c3["low"],
                gap_low=c1["high"],
                c1_idx=idx - 2,
                c2_idx=idx - 1,
                c3_idx=idx,
                c1_high=c1["high"],
                c1_low=c1["low"],
                formed_at=c3["datetime"],
            )

        if direction == "bearish" and c1["low"] > c3["high"]:
            return FairValueGap(
                direction="bearish",
                gap_high=c1["low"],
                gap_low=c3["high"],
                c1_idx=idx - 2,
                c2_idx=idx - 1,
                c3_idx=idx,
                c1_high=c1["high"],
                c1_low=c1["low"],
                formed_at=c3["datetime"],
            )

        return None

    # ------------------------------------------------------------------
    # Step 3: FVG retest
    # ------------------------------------------------------------------

    def _is_retest(self, bar: pd.Series, fvg: FairValueGap) -> bool:
        """Return True when `bar` overlaps the FVG zone (price has pulled back)."""
        if fvg.direction == "bullish":
            return bar["low"] <= fvg.gap_high and bar["high"] >= fvg.gap_low
        else:
            return bar["high"] >= fvg.gap_low and bar["low"] <= fvg.gap_high

    # ------------------------------------------------------------------
    # Step 4: Engulfing candle
    # ------------------------------------------------------------------

    def _is_engulfing(
        self, prev: pd.Series, curr: pd.Series, direction: str
    ) -> bool:
        """
        True when `curr` is a full-body engulfing candle in `direction`.

        Bullish engulfing:
          curr is a green candle whose body fully covers prev's body.
        Bearish engulfing:
          curr is a red candle whose body fully covers prev's body.
        """
        prev_hi = max(prev["open"], prev["close"])
        prev_lo = min(prev["open"], prev["close"])
        curr_hi = max(curr["open"], curr["close"])
        curr_lo = min(curr["open"], curr["close"])

        if direction == "bullish":
            return (
                curr["close"] > curr["open"]   # green body
                and curr_hi >= prev_hi
                and curr_lo <= prev_lo
            )
        else:
            return (
                curr["close"] < curr["open"]   # red body
                and curr_hi >= prev_hi
                and curr_lo <= prev_lo
            )

    # ------------------------------------------------------------------
    # Step 5: Volume filter
    # ------------------------------------------------------------------

    def _rolling_avg_volume(self, bars: pd.DataFrame, idx: int) -> float:
        start = max(0, idx - self.vol_lookback)
        return float(bars.iloc[start:idx]["volume"].mean())

    # ------------------------------------------------------------------
    # Main scanner — runs once per trading day
    # ------------------------------------------------------------------

    def scan_for_signals(
        self,
        day_data: pd.DataFrame,
        opening_high: float,
        opening_low: float,
    ) -> List[TradeSignal]:
        """
        Scan 1-minute bars from trading_start to trading_end for signals.

        State machine per direction:
          WAITING_BREAK → WAITING_RETEST → WAITING_ENGULF → DONE

        Returns a list of TradeSignal objects (at most one per direction).
        """
        signals: List[TradeSignal] = []

        window = day_data[
            (day_data["datetime"].dt.time >= self.trading_start)
            & (day_data["datetime"].dt.time < self.trading_end)
        ].copy().reset_index(drop=True)

        if len(window) < 3:
            return signals

        # Per-direction state
        bull_state: str                 = "WAITING_BREAK"
        bear_state: str                 = "WAITING_BREAK"
        active_bull_fvg: Optional[FairValueGap] = None
        active_bear_fvg: Optional[FairValueGap] = None
        bull_done = False
        bear_done = False

        for i in range(2, len(window)):
            curr = window.iloc[i]
            prev = window.iloc[i - 1]

            # ── BULLISH SIDE ─────────────────────────────────────────────
            if not bull_done:

                if bull_state == "WAITING_BREAK":
                    # Condition 1 + 2: close above OH and FVG present
                    if curr["close"] > opening_high:
                        fvg = self._detect_fvg(window, i, "bullish")
                        if fvg:
                            active_bull_fvg = fvg
                            bull_state = "WAITING_RETEST"
                            logger.debug(
                                "Bull FVG @ %s  zone=[%.2f – %.2f]",
                                curr["datetime"], fvg.gap_low, fvg.gap_high,
                            )

                elif bull_state == "WAITING_RETEST" and active_bull_fvg:
                    # Condition 3: price pulls back into FVG
                    if self._is_retest(curr, active_bull_fvg):
                        bull_state = "WAITING_ENGULF"
                        logger.debug("Bull FVG retest @ %s", curr["datetime"])
                    # Invalidate if price closes below the bottom of the FVG
                    elif curr["close"] < active_bull_fvg.gap_low:
                        active_bull_fvg = None
                        bull_state = "WAITING_BREAK"

                elif bull_state == "WAITING_ENGULF" and active_bull_fvg:
                    if self._is_retest(curr, active_bull_fvg):
                        # Condition 4 + 5: engulfing with volume
                        if self._is_engulfing(prev, curr, "bullish"):
                            avg_vol = self._rolling_avg_volume(window, i)
                            if avg_vol > 0 and curr["volume"] >= avg_vol * self.vol_multiplier:
                                entry  = curr["close"]
                                stop   = active_bull_fvg.c1_low
                                risk   = entry - stop
                                if risk > 0:
                                    signals.append(TradeSignal(
                                        direction        = "LONG",
                                        entry_price      = entry,
                                        stop_price       = stop,
                                        target_price     = entry + risk * self.rr_ratio,
                                        risk_points      = risk,
                                        reward_points    = risk * self.rr_ratio,
                                        signal_time      = curr["datetime"],
                                        fvg              = active_bull_fvg,
                                        engulfing_idx    = i,
                                        engulfing_volume = curr["volume"],
                                        avg_volume       = avg_vol,
                                    ))
                                    bull_done  = True
                                    bull_state = "DONE"
                                    logger.debug(
                                        "LONG signal @ %s  entry=%.2f stop=%.2f target=%.2f",
                                        curr["datetime"], entry, stop,
                                        entry + risk * self.rr_ratio,
                                    )
                            else:
                                # Volume not enough — stay in WAITING_ENGULF,
                                # next bar may still form a valid engulf
                                pass
                    else:
                        # Price exited FVG zone upward — back to WAITING_RETEST
                        # or invalidate if it has broken far above
                        if active_bull_fvg and curr["close"] < active_bull_fvg.gap_low:
                            active_bull_fvg = None
                            bull_state = "WAITING_BREAK"
                        else:
                            bull_state = "WAITING_RETEST"

            # ── BEARISH SIDE ─────────────────────────────────────────────
            if not bear_done:

                if bear_state == "WAITING_BREAK":
                    if curr["close"] < opening_low:
                        fvg = self._detect_fvg(window, i, "bearish")
                        if fvg:
                            active_bear_fvg = fvg
                            bear_state = "WAITING_RETEST"
                            logger.debug(
                                "Bear FVG @ %s  zone=[%.2f – %.2f]",
                                curr["datetime"], fvg.gap_low, fvg.gap_high,
                            )

                elif bear_state == "WAITING_RETEST" and active_bear_fvg:
                    if self._is_retest(curr, active_bear_fvg):
                        bear_state = "WAITING_ENGULF"
                        logger.debug("Bear FVG retest @ %s", curr["datetime"])
                    elif curr["close"] > active_bear_fvg.gap_high:
                        active_bear_fvg = None
                        bear_state = "WAITING_BREAK"

                elif bear_state == "WAITING_ENGULF" and active_bear_fvg:
                    if self._is_retest(curr, active_bear_fvg):
                        if self._is_engulfing(prev, curr, "bearish"):
                            avg_vol = self._rolling_avg_volume(window, i)
                            if avg_vol > 0 and curr["volume"] >= avg_vol * self.vol_multiplier:
                                entry  = curr["close"]
                                stop   = active_bear_fvg.c1_high
                                risk   = stop - entry
                                if risk > 0:
                                    signals.append(TradeSignal(
                                        direction        = "SHORT",
                                        entry_price      = entry,
                                        stop_price       = stop,
                                        target_price     = entry - risk * self.rr_ratio,
                                        risk_points      = risk,
                                        reward_points    = risk * self.rr_ratio,
                                        signal_time      = curr["datetime"],
                                        fvg              = active_bear_fvg,
                                        engulfing_idx    = i,
                                        engulfing_volume = curr["volume"],
                                        avg_volume       = avg_vol,
                                    ))
                                    bear_done  = True
                                    bear_state = "DONE"
                                    logger.debug(
                                        "SHORT signal @ %s  entry=%.2f stop=%.2f target=%.2f",
                                        curr["datetime"], entry, stop,
                                        entry - risk * self.rr_ratio,
                                    )
                            else:
                                pass
                    else:
                        if active_bear_fvg and curr["close"] > active_bear_fvg.gap_high:
                            active_bear_fvg = None
                            bear_state = "WAITING_BREAK"
                        else:
                            bear_state = "WAITING_RETEST"

            if bull_done and bear_done:
                break   # No more signals possible today

        return signals
