"""
config.py — Central configuration for the NSE ORB Strategy Backtester.

HOW TO SWITCH DATA PROVIDERS:
  This file is the single source of truth for all credentials and constants.
  When you move from Angel Broking to another provider (e.g. Zerodha Kite),
  only update the credentials block and the INSTRUMENTS token map below.
  The data_fetcher.py module handles the provider-specific API calls.
"""

# ---------------------------------------------------------------------------
# Angel Broking Smart API Credentials
# Sign up at https://smartapi.angelbroking.com/
# ---------------------------------------------------------------------------
ANGEL_API_KEY     = "YOUR_API_KEY_HERE"
ANGEL_CLIENT_ID   = "YOUR_CLIENT_ID_HERE"
ANGEL_PASSWORD    = "YOUR_MPIN_HERE"          # 4-digit MPIN
ANGEL_TOTP_SECRET = "YOUR_TOTP_SECRET_HERE"   # Base32 secret from Authenticator setup

# ---------------------------------------------------------------------------
# Instrument Registry
# Add new instruments here as NSE expands the strategy universe.
#
# Angel Broking index tokens:
#   NIFTY 50   → 26000  (exchange: NSE)  [spot index — Angel API returns no OHLCV data]
#   NIFTY BANK → 26009  (exchange: NSE)  [spot index — Angel API returns no OHLCV data]
#
# Angel Broking equity tokens:
#   HDFCBANK   → 1333   (exchange: NSE)
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "NSE NIFTY": {
        "exchange": "NSE",
        "symbol": "Nifty 50",
        "token": "26000",
        "trading_symbol": "NIFTY",
    },
    "NSE BANKNIFTY": {
        "exchange": "NSE",
        "symbol": "Nifty Bank",
        "token": "26009",
        "trading_symbol": "BANKNIFTY",
    },
    "NSE HDFCBANK": {
        "exchange": "NSE",
        "symbol": "HDFC Bank",
        "token": "1333",
        "trading_symbol": "HDFCBANK",
    },
}

# ---------------------------------------------------------------------------
# Strategy Parameters
# ---------------------------------------------------------------------------
TRADING_CONFIG = {
    # Opening range window (15-min candle)
    "opening_range_start": "09:15",
    "opening_range_end":   "09:30",

    # Active scanning window (1-min bars)
    "trading_start": "09:30",
    "trading_end":   "15:00",

    # Trade management
    "risk_reward_ratio": 3.0,         # Target = entry ± risk * 3

    # Volume filter for engulfing confirmation
    "volume_multiplier": 1.25,        # Engulfing volume must be ≥ 1.25× rolling average
    "volume_lookback":   20,          # Bars used to compute the rolling average
}

# ---------------------------------------------------------------------------
# Backtesting Parameters
# ---------------------------------------------------------------------------
BACKTEST_CONFIG = {
    "initial_capital": 100_000,       # Starting capital in INR

    # NSE lot sizes (used for P&L calculation in rupees)
    "lot_size": {
        "NSE NIFTY":     50,
        "NSE BANKNIFTY": 15,
        "NSE HDFCBANK":  550,
    },

    "brokerage_per_trade": 40,        # INR charged per round trip (entry + exit)
    "slippage_points":     2,         # Points of adverse slippage per fill
}
