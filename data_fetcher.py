"""
data_fetcher.py — Data-provider abstraction layer.

Architecture
------------
BaseDataFetcher (ABC)
    ├── AngelBrokingFetcher   ← current default (Angel Broking Smart API)
    └── CSVDataFetcher        ← offline / testing fallback

TO SWITCH PROVIDERS:
  1. Create a new class that inherits BaseDataFetcher.
  2. Implement the three abstract methods: connect(), get_historical_data(),
     disconnect().
  3. In main.py, swap the instantiation line (currently AngelBrokingFetcher)
     for your new class — nothing else in the strategy or backtest code needs
     to change.

Angel Broking API reference: https://smartapi.angelbroking.com/docs
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDataFetcher(ABC):
    """
    Provider-agnostic interface for OHLCV data.

    All concrete fetchers must return a DataFrame with exactly these columns:
        datetime  | open | high | low | close | volume
        --------- | ---- | ---- | --- | ----- | ------
        pd.Timestamp (tz-naive, IST)
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish a session with the data provider. Returns True on success."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        token: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.

        Parameters
        ----------
        symbol   : Human-readable name (used for logging / CSV filenames).
        exchange : Market segment, e.g. "NSE".
        token    : Provider-specific instrument identifier.
        interval : Canonical interval string — "1min", "5min", "15min", "1day".
        from_date: Start of the requested window (inclusive).
        to_date  : End of the requested window (inclusive).

        Returns
        -------
        pd.DataFrame with columns [datetime, open, high, low, close, volume].
        Returns an empty DataFrame if no data is available.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Terminate the session and release any resources."""


# ---------------------------------------------------------------------------
# Angel Broking Smart API implementation
# ---------------------------------------------------------------------------

class AngelBrokingFetcher(BaseDataFetcher):
    """
    Fetches historical candle data from Angel Broking's Smart API.

    Limitations (as of 2024):
      • 1-min data  : up to ~30 days per request
      • 5-min data  : up to ~100 days per request
      • 1-day data  : up to ~2 years per request
    The fetcher automatically chunks long date ranges to stay within limits.

    Dependencies:
      pip install smartapi-python pyotp
    """

    # Map from our canonical interval strings → Angel Broking enum strings
    _INTERVAL_MAP = {
        "1min":  "ONE_MINUTE",
        "3min":  "THREE_MINUTE",
        "5min":  "FIVE_MINUTE",
        "10min": "TEN_MINUTE",
        "15min": "FIFTEEN_MINUTE",
        "30min": "THIRTY_MINUTE",
        "1hour": "ONE_HOUR",
        "1day":  "ONE_DAY",
    }

    # Maximum days per API call for each interval (conservative figures)
    _MAX_DAYS = {
        "ONE_MINUTE":    30,
        "THREE_MINUTE":  60,
        "FIVE_MINUTE":   100,
        "TEN_MINUTE":    100,
        "FIFTEEN_MINUTE": 200,
        "THIRTY_MINUTE": 200,
        "ONE_HOUR":      400,
        "ONE_DAY":       500,
    }

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
    ):
        self.api_key     = api_key
        self.client_id   = client_id
        self.password    = password
        self.totp_secret = totp_secret
        self._api        = None   # SmartConnect instance

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            import pyotp
            from SmartApi import SmartConnect
        except ImportError as e:
            raise ImportError(
                "Required packages missing. Run:  pip install smartapi-python pyotp"
            ) from e

        try:
            self._api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            response = self._api.generateSession(self.client_id, self.password, totp)

            if response and response.get("status"):
                logger.info("Connected to Angel Broking Smart API")
                return True

            logger.error("Angel Broking login failed: %s", response)
            return False

        except Exception as exc:
            logger.error("Connection error: %s", exc)
            return False

    # ------------------------------------------------------------------
    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        token: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:

        if self._api is None:
            raise RuntimeError("Not connected. Call connect() first.")

        angel_interval = self._INTERVAL_MAP.get(interval)
        if not angel_interval:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Choose from: {list(self._INTERVAL_MAP)}"
            )

        max_days   = self._MAX_DAYS[angel_interval]
        all_rows   = []
        chunk_from = from_date

        while chunk_from < to_date:
            chunk_to = min(chunk_from + timedelta(days=max_days), to_date)

            params = {
                "exchange":    exchange,
                "symboltoken": token,
                "interval":    angel_interval,
                "fromdate":    chunk_from.strftime("%Y-%m-%d %H:%M"),
                "todate":      chunk_to.strftime("%Y-%m-%d %H:%M"),
            }

            try:
                logger.debug("getCandleData params: %s", params)
                resp = self._api.getCandleData(params)
                logger.debug("getCandleData response: %s", resp)
                if resp and resp.get("data"):
                    all_rows.extend(resp["data"])
                    logger.debug(
                        "Fetched %d bars for %s [%s → %s]",
                        len(resp["data"]), symbol,
                        chunk_from.date(), chunk_to.date(),
                    )
                else:
                    logger.warning(
                        "Empty response for chunk %s→%s: %s",
                        chunk_from.date(), chunk_to.date(), resp,
                    )
            except Exception as exc:
                logger.warning("API error for chunk %s→%s: %s", chunk_from.date(), chunk_to.date(), exc)

            chunk_from = chunk_to + timedelta(minutes=1)

        if not all_rows:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        logger.info("Total bars fetched for %s: %d", symbol, len(df))
        return df

    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        if self._api:
            try:
                self._api.terminateSession(self.client_id)
                logger.info("Angel Broking session terminated")
            except Exception:
                pass
            self._api = None


# ---------------------------------------------------------------------------
# CSV fallback (useful for testing without an API subscription)
# ---------------------------------------------------------------------------

class CSVDataFetcher(BaseDataFetcher):
    """
    Reads pre-downloaded OHLCV data from CSV files.

    Expected filename convention:
        <data_dir>/<trading_symbol>_<interval>.csv
        e.g.  data/NIFTY_1min.csv

    Required CSV columns (case-insensitive):
        datetime, open, high, low, close, volume

    The 'datetime' column must be parseable by pandas (ISO-8601 recommended).
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

    def connect(self) -> bool:
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("CSV fetcher ready (data dir: %s)", self.data_dir)
        return True

    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        token: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        import os

        path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSV file not found: {path}\n"
                f"Please place a file with columns "
                f"[datetime, open, high, low, close, volume] at that path."
            )

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        df = df[(df["datetime"] >= from_date) & (df["datetime"] <= to_date)]
        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        logger.info("Loaded %d bars from %s", len(df), path)
        return df

    def disconnect(self) -> None:
        pass
