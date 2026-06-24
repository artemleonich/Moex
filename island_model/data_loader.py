# Загрузка исторических данных MOEX через ISS API (apimoex)

from typing import Optional

import requests
import apimoex
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_TIMEOUT = 30           # seconds — prevents hanging on stalled TCP sockets
DEFAULT_MAX_RETRIES = 3        # retry count for transient HTTP/connection errors
DEFAULT_BACKOFF_FACTOR = 0.5   # exponential backoff base: 0.5s, 1s, 2s, ...
DEFAULT_RETRY_STATUS = (429, 500, 502, 503, 504)


class _TimeoutSession(requests.Session):
    """``requests.Session`` subclass that injects a default ``timeout`` kwarg.

    Funnels all HTTP calls — including those made by third-party libs that
    accept a Session object (e.g. ``apimoex``) — through a single
    ``request()`` override so we never hang on stalled TCP sockets.
    """

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT):
        super().__init__()
        self._default_timeout = default_timeout

    def request(self, method, url, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


def _build_session(
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = DEFAULT_RETRY_STATUS,
) -> requests.Session:
    """Build a hardened requests.Session with retry policy and default timeout."""
    session: requests.Session = _TimeoutSession(default_timeout=timeout)

    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MOEXDataLoader:
    """Загружает OHLCV свечи и индексные данные через MOEX ISS API."""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: requests.Session = _build_session(
            timeout=timeout,
            max_retries=max_retries,
        )

    def load_candles(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: int = 24,
    ) -> pd.DataFrame:
        """Загрузить дневные свечи для одного тикера."""
        data = apimoex.get_board_candles(
            self.session,
            ticker,
            interval=interval,
            start=start,
            end=end,
        )
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError(f"No data returned for {ticker} ({start} — {end})")
        df["begin"] = pd.to_datetime(df["begin"])
        df = df.set_index("begin")
        df = df.rename(
            columns={"open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"}
        )
        required = ["open", "close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing from API response for {ticker}")
        return df[required].copy()

    def load_multiple(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: int = 24,
    ) -> dict[str, pd.DataFrame]:
        """Загрузить данные для нескольких тикеров сразу."""
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load_candles(ticker, start, end, interval)
                print(f"  Loaded {ticker}: {len(result[ticker])} rows")
            except Exception as e:
                print(f"  SKIP {ticker}: {e}")
        return result

    def load_index(
        self,
        index: str = "IMOEX",
        start: str = "2015-01-01",
        end: str = "2026-01-01",
    ) -> pd.DataFrame:
        """Загрузить историю индекса MOEX."""
        data = apimoex.get_board_history(
            self.session,
            index,
            board="SNDX",
            market="index",
            engine="stock",
            start=start,
            end=end,
        )
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError(f"No index data for {index}")
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df = df.set_index("TRADEDATE")
        return df

    def load_usd_rub(self, start: str, end: str) -> pd.Series:
        """Загрузить курс USD/RUB (фиксинг MOEX)."""
        data = apimoex.get_board_candles(
            self.session,
            "USD000UTSTOM",
            interval=24,
            start=start,
            end=end,
            market="selt",
            engine="currency",
        )
        df = pd.DataFrame(data)
        if df.empty:
            return pd.Series(dtype=float, name="usdrub")
        df["begin"] = pd.to_datetime(df["begin"])
        df = df.set_index("begin")
        return df["close"].rename("usdrub")

    def load_brent(self, start: str, end: str) -> pd.Series:
        """Загрузить фьючерс на Brent (BRJ* на MOEX)."""
        data = apimoex.get_board_candles(
            self.session,
            "BR-3.25",
            interval=24,
            start=start,
            end=end,
            market="forts",
            engine="futures",
            board="RFUD",
        )
        df = pd.DataFrame(data)
        if df.empty:
            return pd.Series(dtype=float, name="brent")
        df["begin"] = pd.to_datetime(df["begin"])
        df = df.set_index("begin")
        return df["close"].rename("brent")
