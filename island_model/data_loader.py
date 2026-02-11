"""
Загрузка исторических данных MOEX через ISS API (apimoex).
"""

import requests
import apimoex
import pandas as pd
import numpy as np


class MOEXDataLoader:
    """Загрузка OHLCV свечей и индексных данных через MOEX ISS API."""

    def __init__(self):
        self.session = requests.Session()

    def load_candles(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: int = 24,
    ) -> pd.DataFrame:
        """
        Загрузка дневных свечей для тикера.

        Parameters
        ----------
        ticker : str
            Тикер MOEX (например, 'SBER', 'GAZP').
        start : str
            Дата начала 'YYYY-MM-DD'.
        end : str
            Дата конца 'YYYY-MM-DD'.
        interval : int
            Интервал свечей (24 = дневной).

        Returns
        -------
        pd.DataFrame с колонками open, close, high, low, volume и datetime-индексом.
        """
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
        """Загрузка данных для нескольких тикеров."""
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
        """Загрузка истории индекса MOEX."""
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
        """Загрузка курса USD/RUB (фиксинг MOEX)."""
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
        """Загрузка фьючерса на Brent (BRJ* на MOEX)."""
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
