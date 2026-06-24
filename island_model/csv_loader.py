# загрузка из csv

import logging
import os
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# Expected exception types from pd.read_csv + the OS layer. Catching only
# these (rather than bare ``Exception``) means real bugs in user code or
# data corruption surface immediately instead of being silently logged.
_CSV_LOAD_ERRORS = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    OSError,
    UnicodeDecodeError,
)


def load_ticker_csv(ticker: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, f"{ticker}_daily.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "begin"
    return df[["open", "high", "low", "close", "volume"]]


def load_usdrub_csv(data_dir: Optional[str] = None) -> pd.Series:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "USDRUB_daily.csv")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    s.name = "usdrub"
    return s


def load_brent_csv(data_dir: Optional[str] = None) -> pd.Series:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "BRENT_daily.csv")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    s.name = "brent"
    return s


def _format_ohlcv_summary(t: str, df: pd.DataFrame) -> str:
    return (
        f"  CSV {t}: {len(df)} rows "
        f"({df.index[0].date()} — {df.index[-1].date()}) "
        f"close {df['close'].iloc[0]:.0f} → {df['close'].iloc[-1]:.0f}"
    )


def load_all_csv(
    tickers: list[str],
    data_dir: Optional[str] = None,
) -> tuple[dict[str, pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Load OHLCV for ``tickers`` plus the optional USDRUB and Brent series.

    USDRUB and Brent are treated as optional context: missing or malformed
    files are logged at WARNING and the loader returns ``None`` for that
    series, allowing the backtest to continue with what it has.

    Tickers are also loaded defensively — a single missing ticker file
    skips that ticker but does not abort the whole load — but their
    failures are logged at WARNING so they remain visible.
    """
    data_dir = data_dir or DATA_DIR

    ticker_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = load_ticker_csv(t, data_dir)
        except _CSV_LOAD_ERRORS as e:
            logger.warning("CSV %s: skipped (%s: %s)", t, type(e).__name__, e)
            continue
        ticker_data[t] = df
        logger.info(_format_ohlcv_summary(t, df))

    usdrub: Optional[pd.Series] = None
    try:
        usdrub = load_usdrub_csv(data_dir)
    except _CSV_LOAD_ERRORS as e:
        logger.warning("CSV USDRUB: skipped (%s: %s)", type(e).__name__, e)

    if usdrub is not None:
        logger.info("  CSV USDRUB: %d rows", len(usdrub))

    brent: Optional[pd.Series] = None
    try:
        brent = load_brent_csv(data_dir)
    except _CSV_LOAD_ERRORS as e:
        logger.warning("CSV BRENT: skipped (%s: %s)", type(e).__name__, e)

    if brent is not None:
        logger.info("  CSV BRENT: %d rows", len(brent))

    return ticker_data, usdrub, brent