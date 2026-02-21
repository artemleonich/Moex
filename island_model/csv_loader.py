# загрузка из csv

import os

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_ticker_csv(ticker: str, data_dir: str | None = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, f"{ticker}_daily.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "begin"
    return df[["open", "high", "low", "close", "volume"]]


def load_usdrub_csv(data_dir: str | None = None) -> pd.Series:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "USDRUB_daily.csv")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    s.name = "usdrub"
    return s


def load_brent_csv(data_dir: str | None = None) -> pd.Series:
    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, "BRENT_daily.csv")
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    s.name = "brent"
    return s


def load_all_csv(
    tickers: list[str],
    data_dir: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Series | None, pd.Series | None]:
    data_dir = data_dir or DATA_DIR
    ticker_data = {}
    for t in tickers:
        try:
            df = load_ticker_csv(t, data_dir)
            ticker_data[t] = df
            print(f"  CSV {t}: {len(df)} rows "
                  f"({df.index[0].date()} — {df.index[-1].date()}) "
                  f"close {df['close'].iloc[0]:.0f} → {df['close'].iloc[-1]:.0f}")
        except Exception as e:
            print(f"  CSV {t}: SKIP ({e})")

    usdrub = None
    try:
        usdrub = load_usdrub_csv(data_dir)
        print(f"  CSV USDRUB: {len(usdrub)} rows")
    except Exception:
        pass

    brent = None
    try:
        brent = load_brent_csv(data_dir)
        print(f"  CSV BRENT: {len(brent)} rows")
    except Exception:
        pass

    return ticker_data, usdrub, brent
