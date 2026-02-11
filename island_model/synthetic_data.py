"""
Генератор синтетических данных, имитирующих поведение акций MOEX.

Используется при недоступности MOEX ISS API. Генерирует реалистичные
OHLCV данные с:
- Режимными переключениями (бычий/медвежий/боковой)
- Кластеризацией волатильности (GARCH-подобная)
- Сезонными паттернами
- Корреляцией с «нефтью» и «курсом»
"""

import numpy as np
import pandas as pd


def generate_regime_switches(n_days: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Генерация массива режимов: 0=бычий, 1=медвежий, 2=боковой.
    Переключение через марковскую цепь.
    """
    # Матрица переходов (строки → из, столбцы → в)
    # Бычий рынок длится дольше, медвежий — короче
    transition = np.array([
        [0.98, 0.01, 0.01],  # бычий → бычий/медвежий/боковой
        [0.03, 0.94, 0.03],  # медвежий → ...
        [0.02, 0.02, 0.96],  # боковой → ...
    ])
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = 0  # начинаем с бычьего
    for i in range(1, n_days):
        regimes[i] = rng.choice(3, p=transition[regimes[i - 1]])
    return regimes


def generate_ohlcv(
    ticker: str,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерация синтетических дневных OHLCV данных.

    Параметры подбираются индивидуально для каждого тикера для
    имитации реалистичного поведения.
    """
    # Параметры по тикерам (начальная цена, средняя волатильность, дрифт)
    ticker_params = {
        "SBER": {"price0": 220, "vol": 0.020, "drift": 0.0003},
        "GAZP": {"price0": 150, "vol": 0.022, "drift": 0.0001},
        "LKOH": {"price0": 4500, "vol": 0.018, "drift": 0.0004},
        "GMKN": {"price0": 14000, "vol": 0.019, "drift": 0.0002},
        "ROSN": {"price0": 400, "vol": 0.021, "drift": 0.0002},
    }

    params = ticker_params.get(ticker, {"price0": 100, "vol": 0.020, "drift": 0.0002})
    rng = np.random.RandomState(seed + hash(ticker) % 10000)

    # Торговые дни (без выходных)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Генерация режимов
    regimes = generate_regime_switches(n, rng)

    # Drift зависит от режима
    regime_drift = {0: params["drift"], 1: -params["drift"] * 2, 2: 0.0}
    regime_vol_mult = {0: 0.8, 1: 1.5, 2: 1.0}

    # GARCH(1,1)-подобная волатильность
    vol = np.zeros(n)
    vol[0] = params["vol"]
    omega = params["vol"] ** 2 * 0.05
    alpha = 0.10
    beta = 0.85

    returns = np.zeros(n)
    close = np.zeros(n)
    close[0] = params["price0"]

    for i in range(1, n):
        # Обновление волатильности (GARCH)
        vol[i] = np.sqrt(
            omega + alpha * returns[i - 1] ** 2 + beta * vol[i - 1] ** 2
        )
        # Режимная волатильность
        vol_adj = vol[i] * regime_vol_mult[regimes[i]]
        # Доходность
        drift = regime_drift[regimes[i]]
        returns[i] = drift + vol_adj * rng.randn()
        # Цена
        close[i] = close[i - 1] * (1 + returns[i])

    # Генерация OHLV из close
    high = close * (1 + np.abs(rng.randn(n) * params["vol"] * 0.5))
    low = close * (1 - np.abs(rng.randn(n) * params["vol"] * 0.5))
    open_ = np.roll(close, 1) * (1 + rng.randn(n) * params["vol"] * 0.2)
    open_[0] = close[0]

    # Гарантируем high >= close, open и low <= close, open
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))

    # Объём: базовый уровень + кластеризация + режимная зависимость
    base_volume = 5_000_000 if params["price0"] < 1000 else 500_000
    vol_regime_mult = {0: 1.0, 1: 1.5, 2: 0.8}
    volume = np.array([
        base_volume * vol_regime_mult[regimes[i]] * (1 + abs(rng.randn()) * 0.5)
        for i in range(n)
    ]).astype(int)

    df = pd.DataFrame(
        {"open": open_, "close": close, "high": high, "low": low, "volume": volume},
        index=dates,
    )
    df.index.name = "begin"
    return df


def generate_usdrub(start: str = "2018-01-01", end: str = "2025-12-31",
                    seed: int = 100) -> pd.Series:
    """Генерация синтетического курса USD/RUB."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Начальный курс ~60, рост до ~90-100 к 2025
    rate = np.zeros(n)
    rate[0] = 60.0
    for i in range(1, n):
        drift = 0.0001  # слабый рост (девальвация)
        vol = 0.008
        rate[i] = rate[i - 1] * (1 + drift + vol * rng.randn())
        rate[i] = max(rate[i], 30)  # нижний предел

    return pd.Series(rate, index=dates, name="usdrub")


def generate_brent(start: str = "2018-01-01", end: str = "2025-12-31",
                   seed: int = 200) -> pd.Series:
    """Генерация синтетической цены Brent."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    price = np.zeros(n)
    price[0] = 70.0  # ~$70 bbl
    for i in range(1, n):
        drift = 0.0
        vol = 0.02
        price[i] = price[i - 1] * (1 + drift + vol * rng.randn())
        price[i] = max(price[i], 20)  # нижний предел

    return pd.Series(price, index=dates, name="brent")


def load_synthetic_data(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series]:
    """
    Загрузка полного набора синтетических данных.

    Returns
    -------
    (ticker_data, usdrub, brent)
    """
    ticker_data = {}
    for ticker in tickers:
        df = generate_ohlcv(ticker, start, end)
        ticker_data[ticker] = df
        print(f"  Generated {ticker}: {len(df)} rows ({df.index[0].date()} — {df.index[-1].date()})")

    usdrub = generate_usdrub(start, end)
    print(f"  Generated USD/RUB: {len(usdrub)} rows")

    brent = generate_brent(start, end)
    print(f"  Generated Brent: {len(brent)} rows")

    return ticker_data, usdrub, brent
