# синтетические данные для тестов (калиброваны под MOEX)

import numpy as np
import pandas as pd


# Параметры, калиброванные под реальный рынок MOEX 2019-2024

TICKER_PARAMS = {
    "SBER": {
        "price0": 230.0,       # начало 2019
        "lot_size": 10,        # лот = 10 акций
        "tick_size": 0.01,     # шаг цены
        "daily_vol": 0.019,    # средняя дневная волатильность
        "drift": 0.00015,      # ~3.8% годовых дрифт
        "mean_spread_bps": 2,  # средний спред в б.п.
        "avg_daily_volume": 80_000_000,
        "sector": "finance",
        "beta_market": 1.1,
        "beta_rub": -0.3,      # рост рубля → рост SBER
    },
    "GAZP": {
        "price0": 160.0,
        "lot_size": 10,
        "tick_size": 0.01,
        "daily_vol": 0.022,
        "drift": 0.00005,
        "mean_spread_bps": 3,
        "avg_daily_volume": 40_000_000,
        "sector": "oil_gas",
        "beta_market": 1.0,
        "beta_rub": -0.2,
    },
    "LKOH": {
        "price0": 5500.0,
        "lot_size": 1,
        "tick_size": 0.5,
        "daily_vol": 0.017,
        "drift": 0.00025,
        "mean_spread_bps": 3,
        "avg_daily_volume": 3_000_000,
        "sector": "oil_gas",
        "beta_market": 0.9,
        "beta_rub": -0.25,
    },
    "GMKN": {
        "price0": 14500.0,
        "lot_size": 1,
        "tick_size": 2.0,
        "daily_vol": 0.018,
        "drift": 0.0001,
        "mean_spread_bps": 4,
        "avg_daily_volume": 800_000,
        "sector": "metals",
        "beta_market": 0.85,
        "beta_rub": -0.15,
    },
    "ROSN": {
        "price0": 420.0,
        "lot_size": 1,
        "tick_size": 0.05,
        "daily_vol": 0.020,
        "drift": 0.00012,
        "mean_spread_bps": 4,
        "avg_daily_volume": 10_000_000,
        "sector": "oil_gas",
        "beta_market": 0.95,
        "beta_rub": -0.2,
    },
}

# Исторические кризисные периоды (индексы дней с начала 2019)
# ~253 торговых дней/год
CRISIS_PERIODS = [
    # COVID crash: март 2020 (~253+40 = day 293 от 2019-01-01)
    {"name": "COVID", "start_day": 290, "end_day": 330, "shock": -0.35, "vol_mult": 3.0},
    # Февраль 2022 (~253*3+35 = day 794)
    {"name": "FEB2022", "start_day": 790, "end_day": 850, "shock": -0.50, "vol_mult": 4.0},
    # Мобилизация сентябрь 2022 (~253*3+180 = day 939)
    {"name": "MOBIL2022", "start_day": 935, "end_day": 960, "shock": -0.15, "vol_mult": 2.5},
]


def _round_to_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 6)


def _generate_common_factors(n_days: int, rng: np.random.RandomState):
    """Общие рыночные факторы (рынок, нефть, рубль)."""
    # Рыночный фактор
    market = rng.randn(n_days) * 0.012

    # Нефтяной фактор (влияет на нефтегаз)
    oil = rng.randn(n_days) * 0.018

    # Рублёвый фактор (девальвация = отрицательный шок для импортёров)
    rub = rng.randn(n_days) * 0.007

    return market, oil, rub


def generate_ohlcv(
    ticker: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    common_factors: tuple | None = None,
) -> pd.DataFrame:
    """Дневные OHLCV данные, калиброванные под реальный MOEX."""
    params = TICKER_PARAMS.get(ticker, TICKER_PARAMS["SBER"])
    rng = np.random.RandomState(seed + hash(ticker) % 10000)

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Общие факторы
    if common_factors is not None:
        market_factor, oil_factor, rub_factor = common_factors
        # Обрезаем/удлиняем до нужной длины
        market_factor = market_factor[:n] if len(market_factor) >= n else np.pad(market_factor, (0, n - len(market_factor)))
        oil_factor = oil_factor[:n] if len(oil_factor) >= n else np.pad(oil_factor, (0, n - len(oil_factor)))
        rub_factor = rub_factor[:n] if len(rub_factor) >= n else np.pad(rub_factor, (0, n - len(rub_factor)))
    else:
        market_factor = np.zeros(n)
        oil_factor = np.zeros(n)
        rub_factor = np.zeros(n)

    # GARCH(1,1) волатильность
    vol = np.zeros(n)
    vol[0] = params["daily_vol"]
    omega = params["daily_vol"] ** 2 * 0.04
    alpha_g = 0.08
    beta_g = 0.88

    returns = np.zeros(n)
    close = np.zeros(n)
    close[0] = params["price0"]

    # Кризисные шоки
    crisis_applied = set()

    for i in range(1, n):
        # GARCH волатильность
        vol[i] = np.sqrt(omega + alpha_g * returns[i - 1] ** 2 + beta_g * vol[i - 1] ** 2)

        # Кризисные мультипликаторы
        vol_mult = 1.0
        crisis_shock = 0.0
        for crisis in CRISIS_PERIODS:
            if crisis["start_day"] <= i <= crisis["end_day"]:
                vol_mult = max(vol_mult, crisis["vol_mult"])
                if i == crisis["start_day"] and crisis["name"] not in crisis_applied:
                    # Первый день кризиса — шоковая доходность
                    crisis_shock = crisis["shock"] / (crisis["end_day"] - crisis["start_day"])
                    crisis_applied.add(crisis["name"])
                elif crisis["start_day"] < i <= crisis["end_day"]:
                    crisis_shock = crisis["shock"] / (crisis["end_day"] - crisis["start_day"])

        vol_adj = vol[i] * vol_mult

        # Доходность = drift + idiosyncratic + factor exposure + crisis
        idio = rng.randn() * vol_adj * 0.6  # 60% идиосинкратический риск
        factor = (
            params["beta_market"] * market_factor[i]
            + (0.5 if params["sector"] == "oil_gas" else 0.1) * oil_factor[i]
            + params["beta_rub"] * rub_factor[i]
        )

        returns[i] = params["drift"] + idio + factor * 0.4 + crisis_shock
        # Ограничиваем доходность ±15% (circuit breaker MOEX)
        returns[i] = np.clip(returns[i], -0.15, 0.15)

        close[i] = close[i - 1] * (1 + returns[i])
        close[i] = max(close[i], params["tick_size"])  # не может быть 0
        close[i] = _round_to_tick(close[i], params["tick_size"])

    # Генерация OHLV
    intraday_range = np.abs(rng.randn(n)) * params["daily_vol"] * 0.7
    high = close * (1 + intraday_range)
    low = close * (1 - np.abs(rng.randn(n)) * params["daily_vol"] * 0.7)

    # Open = close предыдущего дня + overnight gap
    open_ = np.zeros(n)
    open_[0] = close[0]
    for i in range(1, n):
        gap = rng.randn() * params["daily_vol"] * 0.3
        open_[i] = close[i - 1] * (1 + gap)

    # Гарантии OHLC
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))
    low = np.maximum(low, params["tick_size"])

    # Округление до шага цены
    for arr in [open_, high, low, close]:
        for i in range(n):
            arr[i] = _round_to_tick(arr[i], params["tick_size"])

    # Объём: реалистичная модель
    base_vol = params["avg_daily_volume"]
    volume = np.zeros(n, dtype=int)
    for i in range(n):
        # Объём растёт с волатильностью, падает в боковике
        vol_effect = (vol[i] / params["daily_vol"]) ** 1.5
        # Сезонность: меньше в летние месяцы
        month = dates[i].month
        seasonal = 0.8 if month in [6, 7, 8] else 1.0
        # Случайность
        noise = max(0.3, 1 + rng.randn() * 0.4)
        volume[i] = int(base_vol * vol_effect * seasonal * noise)

    df = pd.DataFrame(
        {"open": open_, "close": close, "high": high, "low": low, "volume": volume},
        index=dates,
    )
    df.index.name = "begin"
    return df, params


def generate_usdrub(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int = 100,
) -> pd.Series:
    """USD/RUB с реалистичными уровнями."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    rate = np.zeros(n)
    rate[0] = 66.0  # начало 2019

    for i in range(1, n):
        drift = 0.00008  # слабая девальвация
        vol = 0.006

        # Кризисные шоки для рубля
        shock = 0.0
        vol_m = 1.0
        for crisis in CRISIS_PERIODS:
            if crisis["start_day"] <= i <= crisis["end_day"]:
                vol_m = max(vol_m, crisis["vol_mult"] * 0.7)
                if crisis["name"] == "FEB2022":
                    shock = 0.02  # сильная девальвация
                elif crisis["name"] == "COVID":
                    shock = 0.005

        rate[i] = rate[i - 1] * (1 + drift + shock + vol * vol_m * rng.randn())
        rate[i] = np.clip(rate[i], 50, 130)

    return pd.Series(rate, index=dates, name="usdrub")


def generate_brent(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    seed: int = 200,
) -> pd.Series:
    """Цена Brent с реалистичными уровнями."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    price = np.zeros(n)
    price[0] = 65.0  # начало 2019

    for i in range(1, n):
        drift = 0.0
        vol = 0.022

        shock = 0.0
        vol_m = 1.0
        for crisis in CRISIS_PERIODS:
            if crisis["start_day"] <= i <= crisis["end_day"]:
                vol_m = max(vol_m, crisis["vol_mult"] * 0.8)
                if crisis["name"] == "COVID":
                    shock = -0.015  # обвал нефти

        price[i] = price[i - 1] * (1 + drift + shock + vol * vol_m * rng.randn())
        price[i] = np.clip(price[i], 15, 130)

    return pd.Series(price, index=dates, name="brent")


def load_synthetic_data(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series]:
    """Полный набор синтетических данных с кросс-корреляциями."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Общие факторы для всех инструментов
    common_factors = _generate_common_factors(n, rng)

    ticker_data = {}
    for ticker in tickers:
        df, params = generate_ohlcv(ticker, start, end, common_factors=common_factors)
        ticker_data[ticker] = df
        final_price = df["close"].iloc[-1]
        total_ret = (final_price / df["close"].iloc[0] - 1) * 100
        print(f"  Generated {ticker}: {len(df)} rows "
              f"({df.index[0].date()} — {df.index[-1].date()}) "
              f"price {df['close'].iloc[0]:.0f} → {final_price:.0f} "
              f"({total_ret:+.1f}%)")

    usdrub = generate_usdrub(start, end)
    print(f"  Generated USD/RUB: {len(usdrub)} rows "
          f"({usdrub.iloc[0]:.1f} → {usdrub.iloc[-1]:.1f})")

    brent = generate_brent(start, end)
    print(f"  Generated Brent: {len(brent)} rows "
          f"(${brent.iloc[0]:.1f} → ${brent.iloc[-1]:.1f})")

    return ticker_data, usdrub, brent
