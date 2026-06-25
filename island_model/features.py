# фичи для MOEX

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line.rename("macd"), signal_line.rename("macd_signal"), histogram.rename("macd_hist")


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    sma = _sma(close, window)
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / sma
    return pct_b.rename("bb_pctb"), width.rename("bb_width")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().rename("atr")


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum().rename("obv")


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_window: int = 14, d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_window).mean()
    return k.rename("stoch_k"), d.rename("stoch_d")


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               window: int = 14) -> pd.Series:
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    wr = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return wr.rename("williams_r")


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - sma_tp) / (0.015 * mad).replace(0, np.nan)).rename("cci")


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Обнуляем, если один DM меньше другого
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1 / window, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, min_periods=window).mean().rename("adx")


def ichimoku(
    high: pd.Series, low: pd.Series,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return (
        tenkan_sen.rename("ichimoku_tenkan"),
        kijun_sen.rename("ichimoku_kijun"),
        senkou_a.rename("ichimoku_senkou_a"),
        senkou_b_val.rename("ichimoku_senkou_b"),
    )


class FeatureGenerator:
    """Генерация 30+ признаков из OHLCV данных."""

    def __init__(self, windows: list[int] | None = None):
        self.windows = windows or [5, 10, 20, 60]

    def generate(
        self,
        df: pd.DataFrame,
        usdrub: pd.Series | None = None,
        brent: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Строит все фичи из OHLCV + опционально usdrub/brent."""
        f = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        for w in self.windows:
            f[f"ret_{w}"] = close.pct_change(w)
            f[f"vol_{w}"] = close.pct_change().rolling(w).std()
            f[f"volume_ratio_{w}"] = volume / volume.rolling(w).mean().replace(0, np.nan)
            f[f"high_low_range_{w}"] = ((high - low) / close).rolling(w).mean()

        f["rsi_14"] = rsi(close, 14)

        macd_line, macd_sig, macd_h = macd(close)
        f["macd_diff"] = macd_h

        bb_pctb, bb_w = bollinger_bands(close)
        f["bb_pctb"] = bb_pctb
        f["bb_width"] = bb_w

        f["atr_14"] = atr(high, low, close, 14)

        f["obv"] = obv(close, volume)
        # z-score OBV в скользящем окне
        f["obv_zscore"] = (
            (f["obv"] - f["obv"].rolling(60).mean())
            / f["obv"].rolling(60).std().replace(0, np.nan)
        )

        stoch_k, stoch_d = stochastic(high, low, close)
        f["stoch_k"] = stoch_k
        f["stoch_d"] = stoch_d

        f["williams_r"] = williams_r(high, low, close)
        f["cci_20"] = cci(high, low, close, 20)
        f["adx_14"] = adx(high, low, close, 14)

        tenkan_s, kijun_s, senkou_a, senkou_b = ichimoku(high, low)
        f["ichimoku_tenkan_vs_close"] = (tenkan_s - close) / close
        f["ichimoku_kijun_vs_close"] = (kijun_s - close) / close
        f["ichimoku_cloud_width"] = (senkou_a - senkou_b) / close

        f["volume_ma_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
        f["price_volume_trend"] = (close.pct_change() * volume).cumsum()
        f["pvt_zscore"] = (
            (f["price_volume_trend"] - f["price_volume_trend"].rolling(60).mean())
            / f["price_volume_trend"].rolling(60).std().replace(0, np.nan)
        )

        if hasattr(df.index, "dayofweek"):
            dow = df.index.dayofweek
        else:
            dow = pd.to_datetime(df.index).dayofweek

        f["day_sin"] = np.sin(2 * np.pi * dow / 5)
        f["day_cos"] = np.cos(2 * np.pi * dow / 5)

        day_of_month = df.index.day if hasattr(df.index, "day") else pd.to_datetime(df.index).day
        f["month_end"] = (day_of_month >= 20).astype(int)  # налоговый период

        month = df.index.month if hasattr(df.index, "month") else pd.to_datetime(df.index).month
        f["month_sin"] = np.sin(2 * np.pi * month / 12)
        f["month_cos"] = np.cos(2 * np.pi * month / 12)

        if usdrub is not None:
            usdrub_aligned = usdrub.reindex(df.index, method="ffill")
            f["usdrub_ret_5"] = usdrub_aligned.pct_change(5)
            f["usdrub_ret_20"] = usdrub_aligned.pct_change(20)
            f["usdrub_vol_20"] = usdrub_aligned.pct_change().rolling(20).std()

        if brent is not None:
            brent_aligned = brent.reindex(df.index, method="ffill")
            f["brent_ret_5"] = brent_aligned.pct_change(5)
            f["brent_ret_20"] = brent_aligned.pct_change(20)

            if usdrub is not None:
                brent_rub = brent_aligned * usdrub_aligned
                f["brent_rub_ret_5"] = brent_rub.pct_change(5)
                # скользящая корреляция нефть-рубль (30 дней)
                f["oil_rub_corr_30"] = (
                    brent_aligned.pct_change()
                    .rolling(30)
                    .corr(usdrub_aligned.pct_change())
                )

        return f.dropna()


class TargetGenerator:
    """Бинарная целевая переменная (forward return > 0)."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Возвращает 1 если forward return > 0, иначе 0."""
        fwd = df["close"].pct_change(self.horizon).shift(-self.horizon)
        return (fwd > 0).astype(int).rename(f"target_{self.horizon}d")
