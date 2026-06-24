"""Tests for PortfolioBacktester price-cache optimisation.

The pre-fix ``_portfolio_value`` did an O(N) ``date in ohlcv_t.index``
membership test for every (day, ticker) pair inside the main backtest
loop. With K tickers and N days this made the hot path O(K * N^2).
The fix precomputes a ``{date: {ticker: close}}`` lookup table once,
so each ``_portfolio_value`` call is O(K) and the overall backtest is
O(K * N).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from island_model.portfolio_backtester import PortfolioBacktester, TradingCosts


# ---------- helpers ----------


def _make_ticker_datasets(
    tickers=("SBER", "GAZP", "LKOH"),
    n_days=120,
    start="2024-01-01",
    base_price=100.0,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """Build a minimal but realistic ticker_datasets structure."""
    dates = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.RandomState(42)
    out: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    for ticker in tickers:
        n = len(dates)
        close = base_price * np.exp(np.cumsum(rng.randn(n) * 0.01))
        ohlcv = pd.DataFrame(
            {
                "open": close + rng.randn(n) * 0.1,
                "high": close + abs(rng.randn(n)) * 0.5,
                "low": close - abs(rng.randn(n)) * 0.5,
                "close": close,
                "volume": rng.randint(100_000, 1_000_000, size=n),
            },
            index=dates,
        )
        X = pd.DataFrame(
            {"feat_a": rng.randn(n), "feat_b": rng.randn(n)},
            index=dates,
        )
        y = pd.Series(rng.randint(0, 2, size=n), index=dates, name="target")
        out[ticker] = (X, y, ohlcv)
    return out


# ---------- _build_price_cache ----------


def test_build_price_cache_returns_date_to_ticker_prices() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER", "GAZP"), n_days=10)
    cache = PortfolioBacktester._build_price_cache(datasets)
    assert isinstance(cache, dict)
    assert len(cache) == 10  # one entry per date in either ohlcv frame
    sample_date = next(iter(cache))
    assert isinstance(sample_date, pd.Timestamp)
    assert set(cache[sample_date].keys()) == {"SBER", "GAZP"}
    for ticker, close in cache[sample_date].items():
        assert isinstance(close, float)
        assert close > 0


def test_build_price_cache_uses_actual_close_column() -> None:
    """The cache must reflect the 'close' column, not e.g. 'open' or 'high'."""
    datasets = _make_ticker_datasets(tickers=("SBER",), n_days=5)
    cache = PortfolioBacktester._build_price_cache(datasets)
    sample_date = next(iter(cache))
    expected = float(datasets["SBER"][2]["close"].iloc[0])
    assert cache[sample_date]["SBER"] == pytest.approx(expected)


def test_build_price_cache_skips_nan_prices() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER",), n_days=5)
    # Inject a NaN into the close column
    datasets["SBER"][2].loc[datasets["SBER"][2].index[2], "close"] = np.nan
    cache = PortfolioBacktester._build_price_cache(datasets)
    # The NaN date must be absent (or have no entry for SBER)
    for d, prices in cache.items():
        for ticker, close in prices.items():
            assert not np.isnan(close), f"NaN leaked into cache for {ticker} on {d}"


def test_build_price_cache_handles_empty_ohlcv() -> None:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    cache = PortfolioBacktester._build_price_cache({"SBER": (pd.DataFrame(), pd.Series(), empty)})
    assert cache == {}


# ---------- _portfolio_value (cache-backed) ----------


def test_portfolio_value_requires_cache() -> None:
    """Direct calls without going through run() must fail loudly."""
    bt = PortfolioBacktester()
    with pytest.raises(RuntimeError, match="_price_cache"):
        bt._portfolio_value({"SBER": 10}, pd.Timestamp("2024-01-01"))


def test_portfolio_value_returns_zero_for_unknown_date() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER",), n_days=5)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)
    value = bt._portfolio_value({"SBER": 100}, pd.Timestamp("1999-01-01"))
    assert value == 0.0


def test_portfolio_value_uses_cache_correctly() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER", "GAZP"), n_days=5)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)
    date = next(iter(bt._price_cache))
    positions = {"SBER": 100, "GAZP": 50}
    expected = (
        100 * bt._price_cache[date]["SBER"]
        + 50 * bt._price_cache[date]["GAZP"]
    )
    assert bt._portfolio_value(positions, date) == pytest.approx(expected)


def test_portfolio_value_ignores_zero_or_negative_shares() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER",), n_days=5)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)
    date = next(iter(bt._price_cache))
    assert bt._portfolio_value({"SBER": 0}, date) == 0.0
    assert bt._portfolio_value({"SBER": -10}, date) == 0.0


def test_portfolio_value_skips_unknown_ticker() -> None:
    datasets = _make_ticker_datasets(tickers=("SBER",), n_days=5)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)
    date = next(iter(bt._price_cache))
    # Unknown ticker must not crash; just contribute 0 to the value
    value = bt._portfolio_value({"SBER": 100, "DOES_NOT_EXIST": 1_000_000}, date)
    sber_price = bt._price_cache[date]["SBER"]
    assert value == pytest.approx(100 * sber_price)


# ---------- algorithmic correctness vs the legacy slow path ----------


def test_cache_value_matches_legacy_recomputation() -> None:
    """The cache-backed value must equal what the old O(N) loop would compute."""
    datasets = _make_ticker_datasets(tickers=("SBER", "GAZP", "LKOH"), n_days=80)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)

    for date in list(bt._price_cache.keys())[:20]:
        positions = {"SBER": 7, "GAZP": 0, "LKOH": 3, "MISSING": 99}
        # Cache-backed answer (new)
        new_value = bt._portfolio_value(positions, date)
        # Hand-computed legacy answer (reference)
        ref_value = 0.0
        for ticker, shares in positions.items():
            if shares > 0 and ticker in datasets:
                _, _, ohlcv_t = datasets[ticker]
                if date in ohlcv_t.index:
                    ref_value += shares * ohlcv_t.loc[date, "close"]
        assert new_value == pytest.approx(ref_value), (
            f"value mismatch on {date}: cache={new_value} legacy={ref_value}"
        )


# ---------- algorithmic complexity assertion ----------


def test_portfolio_value_does_not_call_loc_in_inner_loop() -> None:
    """Behavioural assertion: the new _portfolio_value must NOT touch
    ohlcv_t.loc / ``date in ohlcv_t.index`` per (ticker, date) pair.

    We verify this by patching ``pandas.DataFrame.loc`` and
    ``pandas.Index.__contains__`` to count how many times the hot path
    touches them. The old implementation would call ``.loc`` once per
    held ticker per day; the new implementation calls neither at all
    (the cache is built once, separately, by ``_build_price_cache``).
    """
    datasets = _make_ticker_datasets(tickers=("SBER", "GAZP"), n_days=20)
    bt = PortfolioBacktester()
    bt._price_cache = PortfolioBacktester._build_price_cache(datasets)
    date = next(iter(bt._price_cache))

    loc_call_count = 0
    contains_call_count = 0

    real_loc = pd.DataFrame.loc
    real_contains = pd.Index.__contains__

    def counting_loc(self, *a, **kw):
        nonlocal loc_call_count
        loc_call_count += 1
        return real_loc.fget(self)(*a, **kw)  # type: ignore[attr-defined]

    def counting_contains(self, key):
        nonlocal contains_call_count
        contains_call_count += 1
        return real_contains(self, key)

    # Run the hot path N times and observe access counts.
    positions = {"SBER": 5, "GAZP": 3}
    with patch.object(pd.DataFrame, "loc", counting_loc), \
         patch.object(pd.Index, "__contains__", counting_contains):
        for _ in range(50):
            bt._portfolio_value(positions, date)

    assert loc_call_count == 0, (
        f"_portfolio_value touched DataFrame.loc {loc_call_count} times — "
        "the cache isn't being used."
    )
    assert contains_call_count == 0, (
        f"_portfolio_value touched Index.__contains__ {contains_call_count} times — "
        "the membership test is still in the hot path."
    )


# ---------- run() populates the cache ----------


def test_run_populates_price_cache_before_loop() -> None:
    """``run()`` must populate the cache so the hot path is fast."""
    from sklearn.ensemble import RandomForestClassifier

    datasets = _make_ticker_datasets(tickers=("SBER", "GAZP"), n_days=400)
    bt = PortfolioBacktester(
        initial_capital=10_000,
        train_days=120,
        retrain_every=40,
        embargo_days=2,
        costs=TradingCosts(),
        max_position_pct=0.25,
    )

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0, n_jobs=1)
    bt.run(model, datasets, model_name="RF")

    assert hasattr(bt, "_price_cache")
    assert isinstance(bt._price_cache, dict)
    assert len(bt._price_cache) > 0
    # Every cached date must have at least one ticker's price
    for date, prices in bt._price_cache.items():
        assert isinstance(date, pd.Timestamp)
        assert isinstance(prices, dict)
        assert len(prices) > 0
        for ticker, close in prices.items():
            assert ticker in datasets
            assert isinstance(close, float)
            assert close > 0