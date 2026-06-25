# портфельный бэктестер с комиссиями MOEX

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone


@dataclass
class TradingCosts:
    """Торговые издержки MOEX."""
    broker_commission: float = 0.0003   # 0.03% — типичный тариф (Тинькофф Инвестиции)
    exchange_fee: float = 0.0001        # 0.01% — биржевой сбор MOEX
    slippage_bps: float = 5.0           # базовое проскальзывание в б.п.
    execution_delay_days: int = 1       # задержка исполнения в торговых днях

    @property
    def total_commission(self) -> float:
        return self.broker_commission + self.exchange_fee

    def estimate_slippage(self, price, trade_volume, market_volume, spread_bps=3.0):
        """Оценка проскальзывания: полспреда + market impact."""
        half_spread = price * spread_bps / 10000 / 2
        if market_volume > 0:
            impact = price * 0.001 * np.sqrt(trade_volume / market_volume)
        else:
            impact = price * 0.001
        return half_spread + impact


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    side: str           # 'BUY' или 'SELL'
    shares: int         # количество акций
    price: float        # цена исполнения (включая slippage)
    commission: float   # комиссия
    slippage: float     # проскальзывание
    value: float        # полная стоимость


@dataclass
class PortfolioState:
    date: pd.Timestamp
    cash: float
    positions: dict[str, int]       # ticker → количество акций
    prices: dict[str, float]        # ticker → текущая цена
    equity: float                   # cash + сумма(позиция × цена)
    daily_pnl: float
    cumulative_return: float


@dataclass
class PortfolioBacktestResult:
    initial_capital: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[PortfolioState] = field(default_factory=list)
    signals_log: list[dict] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        if not self.equity_curve:
            return self.initial_capital
        return self.equity_curve[-1].equity

    @property
    def total_return(self) -> float:
        return (self.final_equity / self.initial_capital) - 1

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_commissions(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage for t in self.trades)

    def daily_returns(self):
        if len(self.equity_curve) < 2:
            return np.array([])
        equities = [s.equity for s in self.equity_curve]
        return np.diff(equities) / equities[:-1]

    def sharpe_ratio(self, rf_annual=0.21):
        """Sharpe с безрисковой ставкой ЦБ."""
        ret = self.daily_returns()
        if len(ret) == 0 or np.std(ret) == 0:
            return 0.0
        rf_daily = rf_annual / 252
        excess = ret - rf_daily
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    def sortino_ratio(self, rf_annual=0.21):
        ret = self.daily_returns()
        if len(ret) == 0:
            return 0.0
        rf_daily = rf_annual / 252
        excess = ret - rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return np.mean(excess) / np.std(downside) * np.sqrt(252)

    def max_drawdown(self):
        if len(self.equity_curve) < 2:
            return 0.0
        equities = np.array([s.equity for s in self.equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        return float(np.min(drawdowns))

    def profit_factor(self):
        ret = self.daily_returns()
        gains = ret[ret > 0].sum()
        losses = abs(ret[ret < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def calmar_ratio(self):
        if len(self.equity_curve) < 2:
            return 0.0
        n_days = len(self.equity_curve)
        ann_ret = self.total_return * 252 / n_days
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return ann_ret / mdd

    def win_rate(self):
        ret = self.daily_returns()
        if len(ret) == 0:
            return 0.0
        return np.mean(ret > 0)

    def summary(self):
        n_days = len(self.equity_curve)
        ann_factor = 252 / n_days if n_days > 0 else 1
        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(self.final_equity, 2),
            "total_return": f"{self.total_return:.2%}",
            "annualized_return": f"{(1 + self.total_return) ** ann_factor - 1:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "sortino_ratio": round(self.sortino_ratio(), 3),
            "max_drawdown": f"{self.max_drawdown():.2%}",
            "calmar_ratio": round(self.calmar_ratio(), 3),
            "profit_factor": round(self.profit_factor(), 3),
            "win_rate": f"{self.win_rate():.2%}",
            "n_trades": self.n_trades,
            "total_commissions": round(self.total_commissions, 2),
            "total_slippage": round(self.total_slippage, 2),
            "trading_days": n_days,
        }


class PortfolioBacktester:
    """Walk-forward бэктестер: обучение → сигнал → исполнение с задержкой."""

    def __init__(
        self,
        initial_capital=100_000,
        train_days=252,
        retrain_every=21,  # переобучение раз в месяц
        embargo_days=5,
        costs=None,
        max_position_pct=0.25,  # максимум 25% на один тикер
        lot_sizes=None,
    ):
        self.initial_capital = initial_capital
        self.train_days = train_days
        self.retrain_every = retrain_every
        self.embargo_days = embargo_days
        self.costs = costs or TradingCosts()
        self.max_position_pct = max_position_pct
        self.lot_sizes = lot_sizes or {}

    def run(self, model, ticker_datasets, model_name="Model"):
        """Запуск бэктеста: model — sklearn-модель, ticker_datasets — {ticker: (X, y, ohlcv)}."""
        result = PortfolioBacktestResult(initial_capital=self.initial_capital)
        cash = self.initial_capital
        positions = {t: 0 for t in ticker_datasets}  # акции (не лоты)
        current_model = None
        last_train_day = -999

        # Определяем общий торговый диапазон
        all_dates = set()
        for ticker, (X, y, ohlcv) in ticker_datasets.items():
            all_dates.update(X.index)
        all_dates = sorted(all_dates)

        if len(all_dates) < self.train_days + self.embargo_days + 10:
            print(f"    Недостаточно данных для бэктеста ({len(all_dates)} дней)")
            return result

        # Precompute a per-day close-price lookup table so the hot
        # _portfolio_value() path inside the main day loop runs in O(K)
        # instead of O(K * log N) per call (the old ``date in ohlcv_t.index``
        # path was O(N) for unsorted DatetimeIndex, making the whole
        # backtest O(K * N^2) on this method alone).
        self._price_cache: dict[pd.Timestamp, dict[str, float]] = self._build_price_cache(
            ticker_datasets
        )

        # Начало торговли — после первого обучения
        start_idx = self.train_days + self.embargo_days + self.costs.execution_delay_days + 1

        prev_equity = self.initial_capital

        for day_idx in range(start_idx, len(all_dates)):
            current_date = all_dates[day_idx]

            # Переобучение модели (если пришло время)
            if day_idx - last_train_day >= self.retrain_every or current_model is None:
                train_end = day_idx - self.embargo_days - self.costs.execution_delay_days
                train_start = max(0, train_end - self.train_days)

                if train_end - train_start < 60:
                    continue

                # Собираем тренировочные данные по всем тикерам
                X_trains = []
                y_trains = []

                for ticker, (X, y, ohlcv) in ticker_datasets.items():
                    train_dates = [d for d in all_dates[train_start:train_end] if d in X.index]
                    valid_dates = [d for d in train_dates if d in y.index and not np.isnan(y.loc[d])]
                    if len(valid_dates) < 30:
                        continue
                    X_trains.append(X.loc[valid_dates])
                    y_trains.append(y.loc[valid_dates])

                if not X_trains:
                    continue

                X_train = pd.concat(X_trains)
                y_train = pd.concat(y_trains)

                if len(np.unique(y_train)) < 2:
                    continue

                current_model = clone(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    current_model.fit(X_train.values, y_train.values)
                last_train_day = day_idx

            if current_model is None:
                continue

            # Генерация сигналов (по данным до current_date - delay)
            signal_date_idx = day_idx - self.costs.execution_delay_days
            if signal_date_idx < 0 or signal_date_idx >= len(all_dates):
                continue
            signal_date = all_dates[signal_date_idx]

            signals = {}
            for ticker, (X, y, ohlcv) in ticker_datasets.items():
                if signal_date not in X.index:
                    signals[ticker] = 0.5  # нет данных — нейтральный сигнал
                    continue
                features = X.loc[[signal_date]].values
                try:
                    proba = current_model.predict_proba(features)[0]
                    signals[ticker] = proba[1] if len(proba) > 1 else 0.5
                except Exception:
                    signals[ticker] = 0.5

            # Целевые позиции
            # long если P(up) > 0.55, cash если P(up) < 0.45 (без шортов на MOEX)
            target_positions = {}
            n_longs = sum(1 for s in signals.values() if s > 0.55)

            for ticker, (X, y, ohlcv) in ticker_datasets.items():
                if current_date not in ohlcv.index:
                    target_positions[ticker] = 0
                    continue

                current_price = ohlcv.loc[current_date, "open"]  # исполнение по open
                lot_size = self.lot_sizes.get(ticker, 1)

                if signals.get(ticker, 0.5) > 0.55 and n_longs > 0:
                    # Аллокация: равномерная среди long-сигналов, с ограничением на тикер
                    alloc = min(1.0 / n_longs, self.max_position_pct)
                    target_value = (cash + self._portfolio_value(positions, current_date)) * alloc
                    target_shares = int(target_value / (current_price * lot_size)) * lot_size
                    target_positions[ticker] = target_shares
                else:
                    target_positions[ticker] = 0

            # Исполнение сделок
            for ticker, (X, y, ohlcv) in ticker_datasets.items():
                if current_date not in ohlcv.index:
                    continue

                current_shares = positions.get(ticker, 0)
                target_shares = target_positions.get(ticker, 0)
                delta = target_shares - current_shares

                if delta == 0:
                    continue

                exec_price = ohlcv.loc[current_date, "open"]
                market_vol = ohlcv.loc[current_date, "volume"]
                spread_bps = 3.0  # средний спред

                # Проскальзывание
                trade_value = abs(delta) * exec_price
                slippage = self.costs.estimate_slippage(
                    exec_price, abs(delta), market_vol, spread_bps
                )

                # Цена исполнения с проскальзыванием
                if delta > 0:  # покупка
                    fill_price = exec_price + slippage
                else:  # продажа
                    fill_price = exec_price - slippage

                fill_price = max(fill_price, 0.01)
                trade_cost = abs(delta) * fill_price
                commission = trade_cost * self.costs.total_commission

                if delta > 0:  # покупка
                    total_cost = trade_cost + commission
                    if total_cost > cash:
                        # Уменьшаем позицию до доступных средств
                        lot_size = self.lot_sizes.get(ticker, 1)
                        affordable = int((cash - commission) / fill_price / lot_size) * lot_size
                        if affordable <= 0:
                            continue
                        delta = affordable
                        trade_cost = delta * fill_price
                        commission = trade_cost * self.costs.total_commission
                        total_cost = trade_cost + commission
                    cash -= total_cost
                else:  # продажа
                    proceeds = trade_cost - commission
                    cash += proceeds

                positions[ticker] = current_shares + delta

                trade = Trade(
                    date=current_date,
                    ticker=ticker,
                    side="BUY" if delta > 0 else "SELL",
                    shares=abs(delta),
                    price=fill_price,
                    commission=commission,
                    slippage=abs(slippage) * abs(delta),
                    value=trade_cost,
                )
                result.trades.append(trade)

            # Оценка портфеля на конец дня
            portfolio_value = cash
            current_prices = {}
            for ticker, (X, y, ohlcv) in ticker_datasets.items():
                if current_date in ohlcv.index:
                    price = ohlcv.loc[current_date, "close"]
                    current_prices[ticker] = price
                    portfolio_value += positions.get(ticker, 0) * price

            daily_pnl = portfolio_value - prev_equity
            cum_return = portfolio_value / self.initial_capital - 1

            state = PortfolioState(
                date=current_date,
                cash=cash,
                positions=dict(positions),
                prices=current_prices,
                equity=portfolio_value,
                daily_pnl=daily_pnl,
                cumulative_return=cum_return,
            )
            result.equity_curve.append(state)
            prev_equity = portfolio_value

            # Лог сигналов
            result.signals_log.append({
                "date": current_date,
                "signals": dict(signals),
                "positions": dict(positions),
                "equity": portfolio_value,
                "cash": cash,
            })

        return result

    @staticmethod
    def _build_price_cache(
        ticker_datasets: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    ) -> dict[pd.Timestamp, dict[str, float]]:
        """Precompute ``{date: {ticker: close_price}}`` for O(1) per-day lookups.

        Building this once turns every subsequent per-day, per-ticker price
        query inside ``_portfolio_value`` into a dict lookup, instead of an
        O(N) ``date in ohlcv_t.index`` membership test.
        """
        cache: dict[pd.Timestamp, dict[str, float]] = {}
        for ticker, (_X, _y, ohlcv) in ticker_datasets.items():
            if ohlcv.empty or "close" not in ohlcv.columns:
                continue
            for date, close in ohlcv["close"].items():
                if pd.notna(close):
                    cache.setdefault(date, {})[ticker] = float(close)
        return cache

    def _portfolio_value(
        self,
        positions: dict[str, int],
        date: pd.Timestamp,
    ) -> float:
        """Current value of positions on ``date`` (uses the cache from ``run()``).

        Runs in O(K) where K is the number of held tickers. Requires
        ``self._price_cache`` to be populated — ``run()`` does this
        automatically before the main loop. Direct external callers
        must populate the cache themselves or use ``_build_price_cache``.
        """
        cache = getattr(self, "_price_cache", None)
        if not cache:
            raise RuntimeError(
                "_portfolio_value requires _price_cache to be populated; "
                "call backtester.run() first, or backtester._build_price_cache(...) "
                "to populate the cache manually."
            )
        day_prices = cache.get(date)
        if not day_prices:
            return 0.0
        value = 0.0
        for ticker, shares in positions.items():
            if shares > 0:
                close = day_prices.get(ticker)
                if close is not None:
                    value += shares * close
        return value


def run_buy_and_hold_portfolio(
    initial_capital: float,
    ticker_datasets: dict,
    costs: "TradingCosts",
    lot_sizes: dict[str, int],
) -> "PortfolioBacktestResult":
    """Buy & Hold бенчмарк: покупаем в первый день, держим до конца."""
    result = PortfolioBacktestResult(initial_capital=initial_capital)
    cash = initial_capital
    positions = {}

    # Определяем первый торговый день
    all_dates = set()
    for ticker, (X, y, ohlcv) in ticker_datasets.items():
        all_dates.update(ohlcv.index)
    all_dates = sorted(all_dates)

    if not all_dates:
        return result

    # Покупаем равномерно в первый день
    first_date = all_dates[0]
    n_tickers = len(ticker_datasets)
    alloc_per_ticker = initial_capital / n_tickers

    for ticker, (X, y, ohlcv) in ticker_datasets.items():
        if first_date not in ohlcv.index:
            continue
        price = ohlcv.loc[first_date, "open"]
        lot_size = lot_sizes.get(ticker, 1)
        shares = int(alloc_per_ticker / (price * lot_size)) * lot_size
        if shares <= 0:
            continue
        cost = shares * price
        commission = cost * costs.total_commission
        slippage_val = costs.estimate_slippage(price, shares, ohlcv.loc[first_date, "volume"])
        total_cost = shares * (price + slippage_val) + commission
        if total_cost > cash:
            shares = int((cash * 0.99) / ((price + slippage_val) * lot_size)) * lot_size
            if shares <= 0:
                continue
            total_cost = shares * (price + slippage_val) + shares * price * costs.total_commission
        cash -= total_cost
        positions[ticker] = shares

        result.trades.append(Trade(
            date=first_date, ticker=ticker, side="BUY", shares=shares,
            price=price + slippage_val,
            commission=shares * price * costs.total_commission,
            slippage=slippage_val * shares,
            value=shares * price,
        ))

    # Отслеживаем equity
    prev_equity = initial_capital
    for date in all_dates:
        portfolio_value = cash
        prices = {}
        for ticker, (X, y, ohlcv) in ticker_datasets.items():
            if date in ohlcv.index:
                p = ohlcv.loc[date, "close"]
                prices[ticker] = p
                portfolio_value += positions.get(ticker, 0) * p

        daily_pnl = portfolio_value - prev_equity
        cum_return = portfolio_value / initial_capital - 1

        result.equity_curve.append(PortfolioState(
            date=date, cash=cash, positions=dict(positions),
            prices=prices, equity=portfolio_value,
            daily_pnl=daily_pnl, cumulative_return=cum_return,
        ))
        prev_equity = portfolio_value

    return result
