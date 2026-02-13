#!/usr/bin/env python3
"""
Бэктест инвестиционных стратегий, основанных на ключевой ставке ЦБ РФ.

Гипотеза: изменения ключевой ставки влияют на рынок акций.
- Снижение ставки → акции растут (дешёвые деньги, снижение доходности депозитов)
- Повышение ставки → акции падают (дорогие кредиты, привлекательные депозиты)

Тестируемые стратегии:
1. Rate Direction    — в акциях при цикле снижения, в кэше при повышении
2. Rate Level        — аллокация обратно пропорциональна уровню ставки
3. Rate Momentum     — входим когда 2+ снижения подряд, выходим при 2+ повышениях
4. Contrarian        — покупаем при панических повышениях (>2 п.п.), продаём при эйфории
5. Composite         — комбинация сигналов стратегий 1-4
6. Buy & Hold        — бенчмарк

Без look-ahead bias: решение принимается ПОСЛЕ объявления ставки,
исполнение по open следующего торгового дня.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Исторические данные ключевой ставки ЦБ РФ
# Источник: cbr.ru/hd_base/KeyRate/
# ─────────────────────────────────────────────────────────────────────────────

CBR_KEY_RATE_HISTORY = [
    # (дата вступления в силу, ставка в %)
    ("2013-09-13", 5.50),
    ("2014-03-03", 7.00),
    ("2014-04-28", 7.50),
    ("2014-07-28", 8.00),
    ("2014-11-05", 9.50),
    ("2014-12-12", 10.50),
    ("2014-12-16", 17.00),  # экстренное повышение
    ("2015-02-02", 15.00),
    ("2015-03-16", 14.00),
    ("2015-05-05", 12.50),
    ("2015-06-16", 11.50),
    ("2015-08-03", 11.00),
    ("2016-06-14", 10.50),
    ("2016-09-19", 10.00),
    ("2017-03-27", 9.75),
    ("2017-05-02", 9.25),
    ("2017-06-19", 9.00),
    ("2017-09-18", 8.50),
    ("2017-10-30", 8.25),
    ("2017-12-18", 7.75),
    ("2018-02-12", 7.50),
    ("2018-03-26", 7.25),
    ("2018-09-17", 7.50),
    ("2018-12-17", 7.75),
    ("2019-06-17", 7.50),
    ("2019-07-29", 7.25),
    ("2019-09-09", 7.00),
    ("2019-10-28", 6.50),
    ("2019-12-16", 6.25),
    ("2020-02-10", 6.00),
    ("2020-04-27", 5.50),
    ("2020-06-22", 4.50),
    ("2020-07-27", 4.25),
    ("2021-03-22", 4.50),
    ("2021-04-26", 5.00),
    ("2021-06-15", 5.50),
    ("2021-07-26", 6.50),
    ("2021-09-13", 6.75),
    ("2021-10-25", 7.50),
    ("2021-12-20", 8.50),
    ("2022-02-28", 20.00),  # экстренное повышение (СВО)
    ("2022-04-11", 17.00),
    ("2022-05-04", 14.00),
    ("2022-06-14", 9.50),
    ("2022-07-25", 8.00),
    ("2022-09-19", 7.50),
    ("2023-07-24", 8.50),
    ("2023-08-15", 12.00),
    ("2023-09-18", 13.00),
    ("2023-10-30", 15.00),
    ("2023-12-18", 16.00),
    ("2024-07-29", 18.00),
    ("2024-10-28", 21.00),
]


def build_key_rate_series(start_date: str, end_date: str) -> pd.Series:
    """Построить дневной ряд ключевой ставки ЦБ."""
    dates = []
    rates = []
    for date_str, rate in CBR_KEY_RATE_HISTORY:
        dates.append(pd.Timestamp(date_str))
        rates.append(rate)

    rate_df = pd.DataFrame({"rate": rates}, index=dates)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    daily_rate = rate_df["rate"].reindex(date_range, method="ffill")

    # Заполняем начальные NaN первым известным значением до start_date
    if daily_rate.isna().any():
        first_valid = daily_rate.first_valid_index()
        if first_valid is not None:
            daily_rate = daily_rate.ffill().bfill()

    return daily_rate


def get_rate_changes(rate_series: pd.Series) -> pd.DataFrame:
    """Определить даты и направления изменений ключевой ставки."""
    changes = []
    for date_str, rate in CBR_KEY_RATE_HISTORY:
        dt = pd.Timestamp(date_str)
        changes.append({"date": dt, "rate": rate})

    df = pd.DataFrame(changes)
    df["prev_rate"] = df["rate"].shift(1)
    df["change"] = df["rate"] - df["prev_rate"]
    df["direction"] = np.where(df["change"] > 0, "hike",
                               np.where(df["change"] < 0, "cut", "hold"))
    df = df.dropna(subset=["prev_rate"])
    df = df.set_index("date")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных акций
# ─────────────────────────────────────────────────────────────────────────────

def load_stock_data(tickers: list[str], data_dir: str) -> dict[str, pd.DataFrame]:
    """Загрузка дневных OHLCV из CSV."""
    ticker_data = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}_daily.csv")
        if not os.path.exists(path):
            print(f"  SKIP {ticker}: файл не найден")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]]
        ticker_data[ticker] = df
        print(f"  {ticker}: {len(df)} дней "
              f"({df.index[0].date()} — {df.index[-1].date()}) "
              f"close {df['close'].iloc[0]:.1f} → {df['close'].iloc[-1]:.1f}")
    return ticker_data


# ─────────────────────────────────────────────────────────────────────────────
# Результат бэктеста
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Результат бэктеста одной стратегии."""
    name: str
    initial_capital: float
    equity_curve: list = field(default_factory=list)   # [(date, equity)]
    trades: list = field(default_factory=list)          # [(date, action, ticker, shares, price)]
    allocation_history: list = field(default_factory=list)  # [(date, stock_pct)]
    rate_at_trades: list = field(default_factory=list)  # [(date, rate, change)]

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.initial_capital

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_capital - 1

    def daily_returns(self) -> np.ndarray:
        if len(self.equity_curve) < 2:
            return np.array([])
        equities = [e[1] for e in self.equity_curve]
        return np.diff(equities) / equities[:-1]

    def sharpe_ratio(self, rf_annual: float = 0.10) -> float:
        ret = self.daily_returns()
        if len(ret) == 0 or np.std(ret) == 0:
            return 0.0
        rf_daily = rf_annual / 252
        excess = ret - rf_daily
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    def sortino_ratio(self, rf_annual: float = 0.10) -> float:
        ret = self.daily_returns()
        if len(ret) == 0:
            return 0.0
        rf_daily = rf_annual / 252
        excess = ret - rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return float(np.mean(excess) / np.std(downside) * np.sqrt(252))

    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        equities = np.array([e[1] for e in self.equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        return float(np.min(drawdowns))

    def calmar_ratio(self) -> float:
        n_days = len(self.equity_curve)
        if n_days < 2:
            return 0.0
        ann_ret = self.total_return * 252 / n_days
        mdd = abs(self.max_drawdown())
        return ann_ret / mdd if mdd > 0 else 0.0

    def annual_return(self) -> float:
        n_days = len(self.equity_curve)
        if n_days < 2:
            return 0.0
        return (1 + self.total_return) ** (252 / n_days) - 1

    def win_rate(self) -> float:
        ret = self.daily_returns()
        if len(ret) == 0:
            return 0.0
        return float(np.mean(ret > 0))

    def summary(self) -> dict:
        return {
            "Стратегия": self.name,
            "Итого капитал": f"{self.final_equity:,.0f}",
            "Общий доход": f"{self.total_return:+.2%}",
            "Годовой доход": f"{self.annual_return():+.2%}",
            "Sharpe": f"{self.sharpe_ratio():.3f}",
            "Sortino": f"{self.sortino_ratio():.3f}",
            "Max DD": f"{self.max_drawdown():.2%}",
            "Calmar": f"{self.calmar_ratio():.3f}",
            "Win Rate": f"{self.win_rate():.2%}",
            "Сделок": len(self.trades),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Торговые издержки
# ─────────────────────────────────────────────────────────────────────────────

BROKER_COMMISSION = 0.0003  # 0.03%
EXCHANGE_FEE = 0.0001       # 0.01%
SLIPPAGE_BPS = 5.0          # 5 базисных пунктов
TOTAL_COST_PER_SIDE = BROKER_COMMISSION + EXCHANGE_FEE + SLIPPAGE_BPS / 10000  # ~0.09%

LOT_SIZES = {
    "SBER": 10, "GAZP": 10, "LKOH": 1, "GMKN": 1, "ROSN": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Движок бэктеста
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(
    name: str,
    signal_func,
    ticker_data: dict[str, pd.DataFrame],
    rate_series: pd.Series,
    rate_changes: pd.DataFrame,
    initial_capital: float = 1_000_000,
) -> StrategyResult:
    """
    Универсальный движок бэктеста.

    signal_func(date, rate_series, rate_changes, current_positions) -> target_stock_pct
    target_stock_pct: 0.0 = 100% кэш, 1.0 = 100% акции

    Без look-ahead: сигнал на день T → исполнение по open T+1.
    Аллокация: равномерная по всем тикерам.
    Кэш «зарабатывает» по ставке ЦБ (депозит/RUSFAR-аналог).
    """
    result = StrategyResult(name=name, initial_capital=initial_capital)

    # Общий набор торговых дат
    all_dates = sorted(set().union(*[df.index for df in ticker_data.values()]))
    all_dates = [d for d in all_dates if d in rate_series.index]

    if len(all_dates) < 10:
        print(f"  {name}: недостаточно данных")
        return result

    cash = initial_capital
    positions = {t: 0 for t in ticker_data}  # акции
    current_stock_pct = 0.0  # текущая доля акций
    pending_signal = None    # (target_stock_pct, signal_date)

    for i, date in enumerate(all_dates):
        # ─── Начисление % на кэш (дневная ставка ЦБ) ───
        if cash > 0 and i > 0:
            daily_rate = rate_series.loc[date] / 100 / 365
            cash *= (1 + daily_rate)

        # ─── Исполнение отложенного сигнала (по open текущего дня) ───
        if pending_signal is not None:
            target_pct, signal_date = pending_signal
            pending_signal = None

            # Текущая стоимость портфеля
            portfolio_value = cash
            for ticker, shares in positions.items():
                if shares > 0 and date in ticker_data[ticker].index:
                    portfolio_value += shares * ticker_data[ticker].loc[date, "open"]

            # Целевая стоимость акций
            target_stock_value = portfolio_value * target_pct
            current_stock_value = 0
            for ticker, shares in positions.items():
                if shares > 0 and date in ticker_data[ticker].index:
                    current_stock_value += shares * ticker_data[ticker].loc[date, "open"]

            delta_value = target_stock_value - current_stock_value

            if abs(delta_value) > portfolio_value * 0.02:  # ребалансировка если delta > 2%
                n_tickers = len(ticker_data)
                target_per_ticker = target_stock_value / n_tickers if n_tickers > 0 else 0

                for ticker, df in ticker_data.items():
                    if date not in df.index:
                        continue

                    price = df.loc[date, "open"]
                    lot_size = LOT_SIZES.get(ticker, 1)
                    current_shares = positions[ticker]
                    current_value = current_shares * price
                    target_shares = int(target_per_ticker / (price * lot_size)) * lot_size

                    delta_shares = target_shares - current_shares
                    if delta_shares == 0:
                        continue

                    # Издержки
                    trade_value = abs(delta_shares) * price
                    cost = trade_value * TOTAL_COST_PER_SIDE

                    if delta_shares > 0:  # покупка
                        total_cost = trade_value + cost
                        if total_cost > cash:
                            affordable = int((cash - cost) / (price * lot_size)) * lot_size
                            if affordable <= 0:
                                continue
                            delta_shares = affordable
                            trade_value = delta_shares * price
                            cost = trade_value * TOTAL_COST_PER_SIDE
                            total_cost = trade_value + cost
                        cash -= total_cost
                    else:  # продажа
                        proceeds = trade_value - cost
                        cash += proceeds

                    positions[ticker] = current_shares + delta_shares
                    result.trades.append((date, "BUY" if delta_shares > 0 else "SELL",
                                         ticker, abs(delta_shares), price, cost))

            current_stock_pct = target_pct

        # ─── Оценка портфеля на close ───
        portfolio_value = cash
        for ticker, shares in positions.items():
            if shares > 0 and date in ticker_data[ticker].index:
                portfolio_value += shares * ticker_data[ticker].loc[date, "close"]

        result.equity_curve.append((date, portfolio_value))
        result.allocation_history.append((date, current_stock_pct))

        # ─── Генерация сигнала на СЛЕДУЮЩИЙ день ───
        target_pct = signal_func(date, rate_series, rate_changes, positions)
        if target_pct is not None and abs(target_pct - current_stock_pct) > 0.05:
            pending_signal = (target_pct, date)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Стратегии
# ─────────────────────────────────────────────────────────────────────────────

def strategy_buy_and_hold(date, rate_series, rate_changes, positions):
    """Бенчмарк: 100% акции всегда."""
    return 1.0


def strategy_rate_direction(date, rate_series, rate_changes, positions):
    """
    Стратегия 1: Направление ставки.
    - Если последнее изменение = снижение → 100% акции
    - Если последнее изменение = повышение → 0% акции (кэш под ставку ЦБ)
    - Если ставка не менялась давно → 50% акции
    """
    past_changes = rate_changes[rate_changes.index <= date]
    if past_changes.empty:
        return 0.5

    last_change = past_changes.iloc[-1]
    if last_change["direction"] == "cut":
        return 1.0
    elif last_change["direction"] == "hike":
        return 0.0
    return 0.5


def strategy_rate_level(date, rate_series, rate_changes, positions):
    """
    Стратегия 2: Уровень ставки.
    Аллокация в акции обратно пропорциональна уровню ставки.
    - Ставка <= 5%  → 100% акции
    - Ставка 5-8%   → 75% акции
    - Ставка 8-12%  → 50% акции
    - Ставка 12-16% → 25% акции
    - Ставка > 16%  → 0% акции (депозит выгоднее)
    """
    current_rate = rate_series.loc[date] if date in rate_series.index else 7.5

    if current_rate <= 5.0:
        return 1.0
    elif current_rate <= 8.0:
        return 0.75
    elif current_rate <= 12.0:
        return 0.50
    elif current_rate <= 16.0:
        return 0.25
    else:
        return 0.0


def strategy_rate_momentum(date, rate_series, rate_changes, positions):
    """
    Стратегия 3: Momentum ставки.
    Смотрим на последние N изменений:
    - 2+ снижения подряд → 100% акции (устойчивый цикл смягчения)
    - 2+ повышения подряд → 0% акции (устойчивый цикл ужесточения)
    - Иначе → 50% акции
    """
    past_changes = rate_changes[rate_changes.index <= date]
    if len(past_changes) < 2:
        return 0.5

    last_two = past_changes.tail(2)
    directions = last_two["direction"].tolist()

    if all(d == "cut" for d in directions):
        return 1.0
    elif all(d == "hike" for d in directions):
        return 0.0
    else:
        return 0.5


def strategy_contrarian(date, rate_series, rate_changes, positions):
    """
    Стратегия 4: Контрарианская.
    Покупаем при панических повышениях (рынок уже упал, ставка на пике).
    Продаём при эйфорических снижениях (рынок уже вырос).

    Логика:
    - Повышение > 2 п.п. за раз → 100% акции (паника = возможность)
    - Любое повышение, но ставка > 15% → 80% акции (ставка близка к пику)
    - Снижение, ставка < 5% → 30% акции (ставка у дна, акции уже дорогие)
    - По умолчанию → 60% акции
    """
    past_changes = rate_changes[rate_changes.index <= date]
    if past_changes.empty:
        return 0.6

    last_change = past_changes.iloc[-1]
    current_rate = rate_series.loc[date] if date in rate_series.index else 7.5

    # Паническое повышение > 2 п.п.
    if last_change["change"] >= 2.0:
        return 1.0

    # Ставка очень высокая — скорее всего пик
    if current_rate >= 15.0 and last_change["direction"] == "hike":
        return 0.80

    # Ставка очень низкая — скорее всего дно
    if current_rate <= 5.0 and last_change["direction"] == "cut":
        return 0.30

    return 0.60


def strategy_composite(date, rate_series, rate_changes, positions):
    """
    Стратегия 5: Композитная.
    Средневзвешенный сигнал стратегий 1-4 с весами.
    """
    s1 = strategy_rate_direction(date, rate_series, rate_changes, positions) or 0.5
    s2 = strategy_rate_level(date, rate_series, rate_changes, positions) or 0.5
    s3 = strategy_rate_momentum(date, rate_series, rate_changes, positions) or 0.5
    s4 = strategy_contrarian(date, rate_series, rate_changes, positions) or 0.5

    # Веса: level и direction — основные, momentum и contrarian — вспомогательные
    weights = [0.30, 0.30, 0.20, 0.20]
    signals = [s1, s2, s3, s4]

    return sum(w * s for w, s in zip(weights, signals))


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list[StrategyResult], rate_series: pd.Series,
                 rate_changes: pd.DataFrame, output_path: str):
    """4-панельная визуализация."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1]})

    # ── Панель 1: Equity Curves ──
    ax1 = axes[0]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#607D8B"]
    for i, res in enumerate(results):
        dates = [e[0] for e in res.equity_curve]
        equities = [e[1] for e in res.equity_curve]
        style = "--" if res.name == "Buy & Hold" else "-"
        lw = 1.5 if res.name == "Buy & Hold" else 2.0
        ax1.plot(dates, equities, style, label=res.name,
                 color=colors[i % len(colors)], linewidth=lw)

    ax1.axhline(y=results[0].initial_capital, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Стратегии на основе ключевой ставки ЦБ РФ — Equity Curves", fontsize=14)
    ax1.set_ylabel("Капитал (руб.)")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ── Панель 2: Ключевая ставка ──
    ax2 = axes[1]
    rate_dates = rate_series.index
    ax2.fill_between(rate_dates, rate_series.values, alpha=0.3, color="#FF5722")
    ax2.plot(rate_dates, rate_series.values, color="#FF5722", linewidth=1.5)

    # Отметить повышения и снижения
    for date, row in rate_changes.iterrows():
        if date < rate_dates[0] or date > rate_dates[-1]:
            continue
        if row["direction"] == "hike":
            ax2.axvline(x=date, color="red", alpha=0.3, linestyle="--", linewidth=0.8)
        elif row["direction"] == "cut":
            ax2.axvline(x=date, color="green", alpha=0.3, linestyle="--", linewidth=0.8)

    ax2.set_title("Ключевая ставка ЦБ РФ (%)", fontsize=12)
    ax2.set_ylabel("Ставка (%)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ── Панель 3: Drawdown ──
    ax3 = axes[2]
    for i, res in enumerate(results):
        equities = np.array([e[1] for e in res.equity_curve])
        dates = [e[0] for e in res.equity_curve]
        running_max = np.maximum.accumulate(equities)
        drawdown = (equities - running_max) / running_max * 100
        style = "--" if res.name == "Buy & Hold" else "-"
        ax3.plot(dates, drawdown, style, label=res.name,
                 color=colors[i % len(colors)], linewidth=1.2, alpha=0.8)

    ax3.set_title("Просадка (%)", fontsize=12)
    ax3.set_ylabel("Drawdown %")
    ax3.legend(fontsize=9, loc="lower left")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ── Панель 4: Аллокация Composite стратегии ──
    ax4 = axes[3]
    composite = [r for r in results if r.name == "Composite"]
    if composite:
        alloc_dates = [a[0] for a in composite[0].allocation_history]
        alloc_pcts = [a[1] * 100 for a in composite[0].allocation_history]
        ax4.fill_between(alloc_dates, alloc_pcts, alpha=0.4, color="#9C27B0")
        ax4.plot(alloc_dates, alloc_pcts, color="#9C27B0", linewidth=1)
    ax4.set_title("Доля акций — стратегия Composite (%)", fontsize=12)
    ax4.set_ylabel("Акции %")
    ax4.set_ylim(-5, 105)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  График сохранён: {output_path}")


def plot_annual_breakdown(results: list[StrategyResult], output_path: str):
    """Доходность по годам для каждой стратегии."""
    years_data = {}

    for res in results:
        equity_df = pd.DataFrame(res.equity_curve, columns=["date", "equity"])
        equity_df = equity_df.set_index("date")
        equity_df.index = pd.to_datetime(equity_df.index)

        yearly_returns = {}
        for year in equity_df.index.year.unique():
            year_data = equity_df[equity_df.index.year == year]
            if len(year_data) < 10:
                continue
            year_return = year_data["equity"].iloc[-1] / year_data["equity"].iloc[0] - 1
            yearly_returns[year] = year_return * 100

        years_data[res.name] = yearly_returns

    all_years = sorted(set().union(*[set(v.keys()) for v in years_data.values()]))
    if not all_years:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_years))
    width = 0.13
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#607D8B"]

    for i, (name, returns) in enumerate(years_data.items()):
        values = [returns.get(y, 0) for y in all_years]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            if abs(val) > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Год")
    ax.set_ylabel("Доходность (%)")
    ax.set_title("Доходность стратегий по годам", fontsize=14)
    ax.set_xticks(x + width * len(years_data) / 2)
    ax.set_xticklabels(all_years)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  График по годам сохранён: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Основной запуск
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("БЭКТЕСТ: СТРАТЕГИИ ИНВЕСТИРОВАНИЯ НА ОСНОВЕ КЛЮЧЕВОЙ СТАВКИ ЦБ РФ")
    print("=" * 70)

    # Конфигурация
    TICKERS = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN"]
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    INITIAL_CAPITAL = 1_000_000  # 1 млн руб.
    START_DATE = "2019-01-01"
    END_DATE = "2024-12-31"

    # 1. Загрузка ставки ЦБ
    print("\n[1/5] КЛЮЧЕВАЯ СТАВКА ЦБ")
    print("-" * 60)
    rate_series = build_key_rate_series(START_DATE, END_DATE)
    rate_changes = get_rate_changes(rate_series)
    rate_changes_in_range = rate_changes[
        (rate_changes.index >= START_DATE) & (rate_changes.index <= END_DATE)
    ]
    print(f"  Период: {START_DATE} — {END_DATE}")
    print(f"  Ставка на начало: {rate_series.iloc[0]:.2f}%")
    print(f"  Ставка на конец:  {rate_series.iloc[-1]:.2f}%")
    print(f"  Изменений за период: {len(rate_changes_in_range)}")
    n_hikes = len(rate_changes_in_range[rate_changes_in_range["direction"] == "hike"])
    n_cuts = len(rate_changes_in_range[rate_changes_in_range["direction"] == "cut"])
    print(f"  Повышений: {n_hikes}, Снижений: {n_cuts}")
    print(f"  Минимум: {rate_series.min():.2f}%, Максимум: {rate_series.max():.2f}%")

    # 2. Загрузка акций
    print("\n[2/5] ЗАГРУЗКА ДАННЫХ АКЦИЙ")
    print("-" * 60)
    ticker_data = load_stock_data(TICKERS, DATA_DIR)
    if not ticker_data:
        print("  ОШИБКА: нет данных акций")
        sys.exit(1)

    # 3. Запуск стратегий
    print("\n[3/5] ЗАПУСК СТРАТЕГИЙ")
    print("-" * 60)

    strategies = [
        ("Rate Direction", strategy_rate_direction),
        ("Rate Level", strategy_rate_level),
        ("Rate Momentum", strategy_rate_momentum),
        ("Contrarian", strategy_contrarian),
        ("Composite", strategy_composite),
        ("Buy & Hold", strategy_buy_and_hold),
    ]

    results = []
    for name, func in strategies:
        print(f"  {name}...", end=" ", flush=True)
        res = run_strategy(
            name=name,
            signal_func=func,
            ticker_data=ticker_data,
            rate_series=rate_series,
            rate_changes=rate_changes,
            initial_capital=INITIAL_CAPITAL,
        )
        results.append(res)
        print(f"OK — капитал {res.final_equity:,.0f} руб. "
              f"({res.total_return:+.2%}), сделок: {len(res.trades)}")

    # 4. Результаты
    print("\n[4/5] ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    summary_rows = [r.summary() for r in results]
    df_summary = pd.DataFrame(summary_rows).set_index("Стратегия")
    print("\n" + df_summary.to_string())

    # Сравнительный анализ
    print(f"\n{'─' * 70}")
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
    print(f"{'─' * 70}")

    bnh = [r for r in results if r.name == "Buy & Hold"][0]
    print(f"\n  Buy & Hold доходность: {bnh.total_return:+.2%} "
          f"(годовая: {bnh.annual_return():+.2%})")
    print(f"  Buy & Hold max DD: {bnh.max_drawdown():.2%}")

    for r in results:
        if r.name == "Buy & Hold":
            continue
        excess = r.total_return - bnh.total_return
        dd_improvement = abs(bnh.max_drawdown()) - abs(r.max_drawdown())
        print(f"\n  {r.name}:")
        print(f"    vs Buy & Hold: {excess:+.2%} ({'+' if excess > 0 else ''}доходность)")
        print(f"    Улучшение DD: {dd_improvement:+.2%}")
        print(f"    Сделок: {len(r.trades)}, "
              f"Издержки: ~{len(r.trades) * INITIAL_CAPITAL * 0.001 * TOTAL_COST_PER_SIDE:,.0f} руб.")

    # Анализ по периодам
    print(f"\n{'─' * 70}")
    print("АНАЛИЗ ПО РЕЖИМАМ СТАВКИ:")
    print(f"{'─' * 70}")

    periods = [
        ("Снижение 2019", "2019-06-01", "2020-02-28",
         "Цикл снижения 7.75% → 6.00%"),
        ("COVID + снижение 2020", "2020-03-01", "2020-12-31",
         "Обвал + рекордное снижение до 4.25%"),
        ("Цикл повышения 2021", "2021-01-01", "2021-12-31",
         "Инфляция → рост ставки 4.25% → 8.50%"),
        ("Шок СВО + снижение 2022", "2022-01-01", "2022-12-31",
         "Экстренное 20% → быстрое снижение до 7.50%"),
        ("Пауза + повышение 2023", "2023-01-01", "2023-12-31",
         "Стабильность → резкий рост 7.50% → 16%"),
        ("Жёсткая ДКП 2024", "2024-01-01", "2024-12-31",
         "Ставка 16% → 21%, рекордные уровни"),
    ]

    for period_name, start, end, description in periods:
        print(f"\n  {period_name} ({description}):")
        for r in results:
            ec = [(d, e) for d, e in r.equity_curve
                  if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
            if len(ec) < 5:
                continue
            period_ret = ec[-1][1] / ec[0][1] - 1
            print(f"    {r.name:20s}: {period_ret:+.2%}")

    # 5. Графики
    print(f"\n[5/5] ВИЗУАЛИЗАЦИЯ")
    print("-" * 60)
    try:
        plot_results(results, rate_series, rate_changes,
                     "/home/user/Moex/equity_key_rate.png")
        plot_annual_breakdown(results, "/home/user/Moex/equity_key_rate_annual.png")
    except Exception as e:
        print(f"  Ошибка визуализации: {e}")

    # Итоговый вердикт
    print(f"\n{'━' * 70}")
    print("ВЕРДИКТ")
    print(f"{'━' * 70}")

    best = max(results, key=lambda r: r.sharpe_ratio())
    best_ret = max(results, key=lambda r: r.total_return)
    best_dd = min(results, key=lambda r: abs(r.max_drawdown()))

    print(f"\n  Лучший Sharpe:     {best.name} ({best.sharpe_ratio():.3f})")
    print(f"  Лучшая доходность: {best_ret.name} ({best_ret.total_return:+.2%})")
    print(f"  Наименьший DD:     {best_dd.name} ({best_dd.max_drawdown():.2%})")

    print(f"\n  Ключевая ставка ЦБ как фактор для инвестиционных решений:")
    composite = [r for r in results if r.name == "Composite"][0]
    if composite.sharpe_ratio() > bnh.sharpe_ratio():
        print(f"  ✓ Composite Sharpe ({composite.sharpe_ratio():.3f}) > "
              f"Buy & Hold Sharpe ({bnh.sharpe_ratio():.3f})")
        print(f"  → Ключевая ставка УЛУЧШАЕТ risk-adjusted доходность")
    else:
        print(f"  ✗ Composite Sharpe ({composite.sharpe_ratio():.3f}) ≤ "
              f"Buy & Hold Sharpe ({bnh.sharpe_ratio():.3f})")
        print(f"  → Ключевая ставка НЕ улучшает risk-adjusted доходность на данном периоде")

    if abs(composite.max_drawdown()) < abs(bnh.max_drawdown()) * 0.8:
        print(f"  ✓ Composite DD ({composite.max_drawdown():.2%}) значительно лучше "
              f"Buy & Hold DD ({bnh.max_drawdown():.2%})")
        print(f"  → Стратегия существенно снижает просадки")
    elif abs(composite.max_drawdown()) < abs(bnh.max_drawdown()):
        print(f"  ~ Composite DD ({composite.max_drawdown():.2%}) чуть лучше "
              f"Buy & Hold DD ({bnh.max_drawdown():.2%})")
    else:
        print(f"  ✗ Composite DD ({composite.max_drawdown():.2%}) хуже "
              f"Buy & Hold DD ({bnh.max_drawdown():.2%})")

    print(f"\n{'━' * 70}")
    print("БЭКТЕСТ ЗАВЕРШЁН")
    print(f"{'━' * 70}")

    return results


if __name__ == "__main__":
    main()
