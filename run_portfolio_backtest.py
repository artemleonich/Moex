#!/usr/bin/env python3
"""Портфельный бэктест островной модели на MOEX (100k руб)."""

import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from island_model.csv_loader import load_all_csv
from island_model.data_loader import MOEXDataLoader
from island_model.features import FeatureGenerator, TargetGenerator
from island_model.island_forest import IslandForestModel
from island_model.portfolio_backtester import (
    PortfolioBacktester,
    PortfolioBacktestResult,
    TradingCosts,
    run_buy_and_hold_portfolio,
)
from island_model.synthetic_data import TICKER_PARAMS, load_synthetic_data

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Конфигурация
INITIAL_CAPITAL = 100_000  # 100 тыс. руб.
TICKERS = ["SBER", "GAZP", "LKOH", "ROSN", "GMKN"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TARGET_HORIZON = 5

# Walk-Forward
TRAIN_DAYS = 252
RETRAIN_EVERY = 21
EMBARGO_DAYS = 5

# Островная модель
N_ISLANDS = 6
TREES_PER_ISLAND = 50
N_MIGRATIONS = 3
MIGRATION_RATE = 0.1

# Торговые издержки
COSTS = TradingCosts(
    broker_commission=0.0003,    # 0.03%
    exchange_fee=0.0001,         # 0.01%
    slippage_bps=5.0,            # ~5 б.п.
    execution_delay_days=1,      # 1 день задержки
)

LOT_SIZES = {
    "SBER": 10,
    "GAZP": 10,
    "LKOH": 1,
    "GMKN": 1,
    "ROSN": 1,
}


def load_data():
    """Загрузка данных: CSV, MOEX API или синтетика."""
    print("\nЗагрузка данных...")

    # Пробуем локальные CSV
    import os
    csv_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(csv_dir):
        print(f"  Загрузка из CSV ({csv_dir}):")
        ticker_data, usdrub, brent = load_all_csv(TICKERS, csv_dir)
        if ticker_data:
            return ticker_data, usdrub, brent

    # Пробуем MOEX ISS API
    loader = MOEXDataLoader()
    print(f"  Попытка MOEX ISS API: {TICKERS}")
    ticker_data = loader.load_multiple(TICKERS, START_DATE, END_DATE)

    usdrub, brent = None, None

    if ticker_data:
        try:
            usdrub = loader.load_usd_rub(START_DATE, END_DATE)
        except Exception:
            pass
        try:
            brent = loader.load_brent(START_DATE, END_DATE)
        except Exception:
            pass
    else:
        # Fallback: синтетические данные
        print("  API недоступен, синтетические данные\n")
        ticker_data, usdrub, brent = load_synthetic_data(TICKERS, START_DATE, END_DATE)

    return ticker_data, usdrub, brent


def prepare_datasets(ticker_data, usdrub, brent):
    """Генерация признаков и таргетов."""
    print("\nГенерация признаков...")

    feat_gen = FeatureGenerator(windows=[5, 10, 20, 60])
    tgt_gen = TargetGenerator(horizon=TARGET_HORIZON)

    datasets = {}

    for ticker, ohlcv in ticker_data.items():
        features = feat_gen.generate(ohlcv, usdrub=usdrub, brent=brent)
        target = tgt_gen.generate(ohlcv)
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        print(f"  {ticker}: {len(X)} дней, {X.shape[1]} признаков, "
              f"balance={y.mean():.1%}")

        datasets[ticker] = (X, y, ohlcv)

    return datasets


def create_models():
    """Создание моделей для сравнения."""
    models = {}

    models["IslandForest"] = IslandForestModel(
        n_islands=N_ISLANDS,
        trees_per_island=TREES_PER_ISLAND,
        n_migrations=N_MIGRATIONS,
        migration_rate=MIGRATION_RATE,
        val_fraction=0.2,
        n_jobs=-1,
        random_state=42,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=N_ISLANDS * TREES_PER_ISLAND,
        max_features="sqrt",
        min_samples_leaf=2,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_samples=30,
            random_state=42,
            verbose=-1,
        )

    return models


def run_model_backtest(model, model_name, datasets):
    """Запуск бэктеста для одной модели."""
    backtester = PortfolioBacktester(
        initial_capital=INITIAL_CAPITAL,
        train_days=TRAIN_DAYS,
        retrain_every=RETRAIN_EVERY,
        embargo_days=EMBARGO_DAYS,
        costs=COSTS,
        max_position_pct=0.30,
        lot_sizes=LOT_SIZES,
    )
    result = backtester.run(model, datasets, model_name=model_name)
    return result


def plot_equity_curves(results, output_path):
    """Графики equity и drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Equity Curves
    ax1 = axes[0]
    for name, res in results.items():
        if not res.equity_curve:
            continue
        dates = [s.date for s in res.equity_curve]
        equities = [s.equity for s in res.equity_curve]
        ax1.plot(dates, equities, label=name, linewidth=1.5)

    ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Start 100K")
    ax1.set_title("Portfolio Equity Curve (100 000 RUB)", fontsize=14)
    ax1.set_ylabel("Equity (RUB)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    for name, res in results.items():
        if not res.equity_curve:
            continue
        equities = np.array([s.equity for s in res.equity_curve])
        dates = [s.date for s in res.equity_curve]
        running_max = np.maximum.accumulate(equities)
        drawdown = (equities - running_max) / running_max * 100
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, label=name)

    ax2.set_title("Drawdown (%)", fontsize=12)
    ax2.set_ylabel("DD %")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Equity curve saved: {output_path}")


def print_trade_analysis(result, name):
    """Печать статистики по сделкам."""
    if not result.trades:
        print(f"  {name}: нет сделок")
        return

    trades = result.trades
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]

    avg_commission = np.mean([t.commission for t in trades])
    avg_slippage = np.mean([t.slippage for t in trades])

    tickers_traded = set(t.ticker for t in trades)
    print(f"\n  Анализ сделок — {name}:")
    print(f"    Всего сделок: {len(trades)} (BUY={len(buys)}, SELL={len(sells)})")
    print(f"    Средняя комиссия: {avg_commission:.2f} руб./сделку")
    print(f"    Среднее проскальзывание: {avg_slippage:.2f} руб./сделку")
    print(f"    Общие издержки: {result.total_commissions + result.total_slippage:.2f} руб.")
    print(f"    Тикеры: {', '.join(sorted(tickers_traded))}")


def main():
    print(f"Портфельный бэктест островной модели на MOEX")
    print(f"Портфель: {INITIAL_CAPITAL:,.0f} руб. | {START_DATE} -- {END_DATE}")
    print(f"Тикеры: {', '.join(TICKERS)}")
    print(f"Комиссия: {COSTS.total_commission:.2%}/сторону, "
          f"проскальзывание ~{COSTS.slippage_bps:.0f} б.п., "
          f"задержка {COSTS.execution_delay_days} день, "
          f"retrain каждые {RETRAIN_EVERY} дней")

    # Загрузка данных
    ticker_data, usdrub, brent = load_data()
    if not ticker_data:
        print("ОШИБКА: нет данных")
        sys.exit(1)

    # Признаки
    datasets = prepare_datasets(ticker_data, usdrub, brent)
    if not datasets:
        print("ОШИБКА: нет datasets")
        sys.exit(1)

    # Модели
    models = create_models()
    print(f"\nМодели: {', '.join(models.keys())} + Buy&Hold")

    # Бэктест
    print(f"\nЗапуск бэктеста...")

    results = {}
    for name, model in models.items():
        print(f"\n  {name}...", end=" ", flush=True)
        res = run_model_backtest(model, name, datasets)
        results[name] = res
        if res.equity_curve:
            print(f"OK — equity {res.final_equity:,.0f} руб. "
                  f"({res.total_return:+.2%}), trades={res.n_trades}")
        else:
            print("SKIP (нет данных)")

    # Buy & Hold
    print(f"\n  Buy & Hold...", end=" ", flush=True)
    bnh = run_buy_and_hold_portfolio(INITIAL_CAPITAL, datasets, COSTS, LOT_SIZES)
    results["Buy & Hold"] = bnh
    if bnh.equity_curve:
        print(f"OK — equity {bnh.final_equity:,.0f} руб. "
              f"({bnh.total_return:+.2%})")

    # Результаты
    print(f"\nИтоговые результаты:")

    summary_rows = []
    for name, res in results.items():
        s = res.summary()
        summary_rows.append({"Модель": name, **s})

    df = pd.DataFrame(summary_rows).set_index("Модель")
    print("\n" + df.to_string())

    # Анализ сделок
    for name, res in results.items():
        print_trade_analysis(res, name)

    # Влияние издержек
    print(f"\nВлияние издержек на P&L:")
    for name, res in results.items():
        pnl = res.final_equity - res.initial_capital
        costs_total = res.total_commissions + res.total_slippage
        costs_pct = costs_total / res.initial_capital * 100 if res.initial_capital > 0 else 0
        print(f"  {name:15s}: P&L={pnl:+10,.0f} руб. | "
              f"Издержки={costs_total:8,.0f} руб. ({costs_pct:.2f}% от капитала) | "
              f"P&L без издержек={pnl + costs_total:+10,.0f} руб.")

    # Графики
    print(f"\nСохранение графиков...")
    try:
        plot_equity_curves(results, "/home/user/Moex/equity_curves.png")
    except Exception as e:
        print(f"  Ошибка графика: {e}")

    print("\nБэктест завершён.")


if __name__ == "__main__":
    main()
