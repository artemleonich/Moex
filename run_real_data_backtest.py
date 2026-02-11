#!/usr/bin/env python3
"""
Портфельный бэктест Island Model на РЕАЛЬНЫХ данных MOEX.

Данные: SBER (1999-2024) и GAZP (2006-2024) — реальные дневные OHLCV
с Московской биржи (загружены из открытых источников).

Тестовый период: 2015-01-01 — 2024-12-31 (10 лет)
Модели: IslandForest, RandomForest, LightGBM, Buy & Hold
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from island_model.features import FeatureGenerator, TargetGenerator
from island_model.island_forest import IslandForestModel
from island_model.portfolio_backtester import (
    PortfolioBacktester,
    PortfolioBacktestResult,
    TradingCosts,
    run_buy_and_hold_portfolio,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000  # 100 тыс. руб.
TICKERS = ["SBER", "GAZP"]
START_DATE = "2015-01-01"
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
    broker_commission=0.0003,
    exchange_fee=0.0001,
    slippage_bps=5.0,
    execution_delay_days=1,
)

LOT_SIZES = {
    "SBER": 10,
    "GAZP": 10,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "downloaded_data", "csv_files")


def load_real_data():
    """Загрузка реальных данных MOEX из CSV."""
    print("\n[1/6] ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ MOEX")
    print("-" * 60)

    ticker_data = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_D1.csv")
        if not os.path.exists(path):
            print(f"  ОШИБКА: {path} не найден!")
            continue

        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        df.index.name = "begin"

        # Фильтрация по периоду
        df = df.loc[START_DATE:END_DATE]

        # Проверка OHLC-консистентности
        valid = (
            (df["high"] >= df[["open", "close"]].max(axis=1) - 0.01) &
            (df["low"] <= df[["open", "close"]].min(axis=1) + 0.01)
        )
        n_invalid = (~valid).sum()

        print(f"  {ticker}: {len(df)} дней "
              f"({df.index[0].date()} — {df.index[-1].date()}) "
              f"close {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}"
              f"{f'  ⚠ {n_invalid} OHLC errors' if n_invalid > 0 else ''}")

        ticker_data[ticker] = df[["open", "high", "low", "close", "volume"]]

    # Проверка закрытия биржи в феврале-марте 2022
    for ticker, df in ticker_data.items():
        moex_closed_start = pd.Timestamp("2022-02-25")
        moex_closed_end = pd.Timestamp("2022-03-24")
        closed_days = df.loc[moex_closed_start:moex_closed_end]
        if len(closed_days) > 0:
            print(f"\n  ⚠ {ticker}: {len(closed_days)} дней в период закрытия MOEX "
                  f"(25.02-24.03.2022) — реальные данные")
        else:
            print(f"  ✓ {ticker}: Нет торговли в период закрытия MOEX")

    return ticker_data


def prepare_datasets(ticker_data):
    """Генерация признаков и таргетов."""
    print("\n[2/6] ГЕНЕРАЦИЯ ПРИЗНАКОВ")
    print("-" * 60)

    feat_gen = FeatureGenerator(windows=[5, 10, 20, 60])
    tgt_gen = TargetGenerator(horizon=TARGET_HORIZON)

    datasets = {}
    for ticker, ohlcv in ticker_data.items():
        features = feat_gen.generate(ohlcv, usdrub=None, brent=None)
        target = tgt_gen.generate(ohlcv)
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        # Заполняем NaN средними (для межрыночных признаков которые будут NaN)
        X = X.fillna(0)

        print(f"  {ticker}: {len(X)} дней, {X.shape[1]} признаков, "
              f"balance={y.mean():.1%}")

        datasets[ticker] = (X, y, ohlcv)

    return datasets


def create_models():
    """Модели для сравнения."""
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
    """Запуск портфельного бэктеста."""
    backtester = PortfolioBacktester(
        initial_capital=INITIAL_CAPITAL,
        train_days=TRAIN_DAYS,
        retrain_every=RETRAIN_EVERY,
        embargo_days=EMBARGO_DAYS,
        costs=COSTS,
        max_position_pct=0.50,  # 50% max на тикер (только 2 тикера)
        lot_sizes=LOT_SIZES,
    )
    return backtester.run(model, datasets, model_name=model_name)


def plot_equity_curves(results, output_path):
    """Визуализация equity curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    for name, res in results.items():
        if not res.equity_curve:
            continue
        dates = [s.date for s in res.equity_curve]
        equities = [s.equity for s in res.equity_curve]
        ax1.plot(dates, equities, label=name, linewidth=1.5)

    ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Start 100K")
    # Отмечаем период закрытия MOEX
    ax1.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"),
                color="red", alpha=0.1, label="MOEX closed")
    ax1.set_title("Portfolio Equity — REAL MOEX Data (SBER + GAZP)", fontsize=14)
    ax1.set_ylabel("Equity (RUB)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for name, res in results.items():
        if not res.equity_curve:
            continue
        equities = np.array([s.equity for s in res.equity_curve])
        dates = [s.date for s in res.equity_curve]
        running_max = np.maximum.accumulate(equities)
        drawdown = (equities - running_max) / running_max * 100
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, label=name)

    ax2.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"),
                color="red", alpha=0.1)
    ax2.set_title("Drawdown (%)", fontsize=12)
    ax2.set_ylabel("DD %")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Equity curve saved: {output_path}")


def main():
    print("=" * 70)
    print("ПОРТФЕЛЬНЫЙ БЭКТЕСТ НА РЕАЛЬНЫХ ДАННЫХ MOEX")
    print(f"Портфель: {INITIAL_CAPITAL:,.0f} руб.")
    print(f"Период: {START_DATE} — {END_DATE}")
    print(f"Тикеры: {', '.join(TICKERS)}")
    print(f"Комиссия: {COSTS.total_commission:.2%} за сторону")
    print(f"Проскальзывание: ~{COSTS.slippage_bps:.0f} б.п.")
    print(f"Задержка исполнения: {COSTS.execution_delay_days} день")
    print(f"Переобучение: каждые {RETRAIN_EVERY} дней")
    print("=" * 70)

    # 1. Загрузка данных
    ticker_data = load_real_data()
    if not ticker_data:
        print("ОШИБКА: нет данных")
        sys.exit(1)

    # 2. Признаки
    datasets = prepare_datasets(ticker_data)

    # 3. Модели
    models = create_models()
    print(f"\n[3/6] МОДЕЛИ: {', '.join(models.keys())} + Buy&Hold")

    # 4. Бэктест
    print(f"\n[4/6] ПОРТФЕЛЬНЫЙ БЭКТЕСТ")
    print("=" * 60)

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

    # 5. Результаты
    print(f"\n[5/6] ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    summary_rows = []
    for name, res in results.items():
        s = res.summary()
        summary_rows.append({"Модель": name, **s})

    df = pd.DataFrame(summary_rows).set_index("Модель")
    print("\n" + df.to_string())

    # Анализ сделок
    print(f"\n{'─' * 70}")
    for name, res in results.items():
        if not res.trades:
            print(f"  {name}: нет сделок")
            continue
        buys = [t for t in res.trades if t.side == "BUY"]
        sells = [t for t in res.trades if t.side == "SELL"]
        avg_commission = np.mean([t.commission for t in res.trades])
        print(f"\n  {name}: {len(res.trades)} сделок "
              f"(BUY={len(buys)}, SELL={len(sells)}), "
              f"средняя комиссия={avg_commission:.2f} руб.")

    # Анализ кризиса 2022
    print(f"\n{'─' * 70}")
    print("АНАЛИЗ КРИЗИСА ФЕВРАЛЯ 2022:")
    for name, res in results.items():
        if not res.equity_curve:
            continue
        # Найдём equity до и после кризиса
        pre_crisis = [s for s in res.equity_curve
                      if s.date <= pd.Timestamp("2022-02-24")]
        post_crisis = [s for s in res.equity_curve
                       if s.date >= pd.Timestamp("2022-03-25") and s.date <= pd.Timestamp("2022-04-30")]
        if pre_crisis and post_crisis:
            eq_before = pre_crisis[-1].equity
            eq_after = post_crisis[0].equity if post_crisis else eq_before
            loss_pct = (eq_after - eq_before) / eq_before * 100
            print(f"  {name:15s}: equity до={eq_before:,.0f} → после={eq_after:,.0f} "
                  f"({loss_pct:+.1f}%)")

    # Издержки vs P&L
    print(f"\n{'─' * 70}")
    print("ВЛИЯНИЕ ИЗДЕРЖЕК НА P&L:")
    for name, res in results.items():
        pnl = res.final_equity - res.initial_capital
        costs_total = res.total_commissions + res.total_slippage
        costs_pct = costs_total / res.initial_capital * 100
        print(f"  {name:15s}: P&L={pnl:+10,.0f} руб. | "
              f"Издержки={costs_total:8,.0f} руб. ({costs_pct:.2f}%) | "
              f"P&L без издержек={pnl + costs_total:+10,.0f} руб.")

    # Сравнение с банковским депозитом
    print(f"\n{'─' * 70}")
    print("СРАВНЕНИЕ С БАНКОВСКИМ ДЕПОЗИТОМ:")
    # Средняя ставка ЦБ за 2015-2024 ~10%
    avg_rate = 0.10
    n_years = 10
    deposit_final = INITIAL_CAPITAL * (1 + avg_rate) ** n_years
    print(f"  Банковский депозит (~10% средняя): {INITIAL_CAPITAL:,.0f} → {deposit_final:,.0f} руб. "
          f"(+{(deposit_final/INITIAL_CAPITAL - 1)*100:.0f}%)")
    for name, res in results.items():
        diff = res.final_equity - deposit_final
        print(f"  {name:15s} vs депозит: {diff:+,.0f} руб.")

    # 6. Графики
    print(f"\n[6/6] ВИЗУАЛИЗАЦИЯ")
    print("-" * 60)
    try:
        plot_equity_curves(results, "/home/user/Moex/equity_real_data.png")
    except Exception as e:
        print(f"  Ошибка графика: {e}")

    print(f"\n{'━' * 70}")
    print("БЭКТЕСТ НА РЕАЛЬНЫХ ДАННЫХ ЗАВЕРШЁН")
    print(f"{'━' * 70}")


if __name__ == "__main__":
    main()
