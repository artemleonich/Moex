#!/usr/bin/env python3
"""
Бэктест островной модели случайных деревьев на исторических данных MOEX.

Тестирует:
1. IslandForestModel (6 островов, кольцевая миграция)
2. Стандартный RandomForest (бенчмарк)
3. LightGBM (бенчмарк)
4. Buy & Hold (бенчмарк)

Walk-Forward валидация: 252 дня обучение → 63 дня тест → сдвиг.
Тикеры: SBER, GAZP, LKOH, GMKN, ROSN.
Период: 2018-01-01 — 2025-12-31.
"""

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from island_model.backtester import (
    BacktestResult,
    WalkForwardBacktester,
    print_comparison,
    run_buy_and_hold,
)
from island_model.data_loader import MOEXDataLoader
from island_model.features import FeatureGenerator, TargetGenerator
from island_model.island_forest import IslandForestModel
from island_model.synthetic_data import load_synthetic_data

# Попробовать импортировать LightGBM
try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM не установлен, пропускаем бенчмарк LGBM")


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN"]
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TARGET_HORIZON = 5  # forward returns 5 дней

# Walk-Forward параметры
TRAIN_DAYS = 252  # 1 год обучения
TEST_DAYS = 63  # ~1 квартал тест
EMBARGO_DAYS = 5  # зазор между train и test

# Островная модель
N_ISLANDS = 6
TREES_PER_ISLAND = 50
N_MIGRATIONS = 3
MIGRATION_RATE = 0.1


def load_data():
    """Загрузка данных с MOEX ISS API, fallback на синтетические данные."""
    print("\n[1/5] ЗАГРУЗКА ДАННЫХ")
    print("-" * 50)

    # Пробуем реальный API
    loader = MOEXDataLoader()
    print(f"Попытка загрузки с MOEX ISS API: {TICKERS}")
    ticker_data = loader.load_multiple(TICKERS, START_DATE, END_DATE)

    usdrub = None
    brent = None

    if ticker_data:
        # API доступен — загружаем межрыночные данные
        print("Загрузка USD/RUB...")
        try:
            usdrub = loader.load_usd_rub(START_DATE, END_DATE)
            print(f"  USD/RUB: {len(usdrub)} rows")
        except Exception as e:
            print(f"  USD/RUB: SKIP ({e})")

        print("Загрузка Brent...")
        try:
            brent = loader.load_brent(START_DATE, END_DATE)
            print(f"  Brent: {len(brent)} rows")
        except Exception as e:
            print(f"  Brent: SKIP ({e})")
    else:
        # API недоступен — используем синтетические данные
        print("\n  MOEX ISS API недоступен. Переключение на синтетические данные.")
        print("  Синтетические данные имитируют реалистичное поведение акций MOEX")
        print("  (режимные переключения, GARCH-волатильность, корреляции).\n")
        ticker_data, usdrub, brent = load_synthetic_data(TICKERS, START_DATE, END_DATE)

    return ticker_data, usdrub, brent


def prepare_features(ticker_data, usdrub, brent):
    """Генерация признаков и целевой переменной для каждого тикера."""
    print("\n[2/5] ГЕНЕРАЦИЯ ПРИЗНАКОВ")
    print("-" * 50)

    feat_gen = FeatureGenerator(windows=[5, 10, 20, 60])
    tgt_gen = TargetGenerator(horizon=TARGET_HORIZON)

    datasets = {}

    for ticker, df in ticker_data.items():
        print(f"\n  {ticker}:")
        print(f"    Сырых данных: {len(df)} строк ({df.index[0]} — {df.index[-1]})")

        # Признаки
        features = feat_gen.generate(df, usdrub=usdrub, brent=brent)
        print(f"    Признаков: {features.shape[1]}")

        # Целевая переменная
        target = tgt_gen.generate(df)

        # Дневные доходности для расчёта метрик стратегии
        daily_returns = df["close"].pct_change()

        # Выравнивание индексов
        common_idx = features.index.intersection(target.dropna().index).intersection(
            daily_returns.dropna().index
        )
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        ret = daily_returns.loc[common_idx]

        print(f"    Итого: {len(X)} строк после выравнивания")
        print(f"    Баланс классов: {y.mean():.2%} положительных")
        print(f"    Признаки: {list(X.columns)}")

        if len(X) < TRAIN_DAYS + EMBARGO_DAYS + TEST_DAYS:
            print(f"    SKIP — недостаточно данных для Walk-Forward")
            continue

        datasets[ticker] = (X, y, ret)

    return datasets


def create_models():
    """Создание моделей для сравнения."""
    models = {
        "IslandForest": IslandForestModel(
            n_islands=N_ISLANDS,
            trees_per_island=TREES_PER_ISLAND,
            n_migrations=N_MIGRATIONS,
            migration_rate=MIGRATION_RATE,
            val_fraction=0.2,
            n_jobs=-1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ISLANDS * TREES_PER_ISLAND,  # то же число деревьев
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=63,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )

    return models


def run_backtest_for_ticker(ticker, X, y, returns, models):
    """Запуск Walk-Forward бэктеста для одного тикера."""
    print(f"\n  Бэктест {ticker}")
    print(f"  {'─' * 40}")

    backtester = WalkForwardBacktester(
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        embargo_days=EMBARGO_DAYS,
    )

    results = []

    for model_name, model in models.items():
        print(f"    {model_name}...", end=" ", flush=True)
        result = backtester.run(model, X, y, returns, model_name=model_name)
        results.append(result)
        print(f"OK ({len(result.folds)} folds, acc={result.mean_accuracy:.4f})")

    # Buy & Hold бенчмарк: собираем тестовые периоды из первого результата
    if results:
        test_indices = []
        n = len(X)
        start = 0
        while start + TRAIN_DAYS + EMBARGO_DAYS + TEST_DAYS <= n:
            test_start = start + TRAIN_DAYS + EMBARGO_DAYS
            test_end = min(test_start + TEST_DAYS, n)
            test_indices.append((test_start, test_end))
            start += TEST_DAYS

        bnh_result = run_buy_and_hold(returns, test_indices)
        results.append(bnh_result)
        print(f"    Buy & Hold... OK ({len(bnh_result.folds)} folds)")

    return results


def analyze_diversity(ticker, X, y):
    """Анализ разнообразия островной модели vs стандартного RF."""
    print(f"\n  Анализ разнообразия ({ticker})")
    print(f"  {'─' * 40}")

    # Обучаем на всех данных для анализа
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train = y.iloc[:split]

    # Island Forest
    island_model = IslandForestModel(
        n_islands=N_ISLANDS,
        trees_per_island=TREES_PER_ISLAND,
        n_migrations=N_MIGRATIONS,
        migration_rate=MIGRATION_RATE,
        n_jobs=-1,
        random_state=42,
    )
    island_model.fit(X_train, y_train)
    island_stats = island_model.get_diversity_stats(X_test)

    # Standard RF (то же число деревьев)
    rf = RandomForestClassifier(
        n_estimators=N_ISLANDS * TREES_PER_ISLAND,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Считаем попарную корреляцию для RF — группируем деревья по «псевдо-островам»
    rf_preds = np.array([tree.predict(X_test.values) for tree in rf.estimators_])
    n_trees = len(rf.estimators_)
    rf_correlations = []
    # Семплируем пары для скорости
    rng = np.random.RandomState(42)
    n_pairs = min(500, n_trees * (n_trees - 1) // 2)
    for _ in range(n_pairs):
        i, j = rng.choice(n_trees, size=2, replace=False)
        corr = np.corrcoef(rf_preds[i], rf_preds[j])[0, 1]
        if not np.isnan(corr):
            rf_correlations.append(corr)

    rf_mean_corr = np.mean(rf_correlations) if rf_correlations else 0.0

    print(f"    IslandForest средняя попарная корреляция островов: "
          f"{island_stats['mean_pairwise_correlation']:.4f}")
    print(f"    RandomForest средняя попарная корреляция деревьев:  "
          f"{rf_mean_corr:.4f}")
    print(f"    IslandForest веса островов: "
          f"{[f'{w:.3f}' for w in island_stats['island_weights']]}")
    print(f"    IslandForest val scores:    "
          f"{[f'{s:.4f}' for s in island_stats['island_val_scores']]}")

    return {
        "island_mean_corr": island_stats["mean_pairwise_correlation"],
        "rf_mean_corr": rf_mean_corr,
        "island_weights": island_stats["island_weights"],
    }


def main():
    print("=" * 80)
    print("БЭКТЕСТ ОСТРОВНОЙ МОДЕЛИ СЛУЧАЙНЫХ ДЕРЕВЬЕВ НА MOEX")
    print(f"Период: {START_DATE} — {END_DATE}")
    print(f"Тикеры: {', '.join(TICKERS)}")
    print(f"Target: forward return {TARGET_HORIZON}d > 0")
    print(f"Walk-Forward: train={TRAIN_DAYS}d, test={TEST_DAYS}d, embargo={EMBARGO_DAYS}d")
    print(f"IslandForest: {N_ISLANDS} островов × {TREES_PER_ISLAND} деревьев, "
          f"{N_MIGRATIONS} миграций")
    print("=" * 80)

    # Загрузка данных
    ticker_data, usdrub, brent = load_data()

    if not ticker_data:
        print("\nОШИБКА: не удалось загрузить данные ни для одного тикера")
        sys.exit(1)

    # Генерация признаков
    datasets = prepare_features(ticker_data, usdrub, brent)

    if not datasets:
        print("\nОШИБКА: недостаточно данных для бэктеста")
        sys.exit(1)

    # Создание моделей
    models = create_models()
    print(f"\n[3/5] МОДЕЛИ ДЛЯ СРАВНЕНИЯ: {', '.join(models.keys())} + Buy&Hold")

    # Walk-Forward бэктест
    print(f"\n[4/5] WALK-FORWARD БЭКТЕСТ")
    print("=" * 50)

    all_results = {}
    all_diversity = {}

    for ticker, (X, y, ret) in datasets.items():
        results = run_backtest_for_ticker(ticker, X, y, ret, models)
        all_results[ticker] = results

        # Анализ разнообразия
        diversity = analyze_diversity(ticker, X, y)
        all_diversity[ticker] = diversity

    # Итоговое сравнение
    print(f"\n[5/5] ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    for ticker, results in all_results.items():
        print(f"\n{'━' * 80}")
        print(f"ТИКЕР: {ticker}")
        print(f"{'━' * 80}")
        comparison_df = print_comparison(results)

    # Агрегированные результаты по всем тикерам
    print(f"\n{'━' * 80}")
    print("АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ ТИКЕРАМ")
    print(f"{'━' * 80}")

    model_names = list(models.keys()) + ["Buy & Hold"]
    agg = {name: [] for name in model_names}

    for ticker, results in all_results.items():
        for res in results:
            if res.model_name in agg:
                agg[res.model_name].append(res.summary())

    agg_rows = []
    for name in model_names:
        if not agg[name]:
            continue
        summaries = agg[name]
        row = {"model": name}
        for key in summaries[0]:
            if key == "model":
                continue
            vals = [s[key] for s in summaries]
            row[f"avg_{key}"] = round(np.mean(vals), 4)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).set_index("model")
    print(agg_df.to_string())

    # Сводка разнообразия
    print(f"\n{'━' * 80}")
    print("АНАЛИЗ РАЗНООБРАЗИЯ (ρ — попарная корреляция)")
    print(f"{'━' * 80}")
    for ticker, div in all_diversity.items():
        print(f"  {ticker}: IslandForest ρ={div['island_mean_corr']:.4f}, "
              f"RandomForest ρ={div['rf_mean_corr']:.4f}, "
              f"Δρ={div['rf_mean_corr'] - div['island_mean_corr']:+.4f}")

    print(f"\n{'━' * 80}")
    print("БЭКТЕСТ ЗАВЕРШЁН")
    print(f"{'━' * 80}")


if __name__ == "__main__":
    main()
