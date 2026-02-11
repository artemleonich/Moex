"""
IslandForestModel — островная модель ансамбля случайных лесов.

Каждый остров — независимый RF/ExtraTrees с уникальными гиперпараметрами.
Кольцевая миграция лучших деревьев между островами снижает корреляцию ρ.
Взвешенное агрегирование предсказаний на основе валидационной точности.
"""

import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


def _fit_island(island, X, y):
    """Обучение одного острова (для joblib)."""
    island.fit(X, y)
    return island


class IslandForestModel(BaseEstimator, ClassifierMixin):
    """
    Островная модель ансамбля случайных лесов.

    Parameters
    ----------
    n_islands : int
        Количество островов (подпопуляций).
    trees_per_island : int
        Количество деревьев на острове.
    n_migrations : int
        Количество раундов миграции после обучения.
    migration_rate : float
        Доля мигрирующих деревьев (0, 1).
    val_fraction : float
        Доля данных для внутренней валидации (хронологическая).
    n_jobs : int
        Параллелизм (-1 = все ядра).
    random_state : int
        Зерно генератора.
    """

    ISLAND_CONFIGS = [
        {"max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
        {"max_depth": 15, "max_features": "log2", "min_samples_leaf": 2},
        {"max_depth": 20, "max_features": 0.5, "min_samples_leaf": 5},
        {"max_depth": 10, "max_features": "sqrt", "min_samples_leaf": 10},
        {"max_depth": None, "max_features": 0.3, "min_samples_leaf": 1},
        {"max_depth": 25, "max_features": "sqrt", "min_samples_leaf": 3},
    ]

    def __init__(
        self,
        n_islands: int = 6,
        trees_per_island: int = 50,
        n_migrations: int = 3,
        migration_rate: float = 0.1,
        val_fraction: float = 0.2,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_islands = n_islands
        self.trees_per_island = trees_per_island
        self.n_migrations = n_migrations
        self.migration_rate = migration_rate
        self.val_fraction = val_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _create_island(self, idx: int):
        """Создание острова с уникальными гиперпараметрами."""
        cfg = self.ISLAND_CONFIGS[idx % len(self.ISLAND_CONFIGS)]
        # Каждый 3-й остров — ExtraTrees для дополнительного разнообразия
        cls = ExtraTreesClassifier if idx % 3 == 2 else RandomForestClassifier
        return cls(
            n_estimators=self.trees_per_island,
            random_state=self.random_state + idx,
            n_jobs=1,
            **cfg,
        )

    def _score_trees(self, island, X_val, y_val) -> list[float]:
        """Оценка каждого дерева острова на валидационных данных."""
        scores = []
        for tree in island.estimators_:
            pred = tree.predict(X_val)
            scores.append(accuracy_score(y_val, pred))
        return scores

    def _migrate_ring(self, islands: list, X_val, y_val) -> list:
        """
        Кольцевая миграция: остров i отправляет лучших → остров (i+1) % K.
        Лучшие эмигранты → случайная замена (а не худших) для сохранения разнообразия.
        """
        n_mig = max(1, int(self.trees_per_island * self.migration_rate))
        all_scores = [self._score_trees(isl, X_val, y_val) for isl in islands]

        rng = np.random.RandomState(self.random_state)

        for i in range(len(islands)):
            src = i
            tgt = (i + 1) % len(islands)

            # Лучшие деревья источника
            src_ranked = np.argsort(all_scores[src])[::-1]
            best_trees = [
                copy.deepcopy(islands[src].estimators_[j])
                for j in src_ranked[:n_mig]
            ]

            # Случайная замена в приёмнике (не худших — для сохранения разнообразия)
            tgt_indices = rng.choice(
                len(islands[tgt].estimators_), size=n_mig, replace=False
            )
            for k, idx in enumerate(tgt_indices):
                islands[tgt].estimators_[idx] = best_trees[k]

        return islands

    def fit(self, X, y):
        """
        Обучение островной модели.

        1. Хронологическое разделение на train/val.
        2. Параллельное обучение всех островов.
        3. n_migrations раундов кольцевой миграции.
        4. Вычисление весов островов по валидационной точности.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Хронологическое разделение
        split = int(len(X_arr) * (1 - self.val_fraction))
        X_tr, X_val = X_arr[:split], X_arr[split:]
        y_tr, y_val = y_arr[:split], y_arr[split:]

        # Создание и параллельное обучение островов
        islands = [self._create_island(i) for i in range(self.n_islands)]
        self.islands_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_island)(isl, X_tr, y_tr) for isl in islands
        )

        # Миграционные раунды
        for _ in range(self.n_migrations):
            self.islands_ = self._migrate_ring(self.islands_, X_val, y_val)

        # Веса островов пропорциональны точности на валидации
        scores = [
            accuracy_score(y_val, isl.predict(X_val)) for isl in self.islands_
        ]
        total = sum(scores)
        if total > 0:
            self.weights_ = np.array([s / total for s in scores])
        else:
            self.weights_ = np.ones(self.n_islands) / self.n_islands

        self.classes_ = np.unique(y_arr)
        self.island_scores_ = scores
        return self

    def predict_proba(self, X):
        """Взвешенное усреднение вероятностей по островам."""
        X_arr = np.asarray(X)
        probas = np.array([isl.predict_proba(X_arr) for isl in self.islands_])
        return np.tensordot(self.weights_, probas, axes=([0], [0]))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_diversity_stats(self, X) -> dict:
        """
        Оценка разнообразия ансамбля через попарную корреляцию предсказаний.

        Returns
        -------
        dict с метриками разнообразия.
        """
        X_arr = np.asarray(X)
        predictions = np.array([isl.predict(X_arr) for isl in self.islands_])
        n = len(self.islands_)

        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return {
            "mean_pairwise_correlation": np.mean(correlations) if correlations else 0.0,
            "std_pairwise_correlation": np.std(correlations) if correlations else 0.0,
            "min_correlation": np.min(correlations) if correlations else 0.0,
            "max_correlation": np.max(correlations) if correlations else 0.0,
            "island_weights": self.weights_.tolist(),
            "island_val_scores": self.island_scores_,
        }
