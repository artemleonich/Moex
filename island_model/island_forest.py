# Island model ensemble of random forests with ring migration

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
    """Островная модель ансамбля случайных лесов с кольцевой миграцией деревьев."""

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
        n_islands=6,
        trees_per_island=50,
        n_migrations=3,
        migration_rate=0.1,
        val_fraction=0.2,
        n_jobs=-1,
        random_state=42,
    ):
        self.n_islands = n_islands
        self.trees_per_island = trees_per_island
        self.n_migrations = n_migrations
        self.migration_rate = migration_rate
        self.val_fraction = val_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _create_island(self, idx):
        """Создание острова с уникальными гиперпараметрами."""
        # Deep-copy the config dict so that mutating ISLAND_CONFIGS (either
        # directly on the class or on one instance) cannot bleed into
        # other instances, fit calls, or future instantiations. The
        # class-level ISLAND_CONFIGS is a shared mutable list of dicts;
        # without this copy sklearn's set_params() stores references to
        # the original dicts, which is a foot-gun during hyperparameter
        # search and any code that customises islands per-instance.
        cfg = copy.deepcopy(self.ISLAND_CONFIGS[idx % len(self.ISLAND_CONFIGS)])
        # Каждый 3-й остров — ExtraTrees для дополнительного разнообразия
        cls = ExtraTreesClassifier if idx % 3 == 2 else RandomForestClassifier
        return cls(
            n_estimators=self.trees_per_island,
            random_state=self.random_state + idx,
            n_jobs=1,
            **cfg,
        )

    def _score_trees(self, island, X_val, y_val):
        """Оценка каждого дерева острова на валидации."""
        scores = []
        for tree in island.estimators_:
            pred = tree.predict(X_val)
            scores.append(accuracy_score(y_val, pred))
        return scores

    def _migrate_ring(self, islands, X_val, y_val):
        """Кольцевая миграция: лучшие деревья острова i уходят в остров (i+1) % K."""
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
            tgt_idx = rng.choice(
                len(islands[tgt].estimators_), size=n_mig, replace=False
            )
            for k, idx in enumerate(tgt_idx):
                islands[tgt].estimators_[idx] = best_trees[k]

        return islands

    def fit(self, X, y):
        """Обучение: split -> fit islands -> migrate -> compute weights."""
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

    def get_diversity_stats(self, X):
        """Попарная корреляция предсказаний островов — метрики разнообразия."""
        X_arr = np.asarray(X)
        preds = np.array([isl.predict(X_arr) for isl in self.islands_])
        n = len(self.islands_)

        corrs = []
        for i in range(n):
            for j in range(i + 1, n):
                c = np.corrcoef(preds[i], preds[j])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)

        return {
            "mean_pairwise_correlation": np.mean(corrs) if corrs else 0.0,
            "std_pairwise_correlation": np.std(corrs) if corrs else 0.0,
            "min_correlation": np.min(corrs) if corrs else 0.0,
            "max_correlation": np.max(corrs) if corrs else 0.0,
            "island_weights": self.weights_.tolist(),
            "island_val_scores": self.island_scores_,
        }
